"""
Embedding Pipeline

Orchestrates embedding of:
  - Passages (from Passage model instances or dicts)
  - Entity names + aliases (from Character/Place/Object models)
  - Lore rules (from LoreRule model instances)
  - Scene templates (from SceneTemplate model instances)

Key features:
  - Incremental: skips already-embedded items (checked via Chroma IDs + DuckDB log)
  - Batched: configurable batch size
  - Progress reporting via callback or tqdm
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .embedder import Embedder
from .store import (
    VectorStore,
    COLLECTION_PASSAGES,
    COLLECTION_ENTITIES,
    COLLECTION_LORE_RULES,
    COLLECTION_SCENE_TEMPLATES,
)
from .analytics import PassageAnalytics

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Summary of an embedding pipeline run."""
    collection: str
    total_items: int = 0
    already_embedded: int = 0
    newly_embedded: int = 0
    errors: int = 0
    error_ids: list[str] = field(default_factory=list)

    @property
    def skipped(self) -> int:
        return self.already_embedded

    def __str__(self) -> str:
        return (
            f"{self.collection}: {self.newly_embedded} embedded, "
            f"{self.already_embedded} skipped, "
            f"{self.errors} errors "
            f"(total {self.total_items})"
        )


class EmbeddingPipeline:
    """
    Orchestrates all embedding builds for BGA.

    Usage:
        pipeline = EmbeddingPipeline(
            store=VectorStore("data/chroma"),
            analytics=PassageAnalytics("data/analytics.duckdb"),
        )
        result = pipeline.build_passages(passages, incremental=True)
    """

    def __init__(
        self,
        store: VectorStore,
        analytics: Optional[PassageAnalytics] = None,
        embedder: Optional[Embedder] = None,
        batch_size: int = 64,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        self.store = store
        self.analytics = analytics
        self.embedder = embedder or Embedder.from_settings()
        self.batch_size = batch_size
        self.progress_callback = progress_callback  # fn(stage, current, total)

    # ------------------------------------------------------------------
    # Passage pipeline
    # ------------------------------------------------------------------

    def build_passages(
        self,
        passages: list[Any],  # Passage model instances or dicts
        incremental: bool = True,
    ) -> BuildResult:
        """
        Embed all passages and store in Chroma + DuckDB.

        Each passage must have: id, text, book, chapter, chapter_num,
        paragraph_num, sentence_num, word_count, avg_sentence_length,
        passive_ratio, dialogue_density, archaic_word_count, story_era,
        tolkien_register, is_dialogue.

        When using Passage model instances, all fields are available.
        When using dicts, missing fields default to sensible values.
        """
        result = BuildResult(collection=COLLECTION_PASSAGES, total_items=len(passages))

        # Determine already-embedded IDs
        existing_ids = self.store.get_all_ids(COLLECTION_PASSAGES) if incremental else set()
        result.already_embedded = sum(1 for p in passages if _get_id(p) in existing_ids)

        # Filter to only new items
        to_embed = [p for p in passages if _get_id(p) not in existing_ids] if incremental else passages
        logger.info("Passages: %d total, %d to embed, %d skipped",
                    len(passages), len(to_embed), result.already_embedded)

        # Process in batches
        for batch_start in range(0, len(to_embed), self.batch_size):
            batch = to_embed[batch_start : batch_start + self.batch_size]
            ids = [_get_id(p) for p in batch]
            texts = [_get_text(p) for p in batch]
            metas = [_passage_metadata(p) for p in batch]

            try:
                embeddings = self.embedder.embed(texts)
            except Exception as exc:
                logger.error("Embedding error in batch %d: %s", batch_start, exc)
                result.errors += len(batch)
                result.error_ids.extend(ids)
                continue

            # Store in Chroma
            self.store.upsert_passages(ids, embeddings, texts, metas)

            # Store metrics in DuckDB
            if self.analytics:
                passage_dicts = [_passage_to_dict(p) for p in batch]
                self.analytics.upsert_passages_bulk(passage_dicts)
                self.analytics.log_embeddings_bulk(ids, COLLECTION_PASSAGES, self.embedder.model_name)

            result.newly_embedded += len(batch)
            if self.progress_callback:
                self.progress_callback(COLLECTION_PASSAGES, batch_start + len(batch), len(to_embed))

        logger.info("Passages done: %s", result)
        return result

    # ------------------------------------------------------------------
    # Entity names pipeline
    # ------------------------------------------------------------------

    def build_entity_names(
        self,
        entities: list[Any],  # Character/Place/Object model instances or dicts
        incremental: bool = True,
    ) -> BuildResult:
        """
        Embed entity canonical names + all aliases.

        Each entity entry: id, canonical_name, aliases (list[str]),
        entity_type (character/place/object).

        Each alias gets its OWN embedding, keyed as {entity_id}::{alias_index},
        but carries the canonical entity_id in metadata.
        """
        # Expand to (embed_id, text, metadata) tuples
        rows = []
        for entity in entities:
            eid = _get_id(entity)
            canonical = _get_field(entity, "canonical_name", eid)
            aliases = _get_field(entity, "aliases", [])
            etype = _get_field(entity, "entity_type", _get_field(entity, "type", "entity"))

            # Canonical name
            rows.append((eid, canonical, {"entity_id": eid, "canonical_name": canonical,
                                           "is_alias": False, "entity_type": etype}))
            # Aliases
            for i, alias in enumerate(aliases):
                if alias and alias != canonical:
                    alias_id = f"{eid}::alias::{i}"
                    rows.append((alias_id, alias, {"entity_id": eid, "canonical_name": canonical,
                                                    "is_alias": True, "alias_text": alias,
                                                    "entity_type": etype}))

        result = BuildResult(collection=COLLECTION_ENTITIES, total_items=len(rows))

        existing_ids = self.store.get_all_ids(COLLECTION_ENTITIES) if incremental else set()
        to_embed = [(rid, txt, meta) for rid, txt, meta in rows if rid not in existing_ids]
        result.already_embedded = len(rows) - len(to_embed)

        for batch_start in range(0, len(to_embed), self.batch_size):
            batch = to_embed[batch_start : batch_start + self.batch_size]
            ids = [r[0] for r in batch]
            texts = [r[1] for r in batch]
            metas = [r[2] for r in batch]

            try:
                embeddings = self.embedder.embed(texts)
            except Exception as exc:
                logger.error("Entity embedding error: %s", exc)
                result.errors += len(batch)
                continue

            self.store.upsert_entity_names(ids, embeddings, texts, metas)

            if self.analytics:
                self.analytics.log_embeddings_bulk(ids, COLLECTION_ENTITIES, self.embedder.model_name)

            result.newly_embedded += len(batch)

        logger.info("Entities done: %s", result)
        return result

    # ------------------------------------------------------------------
    # Lore rules pipeline
    # ------------------------------------------------------------------

    def build_lore_rules(
        self,
        rules: list[Any],  # LoreRule model instances or dicts
        incremental: bool = True,
    ) -> BuildResult:
        """
        Embed lore rule statements.

        Each rule: id, statement (or description), category (optional).
        """
        result = BuildResult(collection=COLLECTION_LORE_RULES, total_items=len(rules))

        existing_ids = self.store.get_all_ids(COLLECTION_LORE_RULES) if incremental else set()
        to_embed = [r for r in rules if _get_id(r) not in existing_ids] if incremental else rules
        result.already_embedded = len(rules) - len(to_embed)

        for batch_start in range(0, len(to_embed), self.batch_size):
            batch = to_embed[batch_start : batch_start + self.batch_size]
            ids = [_get_id(r) for r in batch]
            # Statement text: try 'statement', then 'description', then 'text'
            texts = [
                _get_field(r, "statement",
                    _get_field(r, "description",
                        _get_field(r, "text", str(r))))
                for r in batch
            ]
            metas = [{"rule_id": _get_id(r),
                      "category": _get_field(r, "category", ""),
                      "severity": _get_field(r, "severity", "")}
                     for r in batch]

            try:
                embeddings = self.embedder.embed(texts)
            except Exception as exc:
                logger.error("Lore rule embedding error: %s", exc)
                result.errors += len(batch)
                continue

            self.store.upsert_lore_rules(ids, embeddings, texts, metas)

            if self.analytics:
                self.analytics.log_embeddings_bulk(ids, COLLECTION_LORE_RULES, self.embedder.model_name)

            result.newly_embedded += len(batch)

        logger.info("Lore rules done: %s", result)
        return result

    # ------------------------------------------------------------------
    # Scene templates pipeline
    # ------------------------------------------------------------------

    def build_scene_templates(
        self,
        templates: list[Any],  # SceneTemplate model instances or dicts
        incremental: bool = True,
    ) -> BuildResult:
        """
        Embed scene template descriptions.

        Each template: id, register (ProseRegister), structural_pattern,
        avg_sentence_length, passive_ratio.
        """
        result = BuildResult(collection=COLLECTION_SCENE_TEMPLATES, total_items=len(templates))

        existing_ids = self.store.get_all_ids(COLLECTION_SCENE_TEMPLATES) if incremental else set()
        to_embed = [t for t in templates if _get_id(t) not in existing_ids] if incremental else templates
        result.already_embedded = len(templates) - len(to_embed)

        for batch_start in range(0, len(to_embed), self.batch_size):
            batch = to_embed[batch_start : batch_start + self.batch_size]
            ids = [_get_id(t) for t in batch]
            # Build a description string from template fields
            texts = [_template_description(t) for t in batch]
            metas = [{
                "template_id": _get_id(t),
                "register": str(_get_field(t, "register", "")),
                "avg_sentence_length": float(_get_field(t, "avg_sentence_length", 0.0)),
            } for t in batch]

            try:
                embeddings = self.embedder.embed(texts)
            except Exception as exc:
                logger.error("Template embedding error: %s", exc)
                result.errors += len(batch)
                continue

            self.store.upsert_scene_templates(ids, embeddings, texts, metas)

            if self.analytics:
                self.analytics.log_embeddings_bulk(ids, COLLECTION_SCENE_TEMPLATES, self.embedder.model_name)

            result.newly_embedded += len(batch)

        logger.info("Templates done: %s", result)
        return result

    # ------------------------------------------------------------------
    # High-level "build all" helper
    # ------------------------------------------------------------------

    def build_all(
        self,
        passages: Optional[list] = None,
        entities: Optional[list] = None,
        lore_rules: Optional[list] = None,
        scene_templates: Optional[list] = None,
        incremental: bool = True,
    ) -> dict[str, BuildResult]:
        """
        Embed all provided data. Returns dict of collection → BuildResult.
        """
        results = {}
        if passages is not None:
            results[COLLECTION_PASSAGES] = self.build_passages(passages, incremental)
        if entities is not None:
            results[COLLECTION_ENTITIES] = self.build_entity_names(entities, incremental)
        if lore_rules is not None:
            results[COLLECTION_LORE_RULES] = self.build_lore_rules(lore_rules, incremental)
        if scene_templates is not None:
            results[COLLECTION_SCENE_TEMPLATES] = self.build_scene_templates(scene_templates, incremental)
        return results


# ------------------------------------------------------------------
# Search helpers (combined embedder + store)
# ------------------------------------------------------------------

class SemanticSearch:
    """
    High-level semantic search combining Embedder + VectorStore.

    Usage:
        searcher = SemanticSearch(store, embedder)
        results = searcher.search_passages("grief for lost beauty", limit=10)
        entity = searcher.entity_match("the grey pilgrim")
    """

    def __init__(self, store: VectorStore, embedder: Optional[Embedder] = None) -> None:
        self.store = store
        self.embedder = embedder or Embedder.from_settings()

    def search_passages(
        self,
        query: str,
        limit: int = 10,
        book: Optional[str] = None,
    ):
        """Find passages semantically similar to the query."""
        q_vec = self.embedder.embed_one(query)
        return self.store.search_passages(q_vec, limit=limit, book=book)

    def search_similar_passage(
        self,
        passage_id: str,
        limit: int = 5,
    ):
        """Find passages similar to a given passage (by its stored embedding)."""
        results = self.store.get(COLLECTION_PASSAGES, [passage_id])
        if not results:
            return []
        # Re-embed the text (we don't store embeddings, only in Chroma)
        text = results[0].text
        q_vec = self.embedder.embed_one(text)
        all_results = self.store.search_passages(q_vec, limit=limit + 1)
        # Exclude the passage itself
        return [r for r in all_results if r.id != passage_id][:limit]

    def entity_match(
        self,
        text: str,
        limit: int = 3,
    ):
        """
        Find the canonical entity closest to the given text.

        Returns SearchResult list with entity_id, canonical_name in metadata.
        """
        q_vec = self.embedder.embed_one(text)
        return self.store.search_entities(q_vec, limit=limit)

    def find_lore_rules(self, scene_description: str, limit: int = 5):
        """Find relevant lore rules for a scene description."""
        q_vec = self.embedder.embed_one(scene_description)
        return self.store.search_lore_rules(q_vec, limit=limit)

    def find_scene_templates(self, scene_description: str, limit: int = 3):
        """Find scene templates matching a description."""
        q_vec = self.embedder.embed_one(scene_description)
        return self.store.search_scene_templates(q_vec, limit=limit)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _get_id(obj: Any) -> str:
    """Get ID from a model instance or dict."""
    if isinstance(obj, dict):
        return str(obj.get("id", ""))
    return str(getattr(obj, "id", id(obj)))


def _get_text(obj: Any) -> str:
    """Get text from a Passage model or dict."""
    if isinstance(obj, dict):
        return obj.get("text", "")
    return getattr(obj, "text", "")


def _get_field(obj: Any, field: str, default: Any = None) -> Any:
    """Get a field from a model instance or dict."""
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


def _passage_metadata(obj: Any) -> dict:
    """Build Chroma metadata dict from a Passage."""
    return {
        "book": str(_get_field(obj, "book", "") or ""),
        "chapter": str(_get_field(obj, "chapter", "") or ""),
        "chapter_num": int(_get_field(obj, "chapter_num", 0) or 0),
        "story_era": str(_get_field(obj, "story_era", "") or ""),
        "tolkien_register": str(_get_field(obj, "tolkien_register", "") or ""),
        "is_dialogue": bool(_get_field(obj, "is_dialogue", False)),
    }


def _passage_to_dict(obj: Any) -> dict:
    """Convert a Passage model to a dict for DuckDB."""
    text = _get_text(obj)
    word_count = len(text.split()) if text else 0
    return {
        "id": _get_id(obj),
        "book": _get_field(obj, "book", ""),
        "chapter": str(_get_field(obj, "chapter", "") or ""),
        "chapter_num": _get_field(obj, "chapter_num", 0),
        "paragraph_num": _get_field(obj, "paragraph_num", 0),
        "sentence_num": _get_field(obj, "sentence_num", 0),
        "word_count": word_count,
        "sentence_count": _get_field(obj, "sentence_count", 0),
        "avg_sentence_length": _get_field(obj, "avg_sentence_length", 0.0),
        "passive_ratio": _get_field(obj, "passive_ratio", 0.0),
        "dialogue_density": _get_field(obj, "dialogue_density", 0.0),
        "archaic_word_count": _get_field(obj, "archaic_word_count", 0),
        "story_era": _get_field(obj, "story_era"),
        "story_year": _get_field(obj, "story_year"),
        "tolkien_register": _get_field(obj, "tolkien_register"),
        "is_dialogue": _get_field(obj, "is_dialogue", False),
    }


def _template_description(obj: Any) -> str:
    """Build an embedding-friendly text for a SceneTemplate."""
    register = str(_get_field(obj, "register", ""))
    pattern = _get_field(obj, "structural_pattern", "")
    avg_len = _get_field(obj, "avg_sentence_length", 0.0)
    passive = _get_field(obj, "passive_ratio", 0.0)

    parts = [f"Register: {register}"]
    if pattern:
        parts.append(f"Pattern: {pattern}")
    if avg_len:
        parts.append(f"Sentence length: ~{avg_len:.0f} words")
    if passive:
        parts.append(f"Passive ratio: {passive:.0%}")

    return " | ".join(parts)
