"""Shared helpers for passage-level graph extraction workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..ingest.splitter import Passage
from .dynamic_resolver import EntityCluster, EntityMention
from .extractor import EntityExtractor, ExtractionResult
from .propositions import PropositionExtractionResult, PropositionExtractor
from .relationships import RelationshipExtractionResult, RelationshipExtractor

_SPEAKER_PREFIXES = {
    "mr",
    "mrs",
    "master",
    "mistress",
    "lord",
    "lady",
    "king",
    "queen",
    "captain",
    "sir",
    "old",
    "young",
}


@dataclass
class BookGraphExtraction:
    """Entity and relationship artifacts for one book-level extraction run."""

    entity_results: list[ExtractionResult]
    relationship_results: list[RelationshipExtractionResult]
    proposition_results: list[PropositionExtractionResult]
    unique_entity_count: int
    resolved_mention_count: int
    unresolved_entity_count: int
    total_relationships: int
    total_propositions: int


def extract_book_graph(
    passages: list[Passage],
    *,
    use_llm: bool = False,
    seed_dir: Path | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> BookGraphExtraction:
    """Run the seeded passage-level entity/relationship extraction pipeline."""
    extractor = EntityExtractor(use_llm=use_llm, seed_dir=seed_dir)
    rel_extractor = RelationshipExtractor(resolver=extractor.resolver, use_llm=use_llm)
    proposition_extractor = PropositionExtractor()

    entity_results: list[ExtractionResult] = []
    relationship_results: list[RelationshipExtractionResult] = []
    proposition_results: list[PropositionExtractionResult] = []
    total = len(passages)

    for idx, passage in enumerate(passages, start=1):
        entity_result = extractor.extract_from_passage(passage)
        entity_results.append(entity_result)

        if len(entity_result.entities) >= 2:
            rel_result = rel_extractor.extract_relationships(
                text=passage.text,
                passage_id=passage.id,
                entities=entity_result.entities,
            )
        else:
            rel_result = RelationshipExtractionResult(
                passage_id=passage.id,
                passage_text=passage.text,
                relationships=[],
                entities_involved=entity_result.entities,
            )
        relationship_results.append(rel_result)
        proposition_results.append(
            proposition_extractor.extract_from_passage(passage, entity_result.entities)
        )

        if progress_callback:
            progress_callback(
                idx,
                total,
                "Extracting entities, relationships, and propositions...",
            )

    resolved_mentions = [
        entity
        for result in entity_results
        for entity in result.entities
        if entity.canonical_id
    ]
    unique_entity_ids = {entity.canonical_id for entity in resolved_mentions if entity.canonical_id}
    unresolved_count = sum(
        1
        for result in entity_results
        for entity in result.entities
        if not entity.canonical_id
    )
    total_relationships = sum(len(result.relationships) for result in relationship_results)
    total_propositions = sum(len(result.propositions) for result in proposition_results)

    return BookGraphExtraction(
        entity_results=entity_results,
        relationship_results=relationship_results,
        proposition_results=proposition_results,
        unique_entity_count=len(unique_entity_ids),
        resolved_mention_count=len(resolved_mentions),
        unresolved_entity_count=unresolved_count,
        total_relationships=total_relationships,
        total_propositions=total_propositions,
    )


def build_entity_id_map(entity_results: list[ExtractionResult]) -> dict[str, str]:
    """Build a lookup map for downstream consumers such as voice analysis."""
    entity_map: dict[str, str] = {}
    for result in entity_results:
        for entity in result.entities:
            if entity.canonical_id and entity.canonical_name:
                for variant in _entity_name_variants(entity.canonical_name):
                    entity_map.setdefault(variant, entity.canonical_id)
                for variant in _entity_name_variants(entity.extracted.text):
                    entity_map.setdefault(variant, entity.canonical_id)
    return entity_map


def _entity_name_variants(name: str) -> set[str]:
    variants: set[str] = set()
    raw = name.strip()
    if not raw:
        return variants

    variants.add(raw)
    variants.add(raw.lower())

    stripped_possessive = re.sub(r"['\u2019]s$", "", raw).strip()
    if stripped_possessive and stripped_possessive != raw:
        variants.add(stripped_possessive)
        variants.add(stripped_possessive.lower())

    ascii_name = re.sub(r"[^A-Za-z\u00C0-\u00D6\u00D8-\u00DE\u00E0-\u00F6\u00F8-\u00FF\s'-]", " ", raw)
    normalized = re.sub(r"\s+", " ", ascii_name).strip()
    if normalized and normalized != raw:
        variants.add(normalized)
        variants.add(normalized.lower())

    tokens = normalized.split()
    if len(tokens) >= 2 and tokens[0].rstrip(".").lower() in _SPEAKER_PREFIXES:
        shortened = " ".join(tokens[1:]).strip()
        if shortened:
            variants.add(shortened)
            variants.add(shortened.lower())

    return variants


def build_entity_clusters(entity_results: list[ExtractionResult]) -> dict[str, EntityCluster]:
    """Aggregate passage-level extraction results into cluster objects for corpus resolution."""
    clusters: dict[str, EntityCluster] = {}

    for result in entity_results:
        passage = result.passage
        for entity in result.entities:
            cluster_id = entity.canonical_id or _make_unresolved_cluster_id(entity.extracted.text, entity.entity_type)
            canonical_name = entity.canonical_name or entity.extracted.text
            entity_type = _normalize_cluster_type(entity.entity_type)

            cluster = clusters.get(cluster_id)
            if cluster is None:
                cluster = EntityCluster(
                    id=cluster_id,
                    canonical_name=canonical_name,
                    entity_type=entity_type,
                    confidence=max(0.0, float(entity.confidence or 0.0)),
                )
                clusters[cluster_id] = cluster

            mention = EntityMention(
                text=entity.extracted.text,
                label=entity.extracted.label,
                passage_id=passage.id,
                passage_text=passage.text,
                char_offset=entity.extracted.start_char,
                context_before=passage.text[max(0, entity.extracted.start_char - 50):entity.extracted.start_char],
                context_after=passage.text[entity.extracted.end_char:entity.extracted.end_char + 50],
            )
            cluster.add_mention(mention)
            cluster.confidence = max(cluster.confidence, float(entity.confidence or 0.0))
            if canonical_name.lower() != entity.extracted.text.lower():
                cluster.aliases.add(entity.extracted.text)

    return clusters


def _normalize_cluster_type(entity_type: str) -> str:
    if entity_type in {"character", "place", "object"}:
        return entity_type
    return "unknown"


def _make_unresolved_cluster_id(text: str, entity_type: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "unknown"
    return f"unresolved_{entity_type}_{slug}"
