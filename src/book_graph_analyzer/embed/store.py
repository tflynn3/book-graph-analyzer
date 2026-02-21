"""
Chroma Vector Store Manager

Four collections:
  passages       — passage text (style grounding, thematic search)
  entity_names   — canonical names + aliases (entity resolution)
  lore_rules     — rule statements (find relevant rules for a scene)
  scene_templates — template descriptions (register retrieval)

Every Chroma entry carries the Neo4j node ID in its metadata so
bidirectional linking is always possible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Collection names
COLLECTION_PASSAGES = "passages"
COLLECTION_ENTITIES = "entity_names"
COLLECTION_LORE_RULES = "lore_rules"
COLLECTION_SCENE_TEMPLATES = "scene_templates"

ALL_COLLECTIONS = [
    COLLECTION_PASSAGES,
    COLLECTION_ENTITIES,
    COLLECTION_LORE_RULES,
    COLLECTION_SCENE_TEMPLATES,
]

# Default distance function (cosine is best for text embeddings)
_DISTANCE_FN = "cosine"


@dataclass
class SearchResult:
    """A single result from a semantic search."""
    id: str
    text: str
    distance: float          # Cosine distance (lower = more similar)
    similarity: float        # 1 - distance (higher = more similar)
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"SearchResult({self.id!r}, sim={self.similarity:.3f}, text={self.text[:60]!r})"


class VectorStore:
    """
    Manages Chroma persistent collections for BGA.

    Usage (persistent):
        store = VectorStore(persist_dir="data/chroma")

    Usage (in-memory, for tests):
        store = VectorStore(ephemeral=True)
    """

    def __init__(
        self,
        persist_dir: Optional[str | Path] = None,
        ephemeral: bool = False,
    ) -> None:
        import chromadb

        if ephemeral:
            self._client = chromadb.EphemeralClient()
        else:
            path = str(persist_dir or Path("data/chroma"))
            Path(path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=path)

        # Get-or-create all collections with cosine distance
        self._cols: dict[str, Any] = {}
        for name in ALL_COLLECTIONS:
            self._cols[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": _DISTANCE_FN},
            )

        logger.debug("VectorStore ready. Collections: %s", list(self._cols.keys()))

    # ------------------------------------------------------------------
    # Generic upsert / search
    # ------------------------------------------------------------------

    def upsert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> int:
        """
        Upsert documents into a collection.

        Returns number of items upserted.
        """
        if not ids:
            return 0
        col = self._get_col(collection)
        # Chroma 1.5+ rejects empty dicts — use None when no metadata provided
        if metadatas is not None:
            # Replace any empty dicts with None
            meta = [m if m else None for m in metadatas]
        else:
            meta = [None] * len(ids)
        col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=meta)
        return len(ids)

    def search(
        self,
        collection: str,
        query_embedding: list[float],
        limit: int = 10,
        where: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Search a collection by embedding similarity.

        Args:
            collection: One of COLLECTION_* constants
            query_embedding: The query vector
            limit: Max results
            where: Optional metadata filter (Chroma where dict)

        Returns list of SearchResult sorted by similarity (highest first).
        """
        col = self._get_col(collection)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(limit, self.count(collection) or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            results = col.query(**kwargs)
        except Exception as exc:
            # Chroma 1.5+ Rust backend may raise InternalError on first query
            # if the HNSW index hasn't been flushed yet. Return empty gracefully.
            exc_str = str(exc)
            if "hnsw" in exc_str.lower() or "nothing found on disk" in exc_str.lower():
                logger.warning("Chroma HNSW not ready (collection may be freshly populated): %s", exc_str)
                return []
            raise

        output = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        dists = results["distances"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)

        for r_id, doc, dist, meta in zip(ids, docs, dists, metas):
            similarity = max(0.0, 1.0 - dist)  # cosine: 0=identical, 2=opposite
            output.append(SearchResult(
                id=r_id,
                text=doc,
                distance=dist,
                similarity=similarity,
                metadata=meta or {},
            ))

        # Sort by similarity desc
        output.sort(key=lambda r: -r.similarity)
        return output

    def get(self, collection: str, ids: list[str]) -> list[SearchResult]:
        """Retrieve specific items by ID."""
        col = self._get_col(collection)
        results = col.get(ids=ids, include=["documents", "metadatas"])
        output = []
        for r_id, doc, meta in zip(
            results["ids"], results["documents"], results.get("metadatas") or [{}] * len(results["ids"])
        ):
            output.append(SearchResult(id=r_id, text=doc, distance=0.0, similarity=1.0, metadata=meta or {}))
        return output

    def delete(self, collection: str, ids: list[str]) -> None:
        """Delete items from a collection."""
        col = self._get_col(collection)
        col.delete(ids=ids)

    # ------------------------------------------------------------------
    # Collection stats
    # ------------------------------------------------------------------

    def count(self, collection: str) -> int:
        """Return number of items in a collection."""
        col = self._get_col(collection)
        return col.count()

    def get_all_ids(self, collection: str) -> set[str]:
        """Return all IDs currently in a collection (for incremental build)."""
        col = self._get_col(collection)
        if col.count() == 0:
            return set()
        result = col.get(include=[])
        return set(result["ids"])

    def stats(self) -> dict[str, int]:
        """Return item counts for all collections."""
        return {name: self._get_col(name).count() for name in ALL_COLLECTIONS}

    # ------------------------------------------------------------------
    # Typed helpers (passages, entities, rules, templates)
    # ------------------------------------------------------------------

    def upsert_passages(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> int:
        """Upsert passage embeddings."""
        return self.upsert(COLLECTION_PASSAGES, ids, embeddings, texts, metadatas)

    def upsert_entity_names(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        names: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> int:
        """Upsert entity name embeddings."""
        return self.upsert(COLLECTION_ENTITIES, ids, embeddings, names, metadatas)

    def upsert_lore_rules(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        statements: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> int:
        """Upsert lore rule embeddings."""
        return self.upsert(COLLECTION_LORE_RULES, ids, embeddings, statements, metadatas)

    def upsert_scene_templates(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        descriptions: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> int:
        """Upsert scene template embeddings."""
        return self.upsert(COLLECTION_SCENE_TEMPLATES, ids, embeddings, descriptions, metadatas)

    def search_passages(
        self, query_embedding: list[float], limit: int = 10, book: Optional[str] = None
    ) -> list[SearchResult]:
        """Search passages by semantic similarity."""
        where = {"book": book} if book else None
        return self.search(COLLECTION_PASSAGES, query_embedding, limit, where=where)

    def search_entities(self, query_embedding: list[float], limit: int = 5) -> list[SearchResult]:
        """Search entity names by semantic similarity (for alias resolution)."""
        return self.search(COLLECTION_ENTITIES, query_embedding, limit)

    def search_lore_rules(self, query_embedding: list[float], limit: int = 5) -> list[SearchResult]:
        """Find relevant lore rules for a scene."""
        return self.search(COLLECTION_LORE_RULES, query_embedding, limit)

    def search_scene_templates(self, query_embedding: list[float], limit: int = 3) -> list[SearchResult]:
        """Find scene templates by semantic similarity."""
        return self.search(COLLECTION_SCENE_TEMPLATES, query_embedding, limit)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_col(self, name: str):
        if name not in self._cols:
            raise ValueError(
                f"Unknown collection: {name!r}. Valid: {list(self._cols.keys())}"
            )
        return self._cols[name]

    def reset_collection(self, collection: str) -> None:
        """Delete and recreate a collection (useful for full rebuilds)."""
        import chromadb
        self._client.delete_collection(collection)
        self._cols[collection] = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": _DISTANCE_FN},
        )
        logger.info("Collection %r reset", collection)
