"""
BGA Vector Store + Embedding Pipeline (Issue #11)

Provides semantic search over passages, entity names, lore rules, and
scene templates using Chroma (local vector store) + sentence-transformers.
DuckDB provides fast tabular analytics on corpus metrics.

Quick start:
    from book_graph_analyzer.embed import VectorStore, EmbeddingPipeline, SemanticSearch

    store = VectorStore("data/chroma")
    pipeline = EmbeddingPipeline(store)
    pipeline.build_passages(my_passages)

    searcher = SemanticSearch(store)
    results = searcher.search_passages("grief for lost beauty", limit=5)
    match = searcher.entity_match("the grey pilgrim")
"""

from .embedder import Embedder
from .store import (
    VectorStore,
    SearchResult,
    COLLECTION_PASSAGES,
    COLLECTION_ENTITIES,
    COLLECTION_LORE_RULES,
    COLLECTION_SCENE_TEMPLATES,
    ALL_COLLECTIONS,
)
from .pipeline import (
    EmbeddingPipeline,
    SemanticSearch,
    BuildResult,
)
from .analytics import PassageAnalytics

__all__ = [
    # Embedder
    "Embedder",
    # Store
    "VectorStore",
    "SearchResult",
    "COLLECTION_PASSAGES",
    "COLLECTION_ENTITIES",
    "COLLECTION_LORE_RULES",
    "COLLECTION_SCENE_TEMPLATES",
    "ALL_COLLECTIONS",
    # Pipeline + search
    "EmbeddingPipeline",
    "SemanticSearch",
    "BuildResult",
    # Analytics
    "PassageAnalytics",
]
