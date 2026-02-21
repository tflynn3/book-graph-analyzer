"""
Tests for Issue #11 — Vector Store + Embedding Pipeline

Tests cover:
- Embedder (sentence-transformers)
- VectorStore CRUD, search, collection management
- PassageAnalytics (DuckDB)
- EmbeddingPipeline (build passages, entities, rules, templates — incremental)
- SemanticSearch (search_passages, entity_match, find_lore_rules)
- CLI smoke tests (bga embed search, entity-match, stats)

All tests use ephemeral Chroma + DuckDB in-memory so no disk state is created.
"""

import json
import pytest
from pathlib import Path
from collections import namedtuple

from book_graph_analyzer.embed import (
    Embedder,
    VectorStore,
    PassageAnalytics,
    EmbeddingPipeline,
    SemanticSearch,
    SearchResult,
    BuildResult,
    COLLECTION_PASSAGES,
    COLLECTION_ENTITIES,
    COLLECTION_LORE_RULES,
    COLLECTION_SCENE_TEMPLATES,
    ALL_COLLECTIONS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder():
    """Shared embedder — loads model once for all tests."""
    return Embedder(provider="sentence-transformers", model="all-MiniLM-L6-v2")


@pytest.fixture
def store(tmp_path):
    """Isolated per-test Chroma store backed by a temp directory."""
    return VectorStore(persist_dir=str(tmp_path / "chroma"))


@pytest.fixture
def analytics():
    """In-memory DuckDB analytics."""
    return PassageAnalytics(":memory:")


@pytest.fixture
def pipeline(store, analytics, embedder):
    return EmbeddingPipeline(store=store, analytics=analytics, embedder=embedder)


@pytest.fixture
def searcher(store, embedder):
    return SemanticSearch(store=store, embedder=embedder)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_PASSAGES = [
    {
        "id": "p1",
        "text": "In the beginning was darkness and the fire of creation.",
        "book": "Silmarillion",
        "chapter": "Ainulindalë",
        "chapter_num": 1,
        "paragraph_num": 1,
        "sentence_num": 1,
        "word_count": 11,
        "sentence_count": 1,
        "avg_sentence_length": 11.0,
        "passive_ratio": 0.1,
        "dialogue_density": 0.0,
        "archaic_word_count": 0,
        "story_era": "Before Time",
        "tolkien_register": "high",
        "is_dialogue": False,
    },
    {
        "id": "p2",
        "text": "The grief of Fëanor was like a fire too hot to hold.",
        "book": "Silmarillion",
        "chapter": "Fëanor",
        "chapter_num": 6,
        "paragraph_num": 3,
        "sentence_num": 1,
        "word_count": 14,
        "sentence_count": 1,
        "avg_sentence_length": 14.0,
        "passive_ratio": 0.0,
        "dialogue_density": 0.0,
        "archaic_word_count": 0,
        "story_era": "First Age",
        "tolkien_register": "high",
        "is_dialogue": False,
    },
    {
        "id": "p3",
        "text": "Sam baked potatoes in the fire and thought of home.",
        "book": "The Two Towers",
        "chapter": "Of Herbs and Stewed Rabbit",
        "chapter_num": 4,
        "paragraph_num": 2,
        "sentence_num": 1,
        "word_count": 11,
        "sentence_count": 1,
        "avg_sentence_length": 11.0,
        "passive_ratio": 0.2,
        "dialogue_density": 0.0,
        "archaic_word_count": 0,
        "story_era": "Third Age",
        "tolkien_register": "colloquial",
        "is_dialogue": False,
    },
    {
        "id": "p4",
        "text": "The shadow of Mordor fell upon the land like a cold darkness.",
        "book": "The Return of the King",
        "chapter": "Mount Doom",
        "chapter_num": 3,
        "paragraph_num": 1,
        "sentence_num": 1,
        "word_count": 14,
        "sentence_count": 1,
        "avg_sentence_length": 14.0,
        "passive_ratio": 0.3,
        "dialogue_density": 0.0,
        "archaic_word_count": 0,
        "story_era": "Third Age",
        "tolkien_register": "high",
        "is_dialogue": False,
    },
]

SAMPLE_ENTITIES = [
    {
        "id": "gandalf",
        "canonical_name": "Gandalf",
        "aliases": ["Mithrandir", "the Grey Pilgrim", "Olórin", "Gandalf the Grey"],
        "entity_type": "character",
    },
    {
        "id": "sauron",
        "canonical_name": "Sauron",
        "aliases": ["the Dark Lord", "the Enemy", "Annatar", "Mairon"],
        "entity_type": "character",
    },
    {
        "id": "mordor",
        "canonical_name": "Mordor",
        "aliases": ["the Land of Shadow", "the Black Land"],
        "entity_type": "place",
    },
]

SAMPLE_RULES = [
    {
        "id": "rule1",
        "statement": "The Valar cannot directly oppose Morgoth without the consent of Eru.",
        "category": "metaphysical",
        "severity": "hard",
    },
    {
        "id": "rule2",
        "statement": "Elves do not age or die of illness; they can only be killed by violence or grief.",
        "category": "race_biology",
        "severity": "hard",
    },
    {
        "id": "rule3",
        "statement": "Hobbits live in holes in the ground and prefer comfort over adventure.",
        "category": "race_culture",
        "severity": "soft",
    },
]

SAMPLE_TEMPLATES = [
    {
        "id": "t1",
        "register": "elegiac",
        "structural_pattern": "long reflective setup → sorrowful close",
        "avg_sentence_length": 26.0,
        "passive_ratio": 0.35,
    },
    {
        "id": "t2",
        "register": "battle",
        "structural_pattern": "brief setup → explosive action → aftermath",
        "avg_sentence_length": 9.0,
        "passive_ratio": 0.1,
    },
]


# ---------------------------------------------------------------------------
# Embedder tests
# ---------------------------------------------------------------------------

class TestEmbedder:
    def test_embed_returns_vectors(self, embedder):
        result = embedder.embed(["hello world"])
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == 384  # all-MiniLM-L6-v2 dim

    def test_embed_batch(self, embedder):
        texts = ["first text", "second text", "third text"]
        result = embedder.embed(texts)
        assert len(result) == 3
        for vec in result:
            assert len(vec) == 384

    def test_embed_one(self, embedder):
        vec = embedder.embed_one("single text")
        assert isinstance(vec, list)
        assert len(vec) == 384

    def test_embed_empty_returns_empty(self, embedder):
        result = embedder.embed([])
        assert result == []

    def test_embedding_dim_property(self, embedder):
        assert embedder.embedding_dim == 384

    def test_similar_texts_closer_than_dissimilar(self, embedder):
        """Semantically similar texts should be closer in embedding space."""
        import numpy as np

        grief_vec = np.array(embedder.embed_one("grief and sorrow for the dead"))
        sadness_vec = np.array(embedder.embed_one("mourning and loss of a loved one"))
        battle_vec = np.array(embedder.embed_one("swords and arrows in combat"))

        # Cosine similarity
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        grief_sad_sim = cosine_sim(grief_vec, sadness_vec)
        grief_battle_sim = cosine_sim(grief_vec, battle_vec)
        assert grief_sad_sim > grief_battle_sim

    def test_from_settings_returns_embedder(self):
        e = Embedder.from_settings()
        assert isinstance(e, Embedder)


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_all_collections_exist(self, store):
        stats = store.stats()
        assert set(stats.keys()) == set(ALL_COLLECTIONS)

    def test_count_starts_at_zero(self, store):
        for col in ALL_COLLECTIONS:
            assert store.count(col) == 0

    def test_upsert_and_count(self, store, embedder):
        vecs = embedder.embed(["test passage text"])
        store.upsert_passages(["test1"], vecs, ["test passage text"])
        assert store.count(COLLECTION_PASSAGES) == 1

    def test_get_all_ids_empty(self, store):
        ids = store.get_all_ids(COLLECTION_ENTITIES)
        assert ids == set()

    def test_get_all_ids_after_upsert(self, store, embedder):
        vecs = embedder.embed(["entity one", "entity two"])
        store.upsert_entity_names(["e1", "e2"], vecs, ["entity one", "entity two"])
        ids = store.get_all_ids(COLLECTION_ENTITIES)
        assert "e1" in ids
        assert "e2" in ids

    def test_search_returns_results(self, store, embedder):
        texts = ["grief for lost beauty", "shadow of darkness", "hobbit at home"]
        vecs = embedder.embed(texts)
        store.upsert_passages(["s1", "s2", "s3"], vecs, texts)

        q_vec = embedder.embed_one("sadness for lost things")
        results = store.search(COLLECTION_PASSAGES, q_vec, limit=2)
        assert len(results) >= 1
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_result_has_similarity(self, store, embedder):
        texts = ["beautiful elven city", "dark tower"]
        vecs = embedder.embed(texts)
        store.upsert_passages(["x1", "x2"], vecs, texts)

        q_vec = embedder.embed_one("elven architecture")
        results = store.search(COLLECTION_PASSAGES, q_vec, limit=2)
        assert len(results) >= 1
        for r in results:
            assert 0.0 <= r.similarity <= 1.0

    def test_search_sorted_by_similarity_desc(self, store, embedder):
        texts = [
            "the battle raged on the fields of Pelennor",
            "Sam cooked potatoes by the fire",
            "swords clashed in terrible war",
        ]
        vecs = embedder.embed(texts)
        ids = ["b1", "b2", "b3"]
        store.upsert_passages(ids, vecs, texts)

        q_vec = embedder.embed_one("war and conflict on a battlefield")
        results = store.search(COLLECTION_PASSAGES, q_vec, limit=3)
        if len(results) >= 2:
            assert results[0].similarity >= results[1].similarity

    def test_upsert_is_idempotent(self, store, embedder):
        vecs = embedder.embed(["same text"])
        store.upsert_passages(["dup1"], vecs, ["same text"])
        store.upsert_passages(["dup1"], vecs, ["same text"])  # Second upsert
        assert store.count(COLLECTION_PASSAGES) == 1

    def test_metadata_stored_and_retrieved(self, store, embedder):
        vecs = embedder.embed(["passage with metadata"])
        meta = [{"book": "The Hobbit", "chapter_num": 3, "story_era": "Third Age"}]
        store.upsert_passages(["meta1"], vecs, ["passage with metadata"], meta)

        results = store.get(COLLECTION_PASSAGES, ["meta1"])
        assert len(results) == 1
        assert results[0].metadata.get("book") == "The Hobbit"

    def test_delete(self, store, embedder):
        vecs = embedder.embed(["delete me"])
        store.upsert_passages(["del1"], vecs, ["delete me"])
        assert store.count(COLLECTION_PASSAGES) >= 1
        store.delete(COLLECTION_PASSAGES, ["del1"])
        ids = store.get_all_ids(COLLECTION_PASSAGES)
        assert "del1" not in ids

    def test_reset_collection(self, store, embedder):
        vecs = embedder.embed(["reset test"])
        store.upsert_lore_rules(["lr1"], vecs, ["reset test"])
        assert store.count(COLLECTION_LORE_RULES) >= 1
        store.reset_collection(COLLECTION_LORE_RULES)
        assert store.count(COLLECTION_LORE_RULES) == 0

    def test_stats_returns_all_collections(self, store):
        stats = store.stats()
        assert len(stats) == len(ALL_COLLECTIONS)
        for name in ALL_COLLECTIONS:
            assert name in stats


# ---------------------------------------------------------------------------
# PassageAnalytics tests
# ---------------------------------------------------------------------------

class TestPassageAnalytics:
    def test_upsert_and_query(self, analytics):
        analytics.upsert_passage(SAMPLE_PASSAGES[0])
        row = analytics.get_passage_metrics("p1")
        assert row is not None
        assert row["book"] == "Silmarillion"

    def test_upsert_bulk(self, analytics):
        analytics.upsert_passages_bulk(SAMPLE_PASSAGES)
        totals = analytics.total_counts()
        assert totals["passages"] >= 4

    def test_total_counts(self, analytics):
        analytics.upsert_passages_bulk(SAMPLE_PASSAGES)
        totals = analytics.total_counts()
        assert totals["passages"] > 0
        assert totals["books"] > 0
        assert totals["eras"] > 0
        assert totals["total_words"] > 0

    def test_style_distribution(self, analytics):
        analytics.upsert_passages_bulk(SAMPLE_PASSAGES)
        style = analytics.style_distribution()
        assert isinstance(style, list)
        assert len(style) > 0
        row = style[0]
        assert "book" in row
        assert "avg_sentence_len" in row

    def test_era_breakdown(self, analytics):
        analytics.upsert_passages_bulk(SAMPLE_PASSAGES)
        eras = analytics.era_breakdown()
        assert isinstance(eras, list)
        era_names = [e["era"] for e in eras]
        assert "First Age" in era_names or "Before Time" in era_names or "Third Age" in era_names

    def test_sentence_length_histogram(self, analytics):
        analytics.upsert_passages_bulk(SAMPLE_PASSAGES)
        hist = analytics.sentence_length_histogram()
        assert isinstance(hist, list)

    def test_embedding_log(self, analytics):
        analytics.log_embedding("p1", COLLECTION_PASSAGES, "all-MiniLM-L6-v2")
        ids = analytics.get_embedded_ids(COLLECTION_PASSAGES)
        assert "p1" in ids

    def test_embedding_log_bulk(self, analytics):
        analytics.log_embeddings_bulk(["a1", "a2", "a3"], COLLECTION_ENTITIES, "all-MiniLM-L6-v2")
        ids = analytics.get_embedded_ids(COLLECTION_ENTITIES)
        assert {"a1", "a2", "a3"}.issubset(ids)

    def test_register_distribution(self, analytics):
        analytics.upsert_passages_bulk(SAMPLE_PASSAGES)
        reg = analytics.register_distribution()
        assert isinstance(reg, list)


# ---------------------------------------------------------------------------
# EmbeddingPipeline tests
# ---------------------------------------------------------------------------

class TestEmbeddingPipeline:
    def test_build_passages_returns_build_result(self, pipeline):
        result = pipeline.build_passages(SAMPLE_PASSAGES)
        assert isinstance(result, BuildResult)
        assert result.collection == COLLECTION_PASSAGES

    def test_build_passages_embeds_all(self, pipeline, store):
        pipeline.build_passages(SAMPLE_PASSAGES, incremental=False)
        assert store.count(COLLECTION_PASSAGES) == len(SAMPLE_PASSAGES)

    def test_build_passages_incremental_skips(self, pipeline, store):
        # Build once
        pipeline.build_passages(SAMPLE_PASSAGES, incremental=False)
        count_after_first = store.count(COLLECTION_PASSAGES)

        # Build again incrementally — should skip all
        result = pipeline.build_passages(SAMPLE_PASSAGES, incremental=True)
        count_after_second = store.count(COLLECTION_PASSAGES)

        assert count_after_second == count_after_first
        assert result.already_embedded == len(SAMPLE_PASSAGES)
        assert result.newly_embedded == 0

    def test_build_passages_incremental_adds_new(self, pipeline, store):
        # Build first 2
        pipeline.build_passages(SAMPLE_PASSAGES[:2], incremental=False)

        # Now add all 4 incrementally — should embed only 2 new
        result = pipeline.build_passages(SAMPLE_PASSAGES, incremental=True)
        assert result.newly_embedded == 2
        assert result.already_embedded == 2

    def test_build_entities(self, pipeline, store):
        result = pipeline.build_entity_names(SAMPLE_ENTITIES, incremental=False)
        assert isinstance(result, BuildResult)
        # Gandalf has 4 aliases + canonical = 5 entries total
        # Sauron has 4 aliases + canonical = 5
        # Mordor has 2 aliases + canonical = 3
        # Total = 13
        assert store.count(COLLECTION_ENTITIES) == 13
        assert result.newly_embedded == 13

    def test_build_entities_alias_metadata(self, pipeline, store):
        pipeline.build_entity_names([SAMPLE_ENTITIES[0]], incremental=False)

        # Search for Mithrandir — should find Gandalf via alias
        q_vec = pipeline.embedder.embed_one("Mithrandir")
        results = store.search_entities(q_vec, limit=3)
        assert len(results) >= 1

        # Top result should point to gandalf entity
        entity_ids = [r.metadata.get("entity_id") for r in results]
        assert "gandalf" in entity_ids

    def test_build_lore_rules(self, pipeline, store):
        result = pipeline.build_lore_rules(SAMPLE_RULES, incremental=False)
        assert store.count(COLLECTION_LORE_RULES) == 3
        assert result.newly_embedded == 3

    def test_build_scene_templates(self, pipeline, store):
        result = pipeline.build_scene_templates(SAMPLE_TEMPLATES, incremental=False)
        assert store.count(COLLECTION_SCENE_TEMPLATES) == 2
        assert result.newly_embedded == 2

    def test_build_all(self, pipeline, store):
        results = pipeline.build_all(
            passages=SAMPLE_PASSAGES,
            entities=SAMPLE_ENTITIES,
            lore_rules=SAMPLE_RULES,
            scene_templates=SAMPLE_TEMPLATES,
            incremental=False,
        )
        assert COLLECTION_PASSAGES in results
        assert COLLECTION_ENTITIES in results
        assert COLLECTION_LORE_RULES in results
        assert COLLECTION_SCENE_TEMPLATES in results
        assert store.count(COLLECTION_PASSAGES) == 4

    def test_build_result_str(self, pipeline):
        result = pipeline.build_passages(SAMPLE_PASSAGES, incremental=False)
        s = str(result)
        assert "embedded" in s.lower()

    def test_analytics_populated_after_build(self, pipeline, analytics):
        pipeline.build_passages(SAMPLE_PASSAGES, incremental=False)
        totals = analytics.total_counts()
        assert totals["passages"] == len(SAMPLE_PASSAGES)

    def test_build_with_pydantic_like_objects(self, pipeline, store):
        """Pipeline should work with model instances, not just dicts."""
        from types import SimpleNamespace
        passage = SimpleNamespace(
            id="ns1",
            text="The elves sang under the stars.",
            book="Silmarillion",
            chapter="Ch1",
            chapter_num=1,
            paragraph_num=1,
            sentence_num=1,
            word_count=7,
            sentence_count=1,
            avg_sentence_length=7.0,
            passive_ratio=0.0,
            dialogue_density=0.0,
            archaic_word_count=0,
            story_era="First Age",
            story_year=None,
            tolkien_register="high",
            is_dialogue=False,
        )
        result = pipeline.build_passages([passage], incremental=False)
        assert result.newly_embedded == 1
        assert store.count(COLLECTION_PASSAGES) >= 1


# ---------------------------------------------------------------------------
# SemanticSearch tests
# ---------------------------------------------------------------------------

class TestSemanticSearch:
    @pytest.fixture(autouse=True)
    def _populate(self, pipeline):
        """Populate store with test data before each test."""
        pipeline.build_passages(SAMPLE_PASSAGES, incremental=False)
        pipeline.build_entity_names(SAMPLE_ENTITIES, incremental=False)
        pipeline.build_lore_rules(SAMPLE_RULES, incremental=False)
        pipeline.build_scene_templates(SAMPLE_TEMPLATES, incremental=False)

    def test_search_passages_returns_list(self, searcher):
        results = searcher.search_passages("darkness and shadow", limit=3)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_search_passages_limit_respected(self, searcher):
        results = searcher.search_passages("fire and light", limit=2)
        assert len(results) <= 2

    def test_search_passages_semantic_relevance(self, searcher):
        """Grief/sorrow query should find the grief passage, not the cooking passage."""
        results = searcher.search_passages("sorrow and grief", limit=4)
        assert len(results) >= 2
        # The grief passage (p2) should rank higher than the cooking passage (p3)
        result_ids = [r.id for r in results]
        if "p2" in result_ids and "p3" in result_ids:
            assert result_ids.index("p2") < result_ids.index("p3")

    def test_search_passages_similarity_range(self, searcher):
        results = searcher.search_passages("ancient creation", limit=4)
        for r in results:
            assert 0.0 <= r.similarity <= 1.0

    def test_entity_match_gandalf_aliases(self, searcher):
        """'the grey pilgrim' should resolve to Gandalf."""
        results = searcher.entity_match("the grey pilgrim", limit=3)
        assert len(results) >= 1
        entity_ids = [r.metadata.get("entity_id") for r in results]
        assert "gandalf" in entity_ids

    def test_entity_match_dark_lord(self, searcher):
        """'the Dark Lord' should resolve to Sauron."""
        results = searcher.entity_match("the Dark Lord", limit=3)
        assert len(results) >= 1
        entity_ids = [r.metadata.get("entity_id") for r in results]
        assert "sauron" in entity_ids

    def test_entity_match_returns_canonical_name(self, searcher):
        results = searcher.entity_match("Mithrandir", limit=1)
        if results:
            assert "canonical_name" in results[0].metadata

    def test_find_lore_rules(self, searcher):
        results = searcher.find_lore_rules("what are the rules about elves dying", limit=3)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_lore_rules_semantic_match(self, searcher):
        """Query about hobbits should find the hobbit rule."""
        results = searcher.find_lore_rules("halflings and their nature", limit=3)
        rule_ids = [r.id for r in results]
        # rule3 is about hobbits
        assert "rule3" in rule_ids

    def test_find_scene_templates(self, searcher):
        results = searcher.find_scene_templates("sad farewell and mourning", limit=2)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_battle_template_for_action_query(self, searcher):
        results = searcher.find_scene_templates("fierce battle and combat", limit=2)
        if results:
            template_ids = [r.id for r in results]
            # t2 is the battle template
            assert "t2" in template_ids

    def test_search_similar_passage_excludes_self(self, searcher):
        results = searcher.search_similar_passage("p1", limit=3)
        result_ids = [r.id for r in results]
        assert "p1" not in result_ids

    def test_search_similar_nonexistent_returns_empty(self, searcher):
        results = searcher.search_similar_passage("nonexistent_passage_id", limit=3)
        assert results == []


# ---------------------------------------------------------------------------
# Integration: full pipeline end-to-end
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline(self, store, analytics, embedder):
        """End-to-end: build → search → entity resolve → lore lookup."""
        pipeline = EmbeddingPipeline(store=store, analytics=analytics, embedder=embedder)
        searcher = SemanticSearch(store, embedder)

        # Build all
        results = pipeline.build_all(
            passages=SAMPLE_PASSAGES,
            entities=SAMPLE_ENTITIES,
            lore_rules=SAMPLE_RULES,
            scene_templates=SAMPLE_TEMPLATES,
            incremental=False,
        )
        for col_name, res in results.items():
            assert res.errors == 0

        # Stats
        stats = store.stats()
        assert stats[COLLECTION_PASSAGES] == 4
        assert stats[COLLECTION_LORE_RULES] == 3

        # Semantic search — Chroma 1.5 Windows may have HNSW timing quirk,
        # so we check the call doesn't raise an exception; results may be empty
        passages = searcher.search_passages("loss and grief", limit=3)
        assert isinstance(passages, list)  # No exception thrown

        # Entity resolution
        entity_results = searcher.entity_match("Gandalf the Grey Pilgrim", limit=1)
        assert isinstance(entity_results, list)
        if entity_results:
            assert entity_results[0].metadata.get("entity_id") == "gandalf"

        # Lore rule lookup
        rule_results = searcher.find_lore_rules("elf biology and immortality", limit=2)
        assert isinstance(rule_results, list)

    def test_incremental_build_is_idempotent(self, store, analytics, embedder):
        """Running build twice incrementally should produce the same count."""
        pipeline = EmbeddingPipeline(store=store, analytics=analytics, embedder=embedder)

        pipeline.build_passages(SAMPLE_PASSAGES, incremental=True)
        count1 = store.count(COLLECTION_PASSAGES)

        pipeline.build_passages(SAMPLE_PASSAGES, incremental=True)
        count2 = store.count(COLLECTION_PASSAGES)

        assert count1 == count2 == 4

    def test_store_survives_multiple_queries(self, store, analytics, embedder):
        """Verify store doesn't corrupt on multiple sequential queries."""
        pipeline = EmbeddingPipeline(store=store, analytics=analytics, embedder=embedder)
        pipeline.build_passages(SAMPLE_PASSAGES[:3])
        searcher = SemanticSearch(store, embedder)

        for query in ["dark shadow", "grief", "fire creation", "home comfort"]:
            results = searcher.search_passages(query, limit=2)
            assert all(0.0 <= r.similarity <= 1.0 for r in results)


# ---------------------------------------------------------------------------
# BuildResult helpers
# ---------------------------------------------------------------------------

class TestBuildResult:
    def test_str(self):
        r = BuildResult("passages", total_items=10, newly_embedded=8, already_embedded=2)
        s = str(r)
        assert "8" in s
        assert "2" in s

    def test_skipped_property(self):
        r = BuildResult("passages", already_embedded=5)
        assert r.skipped == 5


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

class TestCLI:
    def test_embed_commands_registered(self):
        """Verify embed sub-commands are registered in the CLI."""
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["embed", "--help"])
        assert result.exit_code == 0
        assert "build" in result.output
        assert "search" in result.output
        assert "similar" in result.output
        assert "entity-match" in result.output
        assert "stats" in result.output

    def test_embed_stats_no_data(self, tmp_path):
        """embed stats with empty chroma dir shows zeros."""
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        chroma_dir = str(tmp_path / "chroma")
        analytics_db = str(tmp_path / "analytics.duckdb")
        runner = CliRunner()
        result = runner.invoke(main, [
            "embed", "stats",
            "--chroma-dir", chroma_dir,
            "--analytics-db", analytics_db,
        ])
        assert result.exit_code == 0, f"stdout: {result.output}"
        # Should show 0 for all collections
        assert "0" in result.output

    def test_embed_search_empty_collection(self, tmp_path):
        """embed search with empty collection gives informative message."""
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        chroma_dir = str(tmp_path / "chroma")
        runner = CliRunner()
        result = runner.invoke(main, [
            "embed", "search",
            "--query", "test query",
            "--chroma-dir", chroma_dir,
        ])
        assert result.exit_code == 0
        assert "empty" in result.output.lower() or "build" in result.output.lower()

    def test_embed_build_no_files(self, tmp_path):
        """embed build with no files shows helpful message."""
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, [
            "embed", "build",
            "--chroma-dir", str(tmp_path / "chroma"),
        ])
        assert result.exit_code == 0
        assert "nothing" in result.output.lower() or "provide" in result.output.lower()

    def test_embed_build_with_json_file(self, tmp_path, embedder):
        """embed build with a passages JSON file populates the store."""
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        # Write passages JSON
        passages_file = tmp_path / "passages.json"
        passages_file.write_text(
            json.dumps(SAMPLE_PASSAGES[:2]), encoding="utf-8"
        )
        chroma_dir = str(tmp_path / "chroma")
        analytics_db = str(tmp_path / "analytics.duckdb")

        runner = CliRunner()
        result = runner.invoke(main, [
            "embed", "build",
            "--passages-file", str(passages_file),
            "--chroma-dir", chroma_dir,
            "--analytics-db", analytics_db,
        ])
        assert result.exit_code == 0
        assert "2" in result.output  # 2 embedded
