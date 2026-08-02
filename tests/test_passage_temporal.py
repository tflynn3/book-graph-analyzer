"""Tests for Passage Temporal Zoom + Multi-Era Reference Model (Issue #4).

All tests in this module run without a Neo4j connection — they cover:
  - Passage model new fields
  - EraReference model
  - PassageTemporalWriter pure-Python helpers
  - compute_temporal_zoom_batch
  - temporal_depth_visualization_color
  - era_approx_years_back
"""

from book_graph_analyzer.models.passage import Passage
from book_graph_analyzer.models.era_reference import EraReference, TemporalZoomResult
from book_graph_analyzer.graph.passage_writer import (
    PassageTemporalWriter,
    compute_temporal_zoom_batch,
    era_approx_years_back,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_passage(**overrides) -> Passage:
    """Create a minimal Passage with sensible defaults."""
    defaults = dict(
        id="p_001",
        text="Gandalf spoke of the Elder Days when the world was young.",
        book="The Lord of the Rings",
        chapter="The Shadow of the Past",
        chapter_num=2,
        paragraph_num=5,
        sentence_num=3,
        char_offset=1234,
    )
    defaults.update(overrides)
    return Passage(**defaults)


# ---------------------------------------------------------------------------
# Passage model — new temporal fields
# ---------------------------------------------------------------------------

class TestPassageTemporalFields:
    def test_default_temporal_fields_are_none(self):
        p = make_passage()
        assert p.story_era is None
        assert p.story_year is None
        assert p.temporal_depth_era is None
        assert p.temporal_depth_years_back is None
        assert p.era_reference_count == 0

    def test_set_story_era_and_year(self):
        p = make_passage(story_era="Third Age", story_year=3018)
        assert p.story_era == "Third Age"
        assert p.story_year == 3018

    def test_set_temporal_depth_fields(self):
        p = make_passage(
            story_era="Third Age",
            story_year=3018,
            temporal_depth_era="Before Time",
            temporal_depth_years_back=20_000.0,
            era_reference_count=3,
        )
        assert p.temporal_depth_era == "Before Time"
        assert p.temporal_depth_years_back == 20_000.0
        assert p.era_reference_count == 3

    def test_is_dialogue_default_false(self):
        p = make_passage()
        assert p.is_dialogue is False

    def test_speaker_ids_default_empty_list(self):
        p = make_passage()
        assert p.speaker_ids == []

    def test_style_metrics_defaults(self):
        p = make_passage()
        assert p.sentence_count == 0
        assert p.avg_sentence_length == 0.0
        assert p.passive_ratio == 0.0
        assert p.dialogue_density == 0.0
        assert p.archaic_word_count == 0

    def test_set_all_new_fields(self):
        p = make_passage(
            story_era="Third Age",
            story_year=3018,
            temporal_depth_era="First Age",
            temporal_depth_years_back=6000.0,
            era_reference_count=2,
            tolkien_register="archaic",
            pov_character_id="gandalf",
            is_dialogue=True,
            speaker_ids=["gandalf", "frodo"],
            sentence_count=4,
            avg_sentence_length=18.5,
            passive_ratio=0.25,
            dialogue_density=0.8,
            archaic_word_count=3,
        )
        assert p.tolkien_register == "archaic"
        assert p.pov_character_id == "gandalf"
        assert p.is_dialogue is True
        assert "gandalf" in p.speaker_ids
        assert "frodo" in p.speaker_ids
        assert p.sentence_count == 4
        assert p.avg_sentence_length == 18.5

    def test_temporal_summary_no_data(self):
        p = make_passage()
        assert p.temporal_summary() == "no temporal data"

    def test_temporal_summary_with_story_era(self):
        p = make_passage(story_era="Third Age", story_year=3018)
        summary = p.temporal_summary()
        assert "Third Age" in summary
        assert "3018" in summary

    def test_temporal_summary_full(self):
        p = make_passage(
            story_era="Third Age",
            story_year=3018,
            temporal_depth_era="Before Time",
            era_reference_count=3,
        )
        summary = p.temporal_summary()
        assert "Third Age" in summary
        assert "Before Time" in summary
        assert "3" in summary

    def test_model_copy_update_works(self):
        """Passage.model_copy(update=...) correctly propagates new temporal fields."""
        p = make_passage()
        p2 = p.model_copy(update={"temporal_depth_era": "First Age", "era_reference_count": 2})
        assert p2.temporal_depth_era == "First Age"
        assert p2.era_reference_count == 2
        # Original unchanged
        assert p.temporal_depth_era is None


# ---------------------------------------------------------------------------
# EraReference model
# ---------------------------------------------------------------------------

class TestEraReference:
    def test_basic_creation(self):
        ref = EraReference(
            passage_id="p_001",
            era="Second Age",
            reference_type="mentions",
            years_before_story_time=3400.0,
        )
        assert ref.era == "Second Age"
        assert ref.reference_type == "mentions"
        assert ref.years_before_story_time == 3400.0

    def test_default_reference_type_is_mentions(self):
        ref = EraReference(passage_id="p_001", era="First Age")
        assert ref.reference_type == "mentions"

    def test_all_reference_types_valid(self):
        for rt in ("mentions", "quotes", "alludes_to", "sets_scene_in"):
            ref = EraReference(passage_id="p_001", era="Third Age", reference_type=rt)
            assert ref.reference_type == rt

    def test_optional_entity_and_event_ids(self):
        ref = EraReference(
            passage_id="p_001",
            era="Second Age",
            entity_referenced_id="sauron",
            event_referenced_id="forging_of_the_one_ring",
        )
        assert ref.entity_referenced_id == "sauron"
        assert ref.event_referenced_id == "forging_of_the_one_ring"

    def test_to_neo4j_props_includes_reference_type(self):
        ref = EraReference(
            passage_id="p_001",
            era="Before Time",
            reference_type="alludes_to",
            years_before_story_time=20_000.0,
        )
        props = ref.to_neo4j_props()
        assert props["reference_type"] == "alludes_to"
        assert props["years_before_story_time"] == 20_000.0

    def test_to_neo4j_props_omits_none_values(self):
        ref = EraReference(passage_id="p_001", era="First Age")
        props = ref.to_neo4j_props()
        assert "entity_referenced_id" not in props
        assert "event_referenced_id" not in props
        assert "years_before_story_time" not in props

    def test_to_neo4j_props_includes_all_set_values(self):
        ref = EraReference(
            passage_id="p_002",
            era="Second Age",
            reference_type="quotes",
            entity_referenced_id="celebrimbor",
            event_referenced_id="forging_rings_of_power",
            years_before_story_time=3418.0,
        )
        props = ref.to_neo4j_props()
        assert props["entity_referenced_id"] == "celebrimbor"
        assert props["event_referenced_id"] == "forging_rings_of_power"
        assert props["years_before_story_time"] == 3418.0


# ---------------------------------------------------------------------------
# era_approx_years_back
# ---------------------------------------------------------------------------

class TestEraApproxYearsBack:
    def test_before_time_largest(self):
        bt = era_approx_years_back("Before Time")
        fa = era_approx_years_back("First Age")
        assert bt > fa

    def test_second_age_larger_than_third_age(self):
        sa = era_approx_years_back("Second Age")
        ta = era_approx_years_back("Third Age")
        assert sa > ta

    def test_none_returns_none(self):
        assert era_approx_years_back(None) is None

    def test_unknown_era_returns_none(self):
        assert era_approx_years_back("Completely Made Up Era") is None

    def test_alias_sa_works(self):
        sa_alias = era_approx_years_back("SA")
        sa_full = era_approx_years_back("Second Age")
        # Both should map to the same approximate value
        assert sa_alias is not None
        assert sa_full is not None
        assert abs(sa_alias - sa_full) < 1  # same value


# ---------------------------------------------------------------------------
# compute_temporal_zoom_batch (pure Python, no Neo4j)
# ---------------------------------------------------------------------------

class TestComputeTemporalZoomBatch:
    def test_empty_input_returns_empty_dict(self):
        result = compute_temporal_zoom_batch([])
        assert result == {}

    def test_single_passage_zoom_is_one(self):
        p = make_passage(id="p_001", temporal_depth_years_back=5000.0)
        result = compute_temporal_zoom_batch([p])
        assert "p_001" in result
        assert abs(result["p_001"] - 1.0) < 0.001

    def test_passages_without_depth_excluded(self):
        p1 = make_passage(id="p_001", temporal_depth_years_back=5000.0)
        p2 = make_passage(id="p_002")  # no temporal depth
        result = compute_temporal_zoom_batch([p1, p2])
        assert "p_001" in result
        assert "p_002" not in result

    def test_deeper_passage_has_higher_zoom(self):
        p_shallow = make_passage(id="shallow", temporal_depth_years_back=500.0)
        p_deep = make_passage(id="deep", temporal_depth_years_back=10_000.0)
        result = compute_temporal_zoom_batch([p_shallow, p_deep])
        assert result["deep"] > result["shallow"]

    def test_zoom_sum_reflects_corpus_mean(self):
        """
        If depths are [1000, 2000, 3000], mean = 2000.
        Zooms should be [0.5, 1.0, 1.5].
        """
        passages = [
            make_passage(id=f"p_{i}", temporal_depth_years_back=float(v))
            for i, v in enumerate([1000.0, 2000.0, 3000.0])
        ]
        result = compute_temporal_zoom_batch(passages)
        assert abs(result["p_0"] - 0.5) < 0.001
        assert abs(result["p_1"] - 1.0) < 0.001
        assert abs(result["p_2"] - 1.5) < 0.001

    def test_all_passages_no_depth_returns_empty(self):
        passages = [make_passage(id=f"p_{i}") for i in range(3)]
        result = compute_temporal_zoom_batch(passages)
        assert result == {}

    def test_zero_depth_corpus_handles_gracefully(self):
        """All passages with depth=0 — zoom should be 0 or handled without ZeroDivisionError."""
        passages = [make_passage(id=f"p_{i}", temporal_depth_years_back=0.0) for i in range(3)]
        result = compute_temporal_zoom_batch(passages)
        # Zoom of 0 for all, no ZeroDivisionError
        for pid, zoom in result.items():
            assert zoom == 0.0


# ---------------------------------------------------------------------------
# PassageTemporalWriter — pure-Python helpers (no Neo4j)
# ---------------------------------------------------------------------------

class TestPassageTemporalWriterHelpers:
    """Tests for PassageTemporalWriter methods that don't touch Neo4j."""

    def setup_method(self):
        # Create writer without driver — pure-Python methods don't need it
        self.writer = PassageTemporalWriter(driver=None)

    def test_temporal_depth_color_no_data(self):
        p = make_passage()
        color = self.writer.temporal_depth_visualization_color(p)
        assert color == "#f5f5f5"

    def test_temporal_depth_color_before_time_darkest(self):
        p_ancient = make_passage(temporal_depth_era="Before Time")
        p_recent = make_passage(temporal_depth_era="Fourth Age")
        color_ancient = self.writer.temporal_depth_visualization_color(p_ancient)
        color_recent = self.writer.temporal_depth_visualization_color(p_recent)
        # Ancient should be darker (lower total RGB sum)
        def rgb_sum(hex_color: str) -> int:
            h = hex_color.lstrip("#")
            return int(h[0:2], 16) + int(h[2:4], 16) + int(h[4:6], 16)
        assert rgb_sum(color_ancient) < rgb_sum(color_recent)

    def test_temporal_depth_color_valid_hex(self):
        for era in ["Before Time", "First Age", "Second Age", "Third Age", "Fourth Age"]:
            p = make_passage(temporal_depth_era=era)
            color = self.writer.temporal_depth_visualization_color(p)
            assert color.startswith("#")
            assert len(color) == 7
            # Ensure it's valid hex
            int(color[1:], 16)

    def test_upsert_passage_with_references_auto_derives_depth(self):
        """When temporal_depth_era is None but references exist, depth is auto-derived."""
        # We'll monkey-patch the Neo4j session to capture what would be written
        calls = []

        class MockSession:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def run(self, query, **kwargs): calls.append((query, kwargs))

        class MockDriver:
            def session(self): return MockSession()

        writer = PassageTemporalWriter(driver=MockDriver())

        p = make_passage(story_era="Third Age", story_year=3018)
        refs = [
            EraReference(passage_id="p_001", era="Second Age", reference_type="mentions", years_before_story_time=3400.0),
            EraReference(passage_id="p_001", era="First Age", reference_type="alludes_to", years_before_story_time=6000.0),
        ]

        writer.upsert_passage_with_references(p, refs)

        # Check that passage was written with depth info derived from refs
        # The first call should be the passage MERGE
        passage_call = calls[0]
        props = passage_call[1].get("props", {})
        assert props.get("temporal_depth_era") == "First Age"  # oldest era
        assert props.get("temporal_depth_years_back") == 6000.0  # max years back
        assert props.get("era_reference_count") == 2  # two distinct eras

    def test_upsert_passage_with_references_preserves_existing_depth(self):
        """If passage already has temporal_depth_era set, it is not overridden."""
        calls = []

        class MockSession:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def run(self, query, **kwargs): calls.append((query, kwargs))

        class MockDriver:
            def session(self): return MockSession()

        writer = PassageTemporalWriter(driver=MockDriver())

        p = make_passage(
            story_era="Third Age",
            story_year=3018,
            temporal_depth_era="Before Time",  # already set
            temporal_depth_years_back=20_000.0,
        )
        refs = [EraReference(passage_id="p_001", era="Second Age")]

        writer.upsert_passage_with_references(p, refs)

        props = calls[0][1].get("props", {})
        # The pre-set value should be preserved
        assert props.get("temporal_depth_era") == "Before Time"


# ---------------------------------------------------------------------------
# TemporalZoomResult model
# ---------------------------------------------------------------------------

class TestTemporalZoomResult:
    def test_summary_minimal(self):
        r = TemporalZoomResult(passage_id="p_001", passage_text="Hello world")
        s = r.summary()
        assert "p_001" in s

    def test_summary_with_full_data(self):
        r = TemporalZoomResult(
            passage_id="p_001",
            passage_text="Gandalf told Frodo of the forging of the Rings in the Second Age.",
            story_era="Third Age",
            story_year=3018,
            temporal_depth_era="Second Age",
            temporal_depth_years_back=3400.0,
            era_reference_count=1,
            temporal_zoom=2.5,
        )
        s = r.summary()
        assert "Third Age" in s
        assert "Second Age" in s
        assert "3,400" in s or "3400" in s  # formatted with or without thousands separator
        assert "2.50" in s

    def test_summary_long_text_truncated(self):
        long_text = "x" * 200
        r = TemporalZoomResult(passage_id="p_001", passage_text=long_text)
        s = r.summary()
        assert "..." in s

    def test_references_list_empty_by_default(self):
        r = TemporalZoomResult(passage_id="p_001", passage_text="hi")
        assert r.references == []
