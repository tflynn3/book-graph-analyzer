"""Tests for spatiotemporal engine slice 5 (#48):
- LLM-assisted causal extraction (with fallback)
- Confidence calibration with source authority weights
- Location graph seeding MVP
- Report integration (extraction mode + calibration surfaced)
"""

import json
import pytest

from book_graph_analyzer.spatiotemporal.models import (
    CausalLink, NormalizedTime, SpatiotemporalEvent, TimelineConflict,
    ConflictType,
)
from book_graph_analyzer.spatiotemporal.llm_causal_extraction import (
    extract_causal_links,
    CausalExtractionResult,
    ExtractionMode,
    _parse_llm_response,
)
from book_graph_analyzer.spatiotemporal.confidence import (
    SourceAuthorityRegistry,
    CalibrationResult,
    calibrate_event_confidence,
    calibrate_causal_link_confidence,
    calibrate_conflict_confidence,
)
from book_graph_analyzer.spatiotemporal.location_seeds import (
    load_seed_locations,
    load_seed_edges,
    load_seed_location_graph,
)
from book_graph_analyzer.spatiotemporal.report import ReconciliationReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ev(id, entity_id="char1", era="Third Age", year=None, description="",
        source_book=None, **kw):
    return SpatiotemporalEvent(
        id=id, entity_id=entity_id, entity_name=kw.get("entity_name", entity_id),
        location_id=kw.get("location_id"), location_name=kw.get("location_name"),
        time=NormalizedTime(era=era, year_start=year, year_end=year, confidence=0.9),
        description=description, source_book=source_book,
    )


# ===========================================================================
# 1. LLM-Assisted Causal Extraction
# ===========================================================================

class TestLLMCausalExtraction:
    def test_heuristic_mode_when_no_llm(self):
        events = [
            _ev("e1", year=3018, description="War caused destruction"),
            _ev("e2", year=3019, description="As a result of war, rebuilding began"),
        ]
        result = extract_causal_links(events, use_llm=False)
        assert result.mode == ExtractionMode.HEURISTIC
        assert result.event_count == 2

    def test_empty_events(self):
        result = extract_causal_links([], use_llm=False)
        assert result.mode == ExtractionMode.HEURISTIC
        assert result.links == []
        assert result.event_count == 0

    def test_llm_fallback_on_no_client(self):
        """use_llm=True but client=None should use heuristic."""
        events = [
            _ev("e1", year=1, description="X caused Y"),
            _ev("e2", year=2, description="Y happened"),
        ]
        result = extract_causal_links(events, use_llm=True, llm_client=None)
        assert result.mode == ExtractionMode.HEURISTIC

    def test_llm_fallback_on_exception(self):
        """LLM client that raises should fall back to heuristic."""
        class BrokenLLM:
            def generate(self, prompt):
                raise RuntimeError("LLM unavailable")

        events = [
            _ev("e1", year=1, description="X caused Y"),
            _ev("e2", year=2, description="Y resulted in Z"),
        ]
        result = extract_causal_links(events, use_llm=True, llm_client=BrokenLLM())
        assert result.mode == ExtractionMode.LLM_FALLBACK_HEURISTIC

    def test_llm_fallback_on_empty_response(self):
        """LLM returns empty/invalid JSON -> falls back."""
        class EmptyLLM:
            def generate(self, prompt):
                return "not json"

        events = [
            _ev("e1", year=1, description="X caused Y"),
            _ev("e2", year=2, description="Y happened"),
        ]
        result = extract_causal_links(events, use_llm=True, llm_client=EmptyLLM())
        assert result.mode == ExtractionMode.LLM_FALLBACK_HEURISTIC

    def test_llm_success_path(self):
        """LLM returns valid JSON -> uses LLM mode."""
        class GoodLLM:
            def generate(self, prompt):
                return json.dumps([{
                    "cause_event_id": "e1",
                    "effect_event_id": "e2",
                    "description": "X caused Y",
                    "confidence": 0.85,
                }])

        events = [
            _ev("e1", year=1, description="X"),
            _ev("e2", year=2, description="Y"),
        ]
        result = extract_causal_links(events, use_llm=True, llm_client=GoodLLM())
        assert result.mode == ExtractionMode.LLM
        assert len(result.links) == 1
        assert result.links[0].confidence == 0.85

    def test_llm_filters_invalid_ids(self):
        """LLM references nonexistent event IDs -> those links are dropped."""
        class BadIdLLM:
            def generate(self, prompt):
                return json.dumps([
                    {"cause_event_id": "e1", "effect_event_id": "e999",
                     "description": "bad", "confidence": 0.9},
                    {"cause_event_id": "e1", "effect_event_id": "e2",
                     "description": "good", "confidence": 0.8},
                ])

        events = [_ev("e1", year=1), _ev("e2", year=2)]
        result = extract_causal_links(events, use_llm=True, llm_client=BadIdLLM())
        assert result.mode == ExtractionMode.LLM
        assert len(result.links) == 1
        assert result.links[0].effect_event_id == "e2"

    def test_result_to_dict(self):
        result = CausalExtractionResult(
            links=[CausalLink(cause_event_id="a", effect_event_id="b", confidence=0.7)],
            mode=ExtractionMode.LLM,
            event_count=5,
        )
        d = result.to_dict()
        assert d["extraction_mode"] == "llm"
        assert d["link_count"] == 1

    def test_parse_llm_response_with_markdown_fences(self):
        resp = "```json\n[{\"cause_event_id\":\"e1\",\"effect_event_id\":\"e2\",\"confidence\":0.9}]\n```"
        links = _parse_llm_response(resp, {"e1", "e2"})
        assert len(links) == 1

    def test_llm_batch_mode_for_large_event_sets(self):
        class BatchLLM:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt):
                self.calls += 1
                if "\"id\": \"e1\"" in prompt:
                    return json.dumps([{"cause_event_id": "e1", "effect_event_id": "e2", "confidence": 0.8}])
                return json.dumps([{"cause_event_id": "e5", "effect_event_id": "e6", "confidence": 0.75}])

        events = [_ev(f"e{i}", year=i, description=f"event {i}") for i in range(1, 7)]
        llm = BatchLLM()
        result = extract_causal_links(events, use_llm=True, llm_client=llm, llm_batch_size=4)
        assert result.mode == ExtractionMode.LLM
        assert llm.calls == 2
        assert len(result.links) == 2


# ===========================================================================
# 2. Confidence Calibration
# ===========================================================================

class TestConfidenceCalibration:
    def test_default_registry_no_change(self):
        """With empty registry, authority=1.0, confidence unchanged."""
        events = [_ev("e1", year=1, source_book="unknown_book")]
        events[0].time.confidence = 0.8
        result = calibrate_event_confidence(events)
        assert events[0].time.confidence == 0.8
        assert result.events_calibrated == 1

    def test_low_authority_reduces_confidence(self):
        reg = SourceAuthorityRegistry(weights={"draft_notes": 0.5})
        events = [_ev("e1", year=1, source_book="draft_notes")]
        events[0].time.confidence = 0.8
        calibrate_event_confidence(events, reg)
        assert events[0].time.confidence == pytest.approx(0.4, abs=0.01)

    def test_high_authority_preserves_confidence(self):
        reg = SourceAuthorityRegistry(weights={"lotr": 1.0})
        events = [_ev("e1", year=1, source_book="lotr")]
        events[0].time.confidence = 0.9
        calibrate_event_confidence(events, reg)
        assert events[0].time.confidence == pytest.approx(0.9, abs=0.01)

    def test_causal_link_calibration(self):
        reg = SourceAuthorityRegistry(weights={"weak_source": 0.5})
        events = [
            _ev("e1", year=1, source_book="weak_source"),
            _ev("e2", year=2, source_book="weak_source"),
        ]
        links = [CausalLink(cause_event_id="e1", effect_event_id="e2", confidence=0.8)]
        result = calibrate_causal_link_confidence(links, events, reg)
        assert links[0].confidence == pytest.approx(0.4, abs=0.01)
        assert result.links_calibrated == 1

    def test_conflict_calibration(self):
        reg = SourceAuthorityRegistry(weights={"strong": 1.0})
        events = [
            _ev("e1", year=1, source_book="strong"),
            _ev("e2", year=1, source_book="strong"),
        ]
        conflicts = [TimelineConflict(
            id="c1", conflict_type=ConflictType.TEMPORAL_OVERLAP,
            event_a_id="e1", event_b_id="e2", confidence=0.6,
        )]
        result = calibrate_conflict_confidence(conflicts, events, reg)
        assert conflicts[0].confidence == pytest.approx(0.6, abs=0.01)
        assert result.conflicts_calibrated == 1

    def test_from_editorial_layers(self):
        from book_graph_analyzer.models.worldbuilding import TOLKIEN_SOURCES
        reg = SourceAuthorityRegistry.from_editorial_layers(TOLKIEN_SOURCES)
        # "The Hobbit" -> 1.0, "Unfinished Tales" -> 0.7
        assert reg.get("the_hobbit") == 1.0
        assert reg.get("unfinished_tales") == pytest.approx(0.7, abs=0.01)

    def test_source_title_lookup_is_case_and_whitespace_insensitive(self):
        from book_graph_analyzer.models.worldbuilding import TOLKIEN_SOURCES

        reg = SourceAuthorityRegistry.from_editorial_layers(TOLKIEN_SOURCES)

        assert reg.get("The Silmarillion") == reg.get("the silmarillion")
        assert reg.get("  UNFINISHED   TALES ") == pytest.approx(0.7, abs=0.01)

    def test_default_tolkien_registry(self):
        reg = SourceAuthorityRegistry.default_tolkien()
        assert reg.get("src_hobbit") == 1.0
        assert reg.get("src_unfinished_tales") == pytest.approx(0.7, abs=0.01)

    def test_missing_source_uses_default(self):
        reg = SourceAuthorityRegistry(weights={"known": 0.5})
        assert reg.get("unknown") == 1.0
        assert reg.get(None) == 1.0


# ===========================================================================
# 3. Location Graph Seeding
# ===========================================================================

class TestLocationSeeding:
    def test_load_seed_locations_returns_nodes(self):
        locations = load_seed_locations()
        assert len(locations) > 0
        assert "rivendell" in locations or "the_shire" in locations

    def test_seed_locations_have_coordinates(self):
        locations = load_seed_locations()
        if "the_shire" in locations:
            loc = locations["the_shire"]
            assert loc.x != 0.0 or loc.y != 0.0
            assert loc.name == "The Shire"

    def test_seed_locations_have_aliases(self):
        locations = load_seed_locations()
        if "rivendell" in locations:
            assert "Imladris" in locations["rivendell"].aliases

    def test_load_seed_edges(self):
        edges = load_seed_edges()
        assert len(edges) > 0
        assert any(e.source_id == "the_shire" for e in edges)

    def test_load_seed_location_graph(self):
        locations, edges = load_seed_location_graph()
        assert len(locations) > 0
        assert len(edges) > 0

    def test_distance_between_locations(self):
        locations = load_seed_locations()
        if "the_shire" in locations and "mordor" in locations:
            dist = locations["the_shire"].distance_to(locations["mordor"])
            assert dist > 0

    def test_nonexistent_seeds_path(self, tmp_path):
        locations = load_seed_locations(tmp_path / "nonexistent.json")
        assert locations == {}


# ===========================================================================
# 4. Report Integration
# ===========================================================================

class TestReportIntegration:
    def test_report_surfaces_extraction_mode(self):
        result = CausalExtractionResult(
            links=[], mode=ExtractionMode.LLM, event_count=3,
        )
        report = ReconciliationReport(
            conflicts=[], events=[], causal_result=result,
        )
        text = report.to_text()
        assert "CAUSAL EXTRACTION" in text
        assert "llm" in text

        d = report.to_dict()
        assert "causal_extraction" in d
        assert d["causal_extraction"]["extraction_mode"] == "llm"

    def test_report_surfaces_calibration(self):
        cal = CalibrationResult(
            events_calibrated=5, links_calibrated=2,
            conflicts_calibrated=1, avg_authority_weight=0.85,
        )
        report = ReconciliationReport(
            conflicts=[], events=[], calibration=cal,
        )
        text = report.to_text()
        assert "CONFIDENCE CALIBRATION" in text
        assert "0.850" in text

        d = report.to_dict()
        assert "confidence_calibration" in d

    def test_report_backward_compatible_no_extras(self):
        """Report works fine without causal_result or calibration."""
        report = ReconciliationReport(conflicts=[], events=[])
        text = report.to_text()
        assert "CAUSAL EXTRACTION" not in text
        assert "CONFIDENCE CALIBRATION" not in text
        d = report.to_dict()
        assert "causal_extraction" not in d
        assert "confidence_calibration" not in d

    def test_report_includes_source_attribution(self):
        events = [
            _ev("e1", year=1, source_book="The Hobbit"),
            _ev("e2", year=2, source_book="The Hobbit"),
            _ev("e3", year=3, source_book="Unfinished Tales"),
        ]
        report = ReconciliationReport(conflicts=[], events=events)
        text = report.to_text()
        assert "SOURCE ATTRIBUTION" in text
        d = report.to_dict()
        assert d["source_attribution"]["The Hobbit"] == 2


# ===========================================================================
# 5. CLI Integration
# ===========================================================================

class TestCLISlice5:
    def test_timeline_bridge_with_seed_locations(self, tmp_path):
        """Test --seed-locations flag."""
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        events = {
            "events": {
                "e1": {"id": "e1", "description": "Bilbo left the Shire",
                       "agent": "Bilbo", "action": "left", "patient": "Shire",
                       "era": "third_age", "year": 2941, "confidence": 0.9},
            },
            "relations": [],
        }
        events_file = tmp_path / "events.json"
        events_file.write_text(json.dumps(events), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, [
            "lore", "timeline-bridge", str(events_file),
            "--seed-locations", "--format", "json",
        ])
        assert result.exit_code == 0
        assert "seed locations" in result.output.lower() or "Loaded" in result.output

    def test_timeline_bridge_with_calibrate(self, tmp_path):
        """Test --calibrate flag."""
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        events = {
            "events": {
                "e1": {"id": "e1", "description": "Event",
                       "agent": "X", "action": "did", "patient": "Y",
                       "era": "third_age", "year": 3019, "confidence": 0.9},
            },
            "relations": [],
        }
        events_file = tmp_path / "events.json"
        events_file.write_text(json.dumps(events), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, [
            "lore", "timeline-bridge", str(events_file),
            "--calibrate", "--format", "text",
        ])
        assert result.exit_code == 0

    def test_timeline_bridge_causal_links_use_llm_no_client(self, tmp_path):
        """Test --use-llm without available LLM (should fall back gracefully)."""
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        events = {
            "events": {
                "e1": {"id": "e1", "description": "War caused destruction",
                       "agent": "Sauron", "action": "caused", "patient": "destruction",
                       "era": "third_age", "year": 3019, "confidence": 0.9},
                "e2": {"id": "e2", "description": "As a result rebuilding began",
                       "agent": "Aragorn", "action": "rebuilt", "patient": "city",
                       "era": "third_age", "year": 3020, "confidence": 0.8},
            },
            "relations": [],
        }
        events_file = tmp_path / "events.json"
        events_file.write_text(json.dumps(events), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, [
            "lore", "timeline-bridge", str(events_file),
            "--causal-links", "--use-llm", "--format", "json",
        ])
        # Should succeed (fallback to heuristic)
        assert result.exit_code == 0
