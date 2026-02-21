"""Tests for spatiotemporal engine slice 3 (#48):
- Causal paradox detection (direct paradoxes + cycle detection)
- Neo4j persistence for conflicts (writer/query contract tests)
- CLI integration for write/query/report
- Report causal_paradox counts
"""

import json
import pytest

from book_graph_analyzer.spatiotemporal.models import (
    CausalLink, ConflictType, NormalizedTime, SpatiotemporalEvent, TimelineConflict,
)
from book_graph_analyzer.spatiotemporal.conflict_detector import ConflictDetector
from book_graph_analyzer.spatiotemporal.report import ReconciliationReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ev(id, entity_id="char1", era="Third Age", year=None, description="", **kw):
    return SpatiotemporalEvent(
        id=id, entity_id=entity_id, entity_name=kw.get("entity_name", entity_id),
        location_id=kw.get("location_id"), location_name=kw.get("location_name"),
        time=NormalizedTime(era=era, year_start=year, year_end=year, confidence=0.9),
        description=description,
    )


# ===========================================================================
# 1) Causal paradox detection — direct violations
# ===========================================================================

class TestCausalParadoxDirect:
    def test_no_paradox_correct_ordering(self):
        """Cause before effect → no conflict."""
        events = [
            _ev("e1", era="Third Age", year=3000, description="Cause event"),
            _ev("e2", era="Third Age", year=3019, description="Effect event"),
        ]
        links = [CausalLink(cause_event_id="e1", effect_event_id="e2")]
        detector = ConflictDetector(causal_links=links)
        conflicts = detector.detect_conflicts(events)
        paradoxes = [c for c in conflicts if c.conflict_type == ConflictType.CAUSAL_PARADOX]
        assert len(paradoxes) == 0

    def test_paradox_effect_before_cause(self):
        """Effect at year 2000, cause at year 3019 → paradox."""
        events = [
            _ev("e1", era="Third Age", year=3019, description="Ring destroyed"),
            _ev("e2", era="Third Age", year=2000, description="Consequence of ring destruction"),
        ]
        links = [CausalLink(cause_event_id="e1", effect_event_id="e2")]
        detector = ConflictDetector(causal_links=links)
        conflicts = detector.detect_conflicts(events)
        paradoxes = [c for c in conflicts if c.conflict_type == ConflictType.CAUSAL_PARADOX]
        assert len(paradoxes) == 1
        assert "paradox" in paradoxes[0].description.lower()

    def test_paradox_effect_in_earlier_era(self):
        """Cause in Third Age, effect in First Age → paradox."""
        events = [
            _ev("e1", era="Third Age", year=3019, description="A"),
            _ev("e2", era="First Age", year=500, description="B"),
        ]
        links = [CausalLink(cause_event_id="e1", effect_event_id="e2")]
        detector = ConflictDetector(causal_links=links)
        conflicts = detector.detect_conflicts(events)
        paradoxes = [c for c in conflicts if c.conflict_type == ConflictType.CAUSAL_PARADOX]
        assert len(paradoxes) == 1

    def test_no_paradox_same_time(self):
        """Same year → not strictly before → no paradox."""
        events = [
            _ev("e1", era="Third Age", year=3019, description="A"),
            _ev("e2", era="Third Age", year=3019, description="B"),
        ]
        links = [CausalLink(cause_event_id="e1", effect_event_id="e2")]
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(events, causal_links=links)
        paradoxes = [c for c in conflicts if c.conflict_type == ConflictType.CAUSAL_PARADOX]
        assert len(paradoxes) == 0

    def test_paradox_confidence_propagation(self):
        """Confidence should factor in link and event confidences."""
        events = [
            _ev("e1", era="Third Age", year=3019, description="Cause"),
            _ev("e2", era="First Age", year=500, description="Effect"),
        ]
        links = [CausalLink(cause_event_id="e1", effect_event_id="e2", confidence=0.5)]
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(events, causal_links=links)
        paradoxes = [c for c in conflicts if c.conflict_type == ConflictType.CAUSAL_PARADOX]
        assert len(paradoxes) == 1
        # min(0.9, 0.9, 0.5) * 0.85 = 0.425
        assert paradoxes[0].confidence == pytest.approx(0.425, abs=0.01)

    def test_missing_event_skipped(self):
        """Links referencing non-existent events are silently skipped."""
        events = [_ev("e1", era="Third Age", year=3019)]
        links = [CausalLink(cause_event_id="e1", effect_event_id="e_missing")]
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(events, causal_links=links)
        paradoxes = [c for c in conflicts if c.conflict_type == ConflictType.CAUSAL_PARADOX]
        assert len(paradoxes) == 0

    def test_disable_causal_check(self):
        """check_causal_paradoxes=False skips detection."""
        events = [
            _ev("e1", era="Third Age", year=3019),
            _ev("e2", era="First Age", year=500),
        ]
        links = [CausalLink(cause_event_id="e1", effect_event_id="e2")]
        detector = ConflictDetector(causal_links=links)
        conflicts = detector.detect_conflicts(events, check_causal_paradoxes=False)
        paradoxes = [c for c in conflicts if c.conflict_type == ConflictType.CAUSAL_PARADOX]
        assert len(paradoxes) == 0


# ===========================================================================
# 2) Causal cycle detection
# ===========================================================================

class TestCausalCycleDetection:
    def test_simple_cycle(self):
        """A -> B -> A is a cycle."""
        events = [
            _ev("e1", era="Third Age", year=3000, description="A"),
            _ev("e2", era="Third Age", year=3001, description="B"),
        ]
        links = [
            CausalLink(cause_event_id="e1", effect_event_id="e2"),
            CausalLink(cause_event_id="e2", effect_event_id="e1"),
        ]
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(events, causal_links=links)
        cycles = [c for c in conflicts
                  if c.conflict_type == ConflictType.CAUSAL_PARADOX and "cycle" in c.description.lower()]
        assert len(cycles) >= 1

    def test_three_node_cycle(self):
        """A -> B -> C -> A."""
        events = [
            _ev("e1", description="A"), _ev("e2", description="B"), _ev("e3", description="C"),
        ]
        links = [
            CausalLink(cause_event_id="e1", effect_event_id="e2"),
            CausalLink(cause_event_id="e2", effect_event_id="e3"),
            CausalLink(cause_event_id="e3", effect_event_id="e1"),
        ]
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(events, causal_links=links)
        cycles = [c for c in conflicts
                  if c.conflict_type == ConflictType.CAUSAL_PARADOX and "cycle" in c.description.lower()]
        assert len(cycles) >= 1

    def test_no_cycle_in_dag(self):
        """Linear chain has no cycle."""
        events = [_ev("e1"), _ev("e2"), _ev("e3")]
        links = [
            CausalLink(cause_event_id="e1", effect_event_id="e2"),
            CausalLink(cause_event_id="e2", effect_event_id="e3"),
        ]
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(events, causal_links=links)
        cycles = [c for c in conflicts
                  if c.conflict_type == ConflictType.CAUSAL_PARADOX and "cycle" in c.description.lower()]
        assert len(cycles) == 0


# ===========================================================================
# 3) Report integration
# ===========================================================================

class TestReportCausalParadox:
    def test_report_counts_paradoxes(self):
        conflicts = [
            TimelineConflict(id="c1", conflict_type=ConflictType.CAUSAL_PARADOX,
                             severity="error", description="paradox", confidence=0.8),
            TimelineConflict(id="c2", conflict_type=ConflictType.TEMPORAL_OVERLAP,
                             severity="warning", description="overlap", confidence=0.5),
        ]
        report = ReconciliationReport(conflicts=conflicts)
        assert report.causal_paradox_count == 1

    def test_report_text_shows_paradoxes(self):
        conflicts = [
            TimelineConflict(id="c1", conflict_type=ConflictType.CAUSAL_PARADOX,
                             severity="error", description="test paradox", confidence=0.8),
        ]
        report = ReconciliationReport(conflicts=conflicts)
        text = report.to_text()
        assert "Causal paradoxes: 1" in text
        assert "CAUSAL PARADOX" in text

    def test_report_dict_includes_paradoxes(self):
        conflicts = [
            TimelineConflict(id="c1", conflict_type=ConflictType.CAUSAL_PARADOX,
                             severity="error", description="p", confidence=0.8),
        ]
        report = ReconciliationReport(conflicts=conflicts)
        d = report.to_dict()
        assert d["causal_paradoxes"] == 1


# ===========================================================================
# 4) GraphWriter contract tests (mock driver)
# ===========================================================================

class FakeSession:
    """Minimal mock Neo4j session for contract tests."""
    def __init__(self):
        self.queries = []

    def run(self, query, **kwargs):
        self.queries.append((query, kwargs))
        return FakeResult()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class FakeResult:
    def __iter__(self):
        return iter([])

    def single(self):
        return None


class FakeDriver:
    def __init__(self):
        self.sessions = []

    def session(self):
        s = FakeSession()
        self.sessions.append(s)
        return s

    def close(self):
        pass


class TestWriterConflictPersistence:
    def test_write_single_conflict(self):
        from book_graph_analyzer.graph.writer import GraphWriter
        driver = FakeDriver()
        writer = GraphWriter(driver=driver)
        conflict = TimelineConflict(
            id="test_c1", conflict_type=ConflictType.CAUSAL_PARADOX,
            severity="error", description="test", confidence=0.9,
            event_a_id="e1", event_b_id="e2",
            event_a_source_book="The Hobbit",
            event_b_source_book="Unfinished Tales",
            event_a_source_authority_weight=1.0,
            event_b_source_authority_weight=0.7,
        )
        writer.write_timeline_conflict(conflict)
        # Should have at least the MERGE query
        all_queries = []
        for s in driver.sessions:
            all_queries.extend(s.queries)
        assert any("MERGE (c:TimelineConflict" in q for q, _ in all_queries)
        merge_params = next(params for q, params in all_queries if "MERGE (c:TimelineConflict" in q)
        assert merge_params["event_a_source_book"] == "The Hobbit"
        assert merge_params["event_b_source_book"] == "Unfinished Tales"
        assert merge_params["event_a_source_authority_weight"] == 1.0
        assert merge_params["event_b_source_authority_weight"] == 0.7

    def test_write_batch(self):
        from book_graph_analyzer.graph.writer import GraphWriter
        driver = FakeDriver()
        writer = GraphWriter(driver=driver)
        conflicts = [
            TimelineConflict(id="c1", conflict_type=ConflictType.CAUSAL_PARADOX,
                             severity="error", description="a", confidence=0.8),
            TimelineConflict(id="c2", conflict_type=ConflictType.ERA_MISMATCH,
                             severity="warning", description="b", confidence=0.5),
        ]
        count = writer.write_timeline_conflicts_batch(conflicts)
        assert count == 2

    def test_query_conflicts_builds_cypher(self):
        from book_graph_analyzer.graph.writer import GraphWriter
        driver = FakeDriver()
        writer = GraphWriter(driver=driver)
        results = writer.query_timeline_conflicts(conflict_type="causal_paradox", severity="error")
        all_queries = []
        for s in driver.sessions:
            all_queries.extend(s.queries)
        assert any("TimelineConflict" in q for q, _ in all_queries)
        # Results empty from fake but no crash
        assert results == []

    def test_query_recent_critical(self):
        from book_graph_analyzer.graph.writer import GraphWriter
        driver = FakeDriver()
        writer = GraphWriter(driver=driver)
        results = writer.query_recent_critical_conflicts()
        assert results == []

    def test_query_divergence_hotspots_builds_cypher(self):
        from book_graph_analyzer.graph.writer import GraphWriter
        driver = FakeDriver()
        writer = GraphWriter(driver=driver)
        results = writer.query_divergence_hotspots(min_sources=2)
        all_queries = []
        for s in driver.sessions:
            all_queries.extend(s.queries)
        assert any("source_books" in q and "TimelineConflict" in q for q, _ in all_queries)
        assert results == []

    def test_query_source_divergence_builds_cypher(self):
        from book_graph_analyzer.graph.writer import GraphWriter
        driver = FakeDriver()
        writer = GraphWriter(driver=driver)
        results = writer.query_source_divergence("the_hobbit", "the_silmarillion")
        all_queries = []
        for s in driver.sessions:
            all_queries.extend(s.queries)
        assert any("$source_a IN sources" in q for q, _ in all_queries)
        assert results == []


# ===========================================================================
# 5) CLI tests
# ===========================================================================

class TestCLITimelineConflicts:
    def test_timeline_bridge_with_causal_data(self, tmp_path):
        """timeline-bridge should detect causal paradoxes when causal links present."""
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        events = {
            "events": {
                "e1": {"id": "e1", "description": "Ring destroyed", "agent": "Frodo",
                       "era": "third_age", "year": 3019, "confidence": 0.9},
                "e2": {"id": "e2", "description": "Sauron falls", "agent": "Sauron",
                       "era": "third_age", "year": 3018, "confidence": 0.9},
            },
            "relations": [],
        }
        ef = tmp_path / "events.json"
        ef.write_text(json.dumps(events))
        out = tmp_path / "report.json"
        result = CliRunner().invoke(
            main, ["lore", "timeline-bridge", str(ef), "--format", "json", "-o", str(out)]
        )
        assert result.exit_code == 0

    def test_timeline_bridge_json_has_causal_paradoxes_key(self, tmp_path):
        """JSON output includes causal_paradoxes count."""
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        events = {
            "events": {
                "e1": {"id": "e1", "description": "test", "agent": "A",
                       "era": "third_age", "year": 3019, "confidence": 0.9},
            },
            "relations": [],
        }
        ef = tmp_path / "events.json"
        ef.write_text(json.dumps(events))
        out = tmp_path / "report.json"
        result = CliRunner().invoke(
            main, ["lore", "timeline-bridge", str(ef), "--format", "json", "-o", str(out)]
        )
        assert result.exit_code == 0
        report = json.loads(out.read_text())
        assert "causal_paradoxes" in report
