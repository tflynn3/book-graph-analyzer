"""Tests for spatiotemporal engine (Issue #48)."""

import json
import pytest

from book_graph_analyzer.spatiotemporal.models import (
    ConflictType, LocationEdge, LocationNode,
    NormalizedTime, SpatiotemporalEvent, TimelineConflict,
)
from book_graph_analyzer.spatiotemporal.normalizer import TimeNormalizer
from book_graph_analyzer.spatiotemporal.conflict_detector import ConflictDetector
from book_graph_analyzer.spatiotemporal.report import ReconciliationReport


class TestNormalizedTime:
    def test_precise_time(self):
        t = NormalizedTime(era="Third Age", year_start=3019, year_end=3019, confidence=0.95)
        assert t.is_precise
        assert t.midpoint == 3019.0

    def test_fuzzy_time(self):
        t = NormalizedTime(era="Third Age", year_start=3000, year_end=3020, confidence=0.4)
        assert not t.is_precise
        assert t.midpoint == 3010.0

    def test_overlaps_same_era(self):
        a = NormalizedTime(era="Third Age", year_start=3000, year_end=3010)
        b = NormalizedTime(era="Third Age", year_start=3005, year_end=3015)
        assert a.overlaps(b) and b.overlaps(a)

    def test_no_overlap_same_era(self):
        a = NormalizedTime(era="Third Age", year_start=3000, year_end=3005)
        b = NormalizedTime(era="Third Age", year_start=3010, year_end=3020)
        assert not a.overlaps(b)

    def test_no_overlap_different_era(self):
        a = NormalizedTime(era="First Age", year_start=500, year_end=510)
        b = NormalizedTime(era="Third Age", year_start=500, year_end=510)
        assert not a.overlaps(b)

    def test_overlap_unknown_bounds(self):
        a = NormalizedTime(era="Third Age", year_start=3000, year_end=3010)
        b = NormalizedTime(era="Third Age")
        assert a.overlaps(b)

    def test_to_dict(self):
        t = NormalizedTime(era="Third Age", year_start=3019, year_end=3019, confidence=0.9)
        d = t.to_dict()
        assert d["era"] == "Third Age" and d["year_start"] == 3019


class TestTimeNormalizer:
    def setup_method(self):
        self.norm = TimeNormalizer()

    def test_ta_year(self):
        t = self.norm.normalize("TA 3019")
        assert t.era == "Third Age" and t.year_start == 3019 and t.confidence >= 0.8

    def test_third_age_year(self):
        t = self.norm.normalize("Third Age 2941")
        assert t.era == "Third Age" and t.year_start == 2941

    def test_sa_year(self):
        t = self.norm.normalize("SA 3441")
        assert t.era == "Second Age" and t.year_start == 3441

    def test_first_age_year(self):
        t = self.norm.normalize("First Age 587")
        assert t.era == "First Age" and t.year_start == 587

    def test_fuzzy_era(self):
        t = self.norm.normalize("during the Third Age")
        assert t.era == "Third Age" and t.year_start is None and t.confidence < 0.6

    def test_year_of_the_age(self):
        t = self.norm.normalize("Year 3019 of the Third Age")
        assert t.era == "Third Age" and t.year_start == 3019

    def test_unknown_text(self):
        t = self.norm.normalize("long ago in the mists of time")
        assert t.confidence <= 0.2

    def test_normalize_event_time_structured(self):
        t = self.norm.normalize_event_time(raw_text=None, era="TA", year=3019)
        assert t.era == "Third Age" and t.year_start == 3019


def _make_event(id, entity_id, location_id, location_name,
                era="Third Age", year=None, year_end=None,
                confidence=0.9, entity_name=None):
    return SpatiotemporalEvent(
        id=id, entity_id=entity_id,
        entity_name=entity_name or entity_id,
        location_id=location_id, location_name=location_name,
        time=NormalizedTime(era=era, year_start=year, year_end=year_end or year, confidence=confidence),
    )


class TestConflictDetector:
    def test_no_conflicts(self):
        events = [
            _make_event("e1", "frodo", "shire", "The Shire", year=3001),
            _make_event("e2", "frodo", "rivendell", "Rivendell", year=3018),
        ]
        assert len(ConflictDetector().detect_conflicts(events)) == 0

    def test_temporal_overlap_detected(self):
        events = [
            _make_event("e1", "gandalf", "shire", "The Shire", year=3018, year_end=3019),
            _make_event("e2", "gandalf", "isengard", "Isengard", year=3018, year_end=3018),
        ]
        conflicts = ConflictDetector().detect_conflicts(events)
        assert len(conflicts) >= 1
        assert conflicts[0].conflict_type == ConflictType.TEMPORAL_OVERLAP

    def test_no_overlap_different_entities(self):
        events = [
            _make_event("e1", "frodo", "shire", "The Shire", year=3018),
            _make_event("e2", "gandalf", "isengard", "Isengard", year=3018),
        ]
        assert len(ConflictDetector().detect_conflicts(events)) == 0

    def test_travel_infeasibility_with_gap(self):
        locs = {
            "shire": LocationNode(id="shire", name="The Shire", x=0, y=0),
            "mordor": LocationNode(id="mordor", name="Mordor", x=1000, y=0),
        }
        events = [
            _make_event("e1", "frodo", "shire", "The Shire", year=3018),
            _make_event("e2", "frodo", "mordor", "Mordor", year=3019),
        ]
        conflicts = ConflictDetector(locations=locs).detect_conflicts(events)
        travel = [c for c in conflicts if c.conflict_type == ConflictType.TRAVEL_INFEASIBLE]
        assert len(travel) >= 1

    def test_travel_with_edges(self):
        locs = {
            "shire": LocationNode(id="shire", name="The Shire", x=0, y=0),
            "bree": LocationNode(id="bree", name="Bree", x=5, y=0),
        }
        edges = [LocationEdge(source_id="shire", target_id="bree", travel_days=3)]
        d = ConflictDetector(locations=locs, edges=edges)
        assert d.get_travel_days("shire", "bree") == 3
        assert d.get_travel_days("bree", "shire") == 3

    def test_same_location_no_conflict(self):
        events = [
            _make_event("e1", "frodo", "shire", "The Shire", year=3018),
            _make_event("e2", "frodo", "shire", "The Shire", year=3018),
        ]
        assert len(ConflictDetector().detect_conflicts(events)) == 0


class TestReconciliationReport:
    def test_no_conflicts_report(self):
        r = ReconciliationReport(conflicts=[], events=[])
        assert "No timeline conflicts" in r.summary_line()
        assert "No inconsistencies" in r.to_text()

    def test_report_with_conflicts(self):
        conflicts = [
            TimelineConflict(id="c1", conflict_type=ConflictType.TEMPORAL_OVERLAP,
                             severity="error", description="test overlap", confidence=0.8),
            TimelineConflict(id="c2", conflict_type=ConflictType.TRAVEL_INFEASIBLE,
                             severity="warning", description="test travel", confidence=0.5),
        ]
        r = ReconciliationReport(conflicts=conflicts, events=[])
        assert r.error_count == 1 and r.warning_count == 1
        text = r.to_text()
        assert "TEMPORAL OVERLAP" in text and "TRAVEL INFEASIBLE" in text

    def test_to_dict(self):
        conflicts = [TimelineConflict(id="c1", conflict_type=ConflictType.TEMPORAL_OVERLAP,
                                      severity="error", description="t", confidence=0.8)]
        d = ReconciliationReport(conflicts=conflicts, events=[]).to_dict()
        assert d["total_conflicts"] == 1 and d["errors"] == 1


class TestCLITimelineReconcile:
    def test_reconcile_no_conflicts(self, tmp_path):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main
        events = [
            {"id": "e1", "entity_id": "frodo", "entity_name": "Frodo",
             "location_id": "shire", "location_name": "The Shire",
             "time": {"era": "Third Age", "year_start": 3001, "year_end": 3001, "confidence": 0.9}},
            {"id": "e2", "entity_id": "frodo", "entity_name": "Frodo",
             "location_id": "rivendell", "location_name": "Rivendell",
             "time": {"era": "Third Age", "year_start": 3018, "year_end": 3018, "confidence": 0.9}},
        ]
        f = tmp_path / "events.json"
        f.write_text(json.dumps(events))
        result = CliRunner().invoke(main, ["lore", "timeline-reconcile", str(f)])
        assert result.exit_code == 0

    def test_reconcile_with_conflicts(self, tmp_path):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main
        events = [
            {"id": "e1", "entity_id": "gandalf", "entity_name": "Gandalf",
             "location_id": "shire", "location_name": "The Shire",
             "time": {"era": "Third Age", "year_start": 3018, "year_end": 3019, "confidence": 0.9}},
            {"id": "e2", "entity_id": "gandalf", "entity_name": "Gandalf",
             "location_id": "isengard", "location_name": "Isengard",
             "time": {"era": "Third Age", "year_start": 3018, "year_end": 3018, "confidence": 0.9}},
        ]
        f = tmp_path / "events.json"
        f.write_text(json.dumps(events))
        result = CliRunner().invoke(main, ["lore", "timeline-reconcile", str(f)])
        assert result.exit_code == 0
        assert "conflict" in result.output.lower() or "error" in result.output.lower()

    def test_reconcile_json_output(self, tmp_path):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main
        events = [{"id": "e1", "entity_id": "frodo", "entity_name": "Frodo",
                    "location_id": "shire", "location_name": "The Shire",
                    "time": {"era": "Third Age", "year_start": 3001, "year_end": 3001, "confidence": 0.9}}]
        ef = tmp_path / "events.json"
        ef.write_text(json.dumps(events))
        out = tmp_path / "report.json"
        result = CliRunner().invoke(main, ["lore", "timeline-reconcile", str(ef), "--format", "json", "-o", str(out)])
        assert result.exit_code == 0 and out.exists()
        assert "total_conflicts" in json.loads(out.read_text())

    def test_reconcile_with_locations(self, tmp_path):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main
        events = [
            {"id": "e1", "entity_id": "frodo", "entity_name": "Frodo",
             "location_id": "shire", "location_name": "The Shire",
             "time": {"era": "Third Age", "year_start": 3018, "year_end": 3018, "confidence": 0.9}},
            {"id": "e2", "entity_id": "frodo", "entity_name": "Frodo",
             "location_id": "mordor", "location_name": "Mordor",
             "time": {"era": "Third Age", "year_start": 3019, "year_end": 3019, "confidence": 0.9}},
        ]
        locs = {"locations": [{"id": "shire", "name": "The Shire", "x": 0, "y": 0},
                               {"id": "mordor", "name": "Mordor", "x": 1000, "y": 0}],
                "edges": [{"source_id": "shire", "target_id": "mordor", "travel_days": 180}]}
        ef = tmp_path / "events.json"
        ef.write_text(json.dumps(events))
        lf = tmp_path / "locations.json"
        lf.write_text(json.dumps(locs))
        result = CliRunner().invoke(main, ["lore", "timeline-reconcile", str(ef), "-l", str(lf)])
        assert result.exit_code == 0
