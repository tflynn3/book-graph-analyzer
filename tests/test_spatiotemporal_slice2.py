"""Tests for spatiotemporal engine slice 2 (#48):
- Extraction bridge (Event -> SpatiotemporalEvent)
- Era mismatch detection
- CLI timeline-bridge command
- Report confidence output
"""

import json
import pytest

from book_graph_analyzer.lore.events import Event
from book_graph_analyzer.lore.temporal import Era
from book_graph_analyzer.spatiotemporal.extraction_bridge import (
    ExtractionBridge, BridgeReport, NormalizationResult,
)
from book_graph_analyzer.spatiotemporal.models import (
    ConflictType, NormalizedTime, SpatiotemporalEvent, TimelineConflict,
)
from book_graph_analyzer.spatiotemporal.conflict_detector import ConflictDetector
from book_graph_analyzer.spatiotemporal.report import ReconciliationReport


# ---------------------------------------------------------------------------
# ExtractionBridge tests
# ---------------------------------------------------------------------------

class TestExtractionBridge:
    def setup_method(self):
        self.bridge = ExtractionBridge()

    def _make_event(self, id="ev1", agent="Bilbo", description="Found the Ring",
                    era=None, year=None, year_text=None, confidence=1.0):
        return Event(
            id=id, description=description, agent=agent,
            era=era, year=year, year_text=year_text,
            confidence=confidence,
        )

    def test_bridge_event_with_year_text(self):
        ev = self._make_event(year_text="Third Age 2941", confidence=0.9)
        result = self.bridge.bridge_event(ev)
        assert result.event.time.era == "Third Age"
        assert result.event.time.year_start == 2941
        assert result.normalization_confidence >= 0.8
        assert result.extraction_confidence == 0.9

    def test_bridge_event_with_era_enum(self):
        ev = self._make_event(era=Era.THIRD_AGE, year=3019, confidence=0.8)
        result = self.bridge.bridge_event(ev)
        assert result.event.time.era == "Third Age"
        assert result.event.time.year_start == 3019

    def test_bridge_event_low_confidence(self):
        ev = self._make_event(confidence=0.3)
        result = self.bridge.bridge_event(ev)
        assert result.normalization_confidence <= 0.2  # no time info -> low

    def test_confidence_delta(self):
        ev = self._make_event(year_text="TA 3019", confidence=0.5)
        result = self.bridge.bridge_event(ev)
        # Normalization of "TA 3019" gives ~0.9, extraction was 0.5
        assert result.confidence_delta < 0  # normalization boosted
        assert result.confidence_category == "normalization_boosted"

    def test_confidence_aligned(self):
        ev = self._make_event(year_text="TA 3019", confidence=0.9)
        result = self.bridge.bridge_event(ev)
        assert result.confidence_category == "aligned"

    def test_era_changed_flag(self):
        # Event says "before the Third Age" -> normalizer might map to Second Age
        ev = self._make_event(year_text="before the Third Age", confidence=0.6)
        result = self.bridge.bridge_event(ev)
        # The normalizer's "before" logic maps to previous era
        assert result.normalized_era is not None

    def test_bridge_events_batch(self):
        events = [
            self._make_event(id="e1", year_text="TA 2941"),
            self._make_event(id="e2", year_text="SA 3441"),
            self._make_event(id="e3"),
        ]
        report = self.bridge.bridge_events(events, source_book="The Hobbit")
        assert report.total == 3
        assert report.events[0].source_book == "The Hobbit"

    def test_bridge_report_summary(self):
        events = [
            self._make_event(id="e1", year_text="TA 2941", confidence=0.9),
            self._make_event(id="e2", confidence=0.9),  # no time info
        ]
        report = self.bridge.bridge_events(events)
        text = report.summary_text()
        assert "Events bridged: 2" in text

    def test_bridge_report_to_dict(self):
        events = [self._make_event(year_text="TA 3019")]
        report = self.bridge.bridge_events(events)
        d = report.to_dict()
        assert "total" in d and "results" in d
        assert len(d["results"]) == 1


# ---------------------------------------------------------------------------
# Era mismatch detection tests
# ---------------------------------------------------------------------------

def _make_st_event(id, entity_id, era, year=None, description="", confidence=0.9,
                   entity_name=None, location_id=None, location_name=None):
    return SpatiotemporalEvent(
        id=id, entity_id=entity_id,
        entity_name=entity_name or entity_id,
        location_id=location_id, location_name=location_name,
        time=NormalizedTime(era=era, year_start=year, year_end=year, confidence=confidence),
        description=description,
    )


class TestEraMismatchDetection:
    def test_no_mismatch_same_era(self):
        events = [
            _make_st_event("e1", "gandalf", "Third Age", 3018),
            _make_st_event("e2", "gandalf", "Third Age", 3019),
        ]
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(events)
        era_mismatches = [c for c in conflicts if c.conflict_type == ConflictType.ERA_MISMATCH]
        assert len(era_mismatches) == 0

    def test_mismatch_non_adjacent_eras(self):
        events = [
            _make_st_event("e1", "gandalf", "Third Age", 3018),
            _make_st_event("e2", "gandalf", "Third Age", 3019),
            _make_st_event("e3", "gandalf", "First Age", 500, description="Gandalf in First Age"),
        ]
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(events)
        era_mismatches = [c for c in conflicts if c.conflict_type == ConflictType.ERA_MISMATCH]
        assert len(era_mismatches) >= 1
        assert "First Age" in era_mismatches[0].description

    def test_no_mismatch_adjacent_eras(self):
        # Second Age and Third Age are adjacent (gap=1), should not trigger
        events = [
            _make_st_event("e1", "elrond", "Second Age", 3441),
            _make_st_event("e2", "elrond", "Third Age", 3019),
        ]
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(events)
        era_mismatches = [c for c in conflicts if c.conflict_type == ConflictType.ERA_MISMATCH]
        assert len(era_mismatches) == 0

    def test_mismatch_different_entities_independent(self):
        events = [
            _make_st_event("e1", "gandalf", "Third Age", 3018),
            _make_st_event("e2", "gandalf", "Third Age", 3019),
            _make_st_event("e3", "feanor", "First Age", 500),
        ]
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(events)
        era_mismatches = [c for c in conflicts if c.conflict_type == ConflictType.ERA_MISMATCH]
        # Feanor only has First Age events, no mismatch within his events
        assert len(era_mismatches) == 0

    def test_mismatch_severity_based_on_confidence(self):
        events = [
            _make_st_event("e1", "gandalf", "Third Age", 3018, confidence=0.9),
            _make_st_event("e2", "gandalf", "Third Age", 3019, confidence=0.9),
            _make_st_event("e3", "gandalf", "First Age", 500, confidence=0.3),
        ]
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(events)
        era_mismatches = [c for c in conflicts if c.conflict_type == ConflictType.ERA_MISMATCH]
        assert len(era_mismatches) >= 1
        # Low confidence * 0.7 = 0.21, should be "warning"
        assert era_mismatches[0].severity == "warning"

    def test_disable_era_mismatch_check(self):
        events = [
            _make_st_event("e1", "gandalf", "Third Age", 3018),
            _make_st_event("e2", "gandalf", "First Age", 500),
        ]
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(events, check_era_mismatches=False)
        era_mismatches = [c for c in conflicts if c.conflict_type == ConflictType.ERA_MISMATCH]
        assert len(era_mismatches) == 0


# ---------------------------------------------------------------------------
# Report with bridge data tests
# ---------------------------------------------------------------------------

class TestReportWithBridge:
    def test_report_includes_bridge_summary(self):
        bridge = ExtractionBridge()
        events = [
            Event(id="e1", description="test", year_text="TA 3019", confidence=0.9),
        ]
        br = bridge.bridge_events(events)
        report = ReconciliationReport(conflicts=[], events=br.events, bridge_report=br)
        text = report.to_text()
        assert "EXTRACTION-VS-NORMALIZED CONFIDENCE" in text
        assert "Events bridged:" in text

    def test_report_dict_includes_bridge(self):
        bridge = ExtractionBridge()
        events = [
            Event(id="e1", description="test", year_text="SA 1600", confidence=0.7),
        ]
        br = bridge.bridge_events(events)
        report = ReconciliationReport(conflicts=[], events=br.events, bridge_report=br)
        d = report.to_dict()
        assert "bridge_report" in d
        assert d["bridge_report"]["total"] == 1

    def test_report_era_mismatch_count(self):
        conflicts = [
            TimelineConflict(id="c1", conflict_type=ConflictType.ERA_MISMATCH,
                             severity="error", description="test", confidence=0.8),
            TimelineConflict(id="c2", conflict_type=ConflictType.TEMPORAL_OVERLAP,
                             severity="warning", description="test2", confidence=0.5),
        ]
        report = ReconciliationReport(conflicts=conflicts)
        assert report.era_mismatch_count == 1
        text = report.to_text()
        assert "Era mismatches: 1" in text
        d = report.to_dict()
        assert d["era_mismatches"] == 1


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestCLITimelineBridge:
    def test_bridge_basic(self, tmp_path):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        events = {
            "events": {
                "e1": {
                    "id": "e1", "description": "Bilbo finds ring",
                    "agent": "Bilbo", "action": "found", "patient": "Ring",
                    "era": "third_age", "year": 2941, "confidence": 0.9,
                },
            },
            "relations": [],
        }
        f = tmp_path / "events.json"
        f.write_text(json.dumps(events))
        result = CliRunner().invoke(main, ["lore", "timeline-bridge", str(f)])
        assert result.exit_code == 0

    def test_bridge_json_output(self, tmp_path):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        events = {
            "events": {
                "e1": {
                    "id": "e1", "description": "Battle of Dagorlad",
                    "agent": "Last Alliance", "era": "second_age", "year": 3434,
                    "confidence": 0.95,
                },
                "e2": {
                    "id": "e2", "description": "Council of Elrond",
                    "agent": "Elrond", "era": "third_age", "year": 3018,
                    "confidence": 0.9,
                },
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
        assert out.exists()
        report = json.loads(out.read_text())
        assert "bridge_report" in report
        assert report["bridge_report"]["total"] == 2

    def test_bridge_with_era_mismatch(self, tmp_path):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        # Gandalf in Third Age mostly, but one event in First Age -> mismatch
        events = {
            "events": {
                "e1": {
                    "id": "e1", "description": "Gandalf arrives in Middle-earth",
                    "agent": "Gandalf", "era": "third_age", "year": 1000,
                    "confidence": 0.9,
                },
                "e2": {
                    "id": "e2", "description": "Gandalf at Council of Elrond",
                    "agent": "Gandalf", "era": "third_age", "year": 3018,
                    "confidence": 0.9,
                },
                "e3": {
                    "id": "e3", "description": "Gandalf in First Age",
                    "agent": "Gandalf", "era": "first_age", "year": 100,
                    "confidence": 0.8,
                },
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
        assert report["era_mismatches"] >= 1


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------

class TestEndToEndBridgeAndReconcile:
    def test_extract_bridge_detect(self):
        """Full path: create Events -> bridge -> detect conflicts."""
        events = [
            Event(id="e1", description="Gandalf in Shire", agent="Gandalf",
                  era=Era.THIRD_AGE, year=3018, confidence=0.9),
            Event(id="e2", description="Gandalf at Isengard", agent="Gandalf",
                  era=Era.THIRD_AGE, year=3018, confidence=0.85),
            Event(id="e3", description="Gandalf in First Age somehow", agent="Gandalf",
                  era=Era.FIRST_AGE, year=400, confidence=0.4),
        ]
        bridge = ExtractionBridge()
        br = bridge.bridge_events(events)

        assert br.total == 3
        assert br.era_changed_count >= 0

        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(br.events)

        report = ReconciliationReport(conflicts=conflicts, events=br.events, bridge_report=br)

        # Should have era mismatch
        assert report.era_mismatch_count >= 1

        # Report text should contain bridge info
        text = report.to_text()
        assert "EXTRACTION-VS-NORMALIZED CONFIDENCE" in text
        assert "ERA MISMATCH" in text

        # Dict output
        d = report.to_dict()
        assert d["era_mismatches"] >= 1
        assert d["bridge_report"]["total"] == 3
