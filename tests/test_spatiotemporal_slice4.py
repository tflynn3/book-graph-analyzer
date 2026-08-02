"""Tests for spatiotemporal engine slice 4 (#48):
- CausalLink extraction integration (heuristic)
- CausalLink persistence (GraphWriter contract tests)
- Cross-book reconciliation (CorpusReconciler)
- CLI integration for corpus timeline-reconcile and --causal-links
"""

import json

from book_graph_analyzer.spatiotemporal.models import (
    CausalLink, ConflictType, NormalizedTime, SpatiotemporalEvent,
)
from book_graph_analyzer.spatiotemporal.causal_extraction import (
    extract_causal_links_heuristic,
)
from book_graph_analyzer.spatiotemporal.corpus_reconciler import (
    CorpusReconciler,
)
from book_graph_analyzer.spatiotemporal.conflict_detector import ConflictDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ev(id, entity_id="char1", era="Third Age", year=None, description="", **kw):
    return SpatiotemporalEvent(
        id=id, entity_id=entity_id, entity_name=kw.get("entity_name", entity_id),
        location_id=kw.get("location_id"), location_name=kw.get("location_name"),
        time=NormalizedTime(era=era, year_start=year, year_end=year, confidence=0.9),
        description=description,
        source_book=kw.get("source_book"),
    )


# ===========================================================================
# 1. CausalLink Extraction (Heuristic)
# ===========================================================================

class TestCausalLinkExtraction:
    def test_no_events_returns_empty(self):
        assert extract_causal_links_heuristic([]) == []

    def test_single_event_returns_empty(self):
        events = [_ev("e1", description="Bilbo found the Ring")]
        assert extract_causal_links_heuristic(events) == []

    def test_causal_language_triggers_link(self):
        events = [
            _ev("e1", year=2941, description="Bilbo found the Ring in the cave"),
            _ev("e2", year=2942, description="Because of the Ring, Bilbo became invisible"),
        ]
        links = extract_causal_links_heuristic(events, min_confidence=0.3)
        assert len(links) >= 1
        assert links[0].cause_event_id == "e1"
        assert links[0].effect_event_id == "e2"
        assert links[0].confidence > 0.3

    def test_forward_causal_language(self):
        events = [
            _ev("e1", year=3018, description="Sauron's forces triggered a great war"),
            _ev("e2", year=3019, description="The war ravaged the land"),
        ]
        links = extract_causal_links_heuristic(events, min_confidence=0.3)
        assert len(links) >= 1

    def test_no_causal_language_no_links(self):
        events = [
            _ev("e1", year=3018, description="Bilbo had a birthday party"),
            _ev("e2", year=3019, description="Frodo went to the market"),
        ]
        links = extract_causal_links_heuristic(events, min_confidence=0.3)
        assert len(links) == 0

    def test_cross_entity_causal_link(self):
        events = [
            _ev("e1", entity_id="gandalf", entity_name="Gandalf", year=3018,
                description="Gandalf led to the discovery and warned Frodo"),
            _ev("e2", entity_id="frodo", entity_name="Frodo", year=3018,
                description="Frodo left the Shire because of Gandalf"),
        ]
        links = extract_causal_links_heuristic(events, min_confidence=0.2)
        # Should find a cross-entity link since e2 mentions Gandalf + causal language
        assert len(links) >= 1

    def test_high_min_confidence_filters(self):
        events = [
            _ev("e1", year=1, description="X therefore Y"),
            _ev("e2", year=2, description="Y happened"),
        ]
        # "therefore" has base_conf 0.65, * 0.9 * 0.9 ~ 0.53
        links_low = extract_causal_links_heuristic(events, min_confidence=0.3)
        links_high = extract_causal_links_heuristic(events, min_confidence=0.9)
        assert len(links_low) >= len(links_high)

    def test_links_sorted_by_confidence_desc(self):
        events = [
            _ev("e1", year=1, description="X caused Y"),  # 0.8 base
            _ev("e2", year=2, description="Y resulted in Z"),  # 0.8 base
            _ev("e3", year=3, description="Z therefore W"),  # 0.65 base
        ]
        links = extract_causal_links_heuristic(events, min_confidence=0.2)
        if len(links) >= 2:
            assert links[0].confidence >= links[1].confidence


# ===========================================================================
# 2. CorpusReconciler
# ===========================================================================

class TestCorpusReconciler:
    def test_empty_reconciler(self):
        r = CorpusReconciler()
        result = r.reconcile()
        assert result.total_events == 0
        assert result.total_conflicts == 0

    def test_single_book_no_conflicts(self):
        r = CorpusReconciler(extract_causal=False)
        events = [
            _ev("e1", year=2941, location_id="bag-end", description="Birthday party"),
            _ev("e2", year=2942, location_id="rivendell", description="Arrived at Rivendell"),
        ]
        r.add_book("hobbit", "The Hobbit", events)
        result = r.reconcile()
        assert result.total_events == 2
        assert len(result.books) == 1

    def test_cross_book_overlap_detected(self):
        r = CorpusReconciler(extract_causal=False)
        # Gandalf in two places at once across books
        r.add_book("book_a", "Book A", [
            _ev("a1", entity_id="gandalf", year=3018, location_id="shire",
                location_name="Shire", description="Gandalf in the Shire"),
        ])
        r.add_book("book_b", "Book B", [
            _ev("b1", entity_id="gandalf", year=3018, location_id="minas_tirith",
                location_name="Minas Tirith", description="Gandalf in Minas Tirith"),
        ])
        result = r.reconcile()
        assert len(result.cross_book_conflicts) > 0
        assert len(result.contradiction_clusters) > 0

    def test_contradiction_cluster_has_recommendation(self):
        r = CorpusReconciler(extract_causal=False)
        r.add_book("book_a", "The Hobbit", [
            _ev("a1", entity_id="gandalf", year=3018, location_id="shire",
                location_name="Shire", description="Gandalf in the Shire", source_book="the_hobbit"),
        ])
        r.add_book("book_b", "The Silmarillion", [
            _ev("b1", entity_id="gandalf", year=3018, location_id="minas_tirith",
                location_name="Minas Tirith", description="Gandalf in Minas Tirith", source_book="the_silmarillion"),
        ])
        result = r.reconcile()
        assert result.contradiction_clusters
        cluster = result.contradiction_clusters[0]
        assert cluster.recommended_resolution in {"use_later_text", "use_most_cited", "flag_for_human"}
        assert cluster.avg_authority_weight > 0

    def test_causal_extraction_integrated(self):
        r = CorpusReconciler(extract_causal=True)
        events = [
            _ev("e1", year=3018, description="The Ring caused corruption"),
            _ev("e2", year=3019, description="As a result of corruption, war began"),
        ]
        r.add_book("lotr", "LOTR", events)
        result = r.reconcile()
        assert len(result.all_causal_links) >= 1

    def test_result_to_dict_and_summary(self):
        r = CorpusReconciler(extract_causal=False)
        r.add_book("b1", "Book 1", [_ev("e1", year=1, description="Ev1")])
        r.add_book("b2", "Book 2", [_ev("e2", year=2, description="Ev2")])
        result = r.reconcile()

        d = result.to_dict()
        assert d["books_analyzed"] == 2
        assert "per_book" in d

        text = result.summary_text()
        assert "CORPUS TIMELINE RECONCILIATION" in text
        assert "Book 1" in text

    def test_add_book_from_json(self, tmp_path):
        events = [
            {"id": "e1", "entity_id": "gandalf", "entity_name": "Gandalf",
             "time": {"era": "Third Age", "year_start": 3018, "year_end": 3018, "confidence": 0.9},
             "description": "Gandalf arrived"},
        ]
        fp = tmp_path / "events.json"
        fp.write_text(json.dumps({"events": events}), encoding="utf-8")

        r = CorpusReconciler(extract_causal=False)
        count = r.add_book_from_json(str(fp), "test", "Test Book")
        assert count == 1
        assert len(r.books) == 1
        assert r.books[0].events[0].structural_stratum is None


# ===========================================================================
# 3. CausalLink Integration with ConflictDetector
# ===========================================================================

class TestCausalLinkIntegration:
    """Test that extracted causal links feed properly into paradox detection."""

    def test_extracted_links_fed_to_detector_no_paradox(self):
        """Heuristic links follow temporal order, so no paradox expected."""
        events = [
            _ev("e1", year=3018, description="War caused destruction of the land"),
            _ev("e2", year=3019, description="As a result of the war, rebuilding began"),
        ]
        links = extract_causal_links_heuristic(events, min_confidence=0.2)
        assert len(links) >= 1
        # Links follow temporal order → no paradox
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(
            events, check_causal_paradoxes=True, causal_links=links,
        )
        paradoxes = [c for c in conflicts if c.conflict_type == ConflictType.CAUSAL_PARADOX]
        assert len(paradoxes) == 0

    def test_manual_paradox_link_detected(self):
        """Manually constructed backwards causal link should trigger paradox."""
        events = [
            _ev("e1", year=3019, description="Battle"),
            _ev("e2", year=3018, description="Peace declared"),
        ]
        # Manually claim e1 (3019) causes e2 (3018) — a paradox
        links = [CausalLink(cause_event_id="e1", effect_event_id="e2",
                            description="battle -> peace", confidence=0.9)]
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(
            events, check_causal_paradoxes=True, causal_links=links,
        )
        paradoxes = [c for c in conflicts if c.conflict_type == ConflictType.CAUSAL_PARADOX]
        assert len(paradoxes) >= 1


# ===========================================================================
# 4. CLI Integration (Click runner)
# ===========================================================================

class TestCLIIntegration:
    def test_lore_timeline_bridge_with_causal_links(self, tmp_path):
        """Test that --causal-links flag works in timeline-bridge command."""
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        events = {
            "events": {
                "e1": {"id": "e1", "description": "War caused destruction",
                       "agent": "Sauron", "action": "caused", "patient": "destruction",
                       "era": "third_age", "year": 3019, "confidence": 0.9},
                "e2": {"id": "e2", "description": "As a result of destruction, rebuilding began",
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
            "--causal-links", "--format", "json",
        ])
        assert result.exit_code == 0

    def test_corpus_timeline_reconcile_no_books(self):
        """Test graceful handling when corpus has no event files."""
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, [
            "corpus", "timeline-reconcile", "nonexistent_corpus",
        ])
        # Should fail gracefully (no books or no event files)
        assert result.exit_code == 0 or "No books" in result.output or "No event files" in result.output

    def test_corpus_timeline_divergence_hotspots(self, monkeypatch):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        class _W:
            def query_divergence_hotspots(self, min_sources=2, limit=25):
                return [{"entity_id": "gandalf", "conflict_type": "temporal_overlap", "conflict_count": 2,
                         "source_count": 2, "avg_authority": 0.9}]

            def close(self):
                pass

        monkeypatch.setattr("book_graph_analyzer.graph.connection.check_neo4j_connection", lambda: True)
        monkeypatch.setattr("book_graph_analyzer.graph.writer.GraphWriter", lambda: _W())

        result = CliRunner().invoke(main, ["corpus", "timeline-divergence"])
        assert result.exit_code == 0
        assert "divergence hotspots" in result.output.lower()
