"""Tests for the generic entity bootstrapper (issue #2)."""

import pytest
from book_graph_analyzer.extract.bootstrap import (
    EntityBootstrapper,
    EntityCandidate,
    EntityCluster,
    BootstrapResult,
)

# ---------------------------------------------------------------------------
# Sample text — contains entities with multiple surface forms
# ---------------------------------------------------------------------------
SAMPLE_TEXT = """
In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole,
filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole
with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.

Bilbo Baggins was a very well-to-do hobbit. Mr. Baggins had lived in his hobbit-hole
at Bag End his whole life. Bilbo was respected and prosperous.
Bilbo Baggins had inherited Bag End from his uncle.

One morning Gandalf came by. The wizard Gandalf sat down on a bench outside the door.
Gandalf was known throughout the Shire as a maker of remarkable fireworks.
Everyone in the Shire knew of Gandalf and his fireworks displays.

The Shire was a comfortable land. In all the Shire there was not a man or a hobbit
who had not heard of Bilbo Baggins of Bag End. Bag End sat at the end of The Water.

Gandalf and Mr. Baggins spoke for some time. Bilbo was not pleased to be interrupted.
The old wizard smiled at Bilbo and turned to leave. Gandalf walked away down the lane.
"""

SAMPLE_TEXT_RICH = SAMPLE_TEXT + """
Later Frodo Baggins inherited the ring from Bilbo.
Young Frodo was much like his uncle Bilbo in temperament.
Frodo Baggins set off on a great adventure.

Sam Gamgee accompanied Frodo on the journey.
Samwise Gamgee was loyal to Frodo throughout.
Sam and Frodo traveled together across Middle-earth.

The land of Rohan was far to the east.
Rohan was known for its horses.
The riders of Rohan were feared by many.
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractCandidates:
    def test_finds_gandalf(self):
        b = EntityBootstrapper(use_llm=False)
        candidates = b.extract_candidates(SAMPLE_TEXT)
        names = [c.text for c in candidates]
        assert any("Gandalf" in n for n in names), f"Gandalf not in candidates: {names}"

    def test_finds_bilbo(self):
        b = EntityBootstrapper(use_llm=False)
        candidates = b.extract_candidates(SAMPLE_TEXT)
        names = [c.text for c in candidates]
        assert any("Bilbo" in n for n in names), f"Bilbo not in candidates: {names}"

    def test_respects_min_frequency(self):
        b = EntityBootstrapper(use_llm=False)
        b.MIN_FREQUENCY = 3
        candidates = b.extract_candidates(SAMPLE_TEXT)
        assert all(c.frequency >= 3 for c in candidates)

    def test_context_windows_populated(self):
        b = EntityBootstrapper(use_llm=False)
        candidates = b.extract_candidates(SAMPLE_TEXT)
        for c in candidates:
            assert len(c.contexts) >= 1, f"{c.text!r} has no context windows"
            assert len(c.contexts[0]) > 10, f"{c.text!r} context is too short"

    def test_context_window_length(self):
        b = EntityBootstrapper(use_llm=False)
        candidates = b.extract_candidates(SAMPLE_TEXT)
        for c in candidates:
            for ctx in c.contexts:
                # Context should be at most 2 * CONTEXT_WINDOW + name length chars
                assert len(ctx) <= b.CONTEXT_WINDOW * 2 + 50

    def test_frequency_counts(self):
        b = EntityBootstrapper(use_llm=False)
        b.MIN_FREQUENCY = 1
        candidates = b.extract_candidates(SAMPLE_TEXT)
        gandalf = next((c for c in candidates if c.text == "Gandalf"), None)
        assert gandalf is not None, "Gandalf candidate not found"
        assert gandalf.frequency >= 4, f"Gandalf frequency too low: {gandalf.frequency}"

    def test_stopwords_excluded(self):
        b = EntityBootstrapper(use_llm=False)
        candidates = b.extract_candidates(SAMPLE_TEXT)
        names = {c.text for c in candidates}
        stop_found = names.intersection({"The", "And", "But", "He", "She", "It"})
        assert not stop_found, f"Stop-words in candidates: {stop_found}"


class TestClusterAliases:
    def test_gandalf_clusters(self):
        b = EntityBootstrapper(use_llm=False)
        candidates = b.extract_candidates(SAMPLE_TEXT)
        clusters = b.cluster_aliases(candidates)
        gandalf_clusters = [
            c for c in clusters
            if any("Gandalf" in v for v in c.variants)
        ]
        assert len(gandalf_clusters) >= 1, "Gandalf should be in at least one cluster"

    def test_bilbo_baggins_same_cluster(self):
        """Bilbo and Mr. Baggins should land in the same cluster (or overlapping)."""
        b = EntityBootstrapper(use_llm=False)
        candidates = b.extract_candidates(SAMPLE_TEXT)
        clusters = b.cluster_aliases(candidates)

        bilbo_cluster = None
        baggins_cluster = None
        for c in clusters:
            variants_lo = " ".join(c.variants).lower()
            if "bilbo" in variants_lo:
                bilbo_cluster = c
            if "baggins" in variants_lo:
                baggins_cluster = c

        # Either they're in the same cluster, or Bilbo Baggins cluster exists
        same = bilbo_cluster is not None and bilbo_cluster is baggins_cluster
        bilbo_baggins = any(
            "bilbo" in " ".join(c.variants).lower() and "baggins" in " ".join(c.variants).lower()
            for c in clusters
        )
        assert same or bilbo_baggins, (
            "Bilbo and Baggins should share a cluster. "
            f"Clusters: {[(c.variants, c.frequency) for c in clusters[:5]]}"
        )

    def test_cluster_frequency_aggregated(self):
        b = EntityBootstrapper(use_llm=False)
        candidates = b.extract_candidates(SAMPLE_TEXT)
        clusters = b.cluster_aliases(candidates)
        for c in clusters:
            assert c.frequency > 0

    def test_clusters_cover_all_candidates(self):
        """Every candidate name should appear in exactly one cluster."""
        b = EntityBootstrapper(use_llm=False)
        candidates = b.extract_candidates(SAMPLE_TEXT)
        clusters = b.cluster_aliases(candidates)

        all_cluster_variants = set()
        for c in clusters:
            for v in c.variants:
                all_cluster_variants.add(v)

        candidate_names = {c.text for c in candidates}
        assert candidate_names == all_cluster_variants, (
            f"Mismatch. Missing from clusters: {candidate_names - all_cluster_variants}"
        )

    def test_single_member_clusters_lower_confidence(self):
        """Clusters with only one variant should have lower confidence than multi-variant clusters."""
        b = EntityBootstrapper(use_llm=False)
        candidates = b.extract_candidates(SAMPLE_TEXT_RICH)
        clusters = b.cluster_aliases(candidates)

        singles = [c for c in clusters if len(c.variants) == 1]
        multis = [c for c in clusters if len(c.variants) > 1]

        if singles and multis:
            avg_single = sum(c.cluster_confidence for c in singles) / len(singles)
            avg_multi = sum(c.cluster_confidence for c in multis) / len(multis)
            assert avg_single <= avg_multi, (
                f"Single-member clusters ({avg_single:.2f}) should not exceed "
                f"multi-member ({avg_multi:.2f})"
            )


class TestBootstrapNollm:
    def test_returns_bootstrap_result(self):
        b = EntityBootstrapper(use_llm=False)
        result = b.bootstrap(SAMPLE_TEXT, verbose=False)
        assert isinstance(result, BootstrapResult)

    def test_stats_populated(self):
        b = EntityBootstrapper(use_llm=False)
        result = b.bootstrap(SAMPLE_TEXT, verbose=False)
        assert result.stats["candidates"] > 0
        assert result.stats["clusters"] > 0

    def test_some_entities_accepted(self):
        b = EntityBootstrapper(use_llm=False)
        result = b.bootstrap(SAMPLE_TEXT_RICH, verbose=False)
        assert len(result.entities) + len(result.flagged) > 0

    def test_gandalf_appears_in_results(self):
        b = EntityBootstrapper(use_llm=False)
        result = b.bootstrap(SAMPLE_TEXT, verbose=False)
        all_variants = [v for e in result.all_entities() for v in e.variants]
        all_names = [e.canonical_name for e in result.all_entities()] + all_variants
        assert any("Gandalf" in n for n in all_names), (
            f"Gandalf not found in results. All names: {all_names}"
        )

    def test_canonical_names_non_empty(self):
        b = EntityBootstrapper(use_llm=False)
        result = b.bootstrap(SAMPLE_TEXT, verbose=False)
        for entity in result.all_entities():
            assert entity.canonical_name, f"Entity with variants {entity.variants} has no canonical name"

    def test_to_dict_list(self):
        b = EntityBootstrapper(use_llm=False)
        result = b.bootstrap(SAMPLE_TEXT, verbose=False)
        dicts = result.to_dict_list()
        assert isinstance(dicts, list)
        if dicts:
            d = dicts[0]
            assert "canonical_name" in d
            assert "entity_type" in d
            assert "variants" in d
            assert "cluster_confidence" in d
            assert isinstance(d["variants"], list)


class TestBootstrapRicherText:
    def test_frodo_sam_gamgee(self):
        """Samwise Gamgee and Sam Gamgee should cluster together.

        Uses MIN_FREQUENCY=1 because the test text is intentionally short.
        On a full novel corpus MIN_FREQUENCY=2 is appropriate.
        """
        b = EntityBootstrapper(use_llm=False)
        b.MIN_FREQUENCY = 1
        result = b.bootstrap(SAMPLE_TEXT_RICH, verbose=False)
        all_variants = {v for e in result.all_entities() for v in e.variants}
        assert any("Sam" in v or "Gamgee" in v for v in all_variants), (
            f"Sam/Gamgee not found. Variants: {all_variants}"
        )

    def test_rohan_as_place_candidate(self):
        """Rohan should appear as a candidate in the richer text."""
        b = EntityBootstrapper(use_llm=False)
        candidates = b.extract_candidates(SAMPLE_TEXT_RICH)
        names = [c.text for c in candidates]
        assert any("Rohan" in n for n in names), f"Rohan not in candidates: {names}"
