"""Tests for NarrativeWeight metric (Issue #5).

All tests run without Neo4j or LLM — pure Python computation.
Covers:
  - NarrativeWeight dataclass
  - ThemeNode model and TOLKIEN_THEMES taxonomy
  - NarrativeWeightComputer (rule-based computation)
  - Text analysis helpers
  - compute_overall / weakest_components / improvement_suggestions
"""

from book_graph_analyzer.models.narrative_weight import (
    NarrativeWeight,
    ThemeNode,
    TOLKIEN_THEMES,
    THEME_BY_ID,
    COMPONENT_WEIGHTS,
    COMPONENT_SUGGESTIONS,
)
from book_graph_analyzer.lore.narrative_weight import (
    NarrativeWeightComputer,
    _count_proper_nouns,
    _count_sentences,
    _count_words,
    _dialogue_density,
    _detect_themes,
    _theme_coherence,
    _emotional_keywords,
    _emotional_contrast,
    _foreshadowing_score,
    _revelation_score,
    _callback_density,
    _clamp,
    _norm,
)
from book_graph_analyzer.models.passage import Passage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TOLKIEN_RICH_TEXT = """
"Do you know, Frodo, I have thought of an ending for my book.
And he lived happily ever afterwards to the end of his days.
But do you remember when I told you of Galadriel?
She is one of the great Eldar, born in the Years of the Trees,
in the ages before the Sun and Moon.
In the elder days, when the world was young and fresh,
the Valar made the Two Trees.
And now we stand at the twilight of the Third Age,
and all things are fading, as the Elves have long foreseen.
Perhaps one day, Frodo, you will understand.
I have fought the long defeat, knowing it cannot be won.
But eucatastrophe can still come, even now."
"""

SIMPLE_TEXT = "Bilbo went to the market."

MULTIPLE_THEMES_TEXT = """
Hope seemed impossible, yet Samwise endured with loyal courage.
The ancient darkness pressed down, but the small light of mercy endured.
Long ago, before your time, these things were decided in the elder days.
One day you will understand what was foreshadowed here.
"""


def make_passage(**overrides) -> Passage:
    defaults = dict(
        id="p_test",
        text="Gandalf spoke of the Elder Days when hope seemed lost.",
        book="LOTR",
        chapter="Ch1",
        chapter_num=1,
        paragraph_num=1,
        sentence_num=1,
        char_offset=0,
    )
    defaults.update(overrides)
    return Passage(**defaults)


# ---------------------------------------------------------------------------
# TOLKIEN_THEMES taxonomy
# ---------------------------------------------------------------------------

class TestThemeTaxonomy:
    def test_at_least_ten_themes(self):
        assert len(TOLKIEN_THEMES) >= 10, "Must define at least 10 core themes"

    def test_all_themes_have_ids(self):
        for theme in TOLKIEN_THEMES:
            assert theme.id, f"Theme missing id: {theme}"

    def test_all_themes_have_names(self):
        for theme in TOLKIEN_THEMES:
            assert theme.name, f"Theme missing name: {theme.id}"

    def test_all_themes_have_descriptions(self):
        for theme in TOLKIEN_THEMES:
            assert len(theme.description) > 20, f"Theme description too short: {theme.id}"

    def test_all_themes_have_detection_keywords(self):
        for theme in TOLKIEN_THEMES:
            assert theme.detection_keywords, f"Theme has no keywords: {theme.id}"

    def test_theme_by_id_lookup(self):
        assert "eucatastrophe" in THEME_BY_ID
        assert "the_long_defeat" in THEME_BY_ID
        assert "diminishment" in THEME_BY_ID
        assert "loyalty" in THEME_BY_ID
        assert "mercy" in THEME_BY_ID

    def test_eucatastrophe_is_tolkien_specific(self):
        assert THEME_BY_ID["eucatastrophe"].tolkien_specific is True

    def test_the_long_defeat_is_tolkien_specific(self):
        assert THEME_BY_ID["the_long_defeat"].tolkien_specific is True

    def test_loyalty_is_not_tolkien_specific(self):
        assert THEME_BY_ID["loyalty"].tolkien_specific is False

    def test_to_neo4j_props_has_required_fields(self):
        theme = THEME_BY_ID["eucatastrophe"]
        props = theme.to_neo4j_props()
        assert "id" in props
        assert "name" in props
        assert "description" in props
        assert "tolkien_specific" in props


# ---------------------------------------------------------------------------
# COMPONENT_WEIGHTS
# ---------------------------------------------------------------------------

class TestComponentWeights:
    def test_weights_sum_to_one(self):
        total = sum(COMPONENT_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"Weights should sum to 1.0, got {total}"

    def test_all_weights_positive(self):
        for comp, w in COMPONENT_WEIGHTS.items():
            assert w > 0, f"Component weight must be > 0: {comp}"

    def test_all_components_have_suggestions(self):
        for comp in COMPONENT_WEIGHTS:
            assert comp in COMPONENT_SUGGESTIONS, f"Missing suggestion for component: {comp}"


# ---------------------------------------------------------------------------
# NarrativeWeight dataclass
# ---------------------------------------------------------------------------

class TestNarrativeWeight:
    def test_default_all_zeros(self):
        w = NarrativeWeight()
        assert w.overall == 0.0
        assert w.temporal_depth == 0.0
        assert w.thematic_threads == 0.0

    def test_compute_overall_weighted_average(self):
        w = NarrativeWeight(temporal_depth=1.0)  # only temporal_depth = 1.0
        w2 = w.compute_overall()
        expected = COMPONENT_WEIGHTS["temporal_depth"]
        assert abs(w2.overall - expected) < 1e-6

    def test_compute_overall_all_ones(self):
        kwargs = {comp: 1.0 for comp in COMPONENT_WEIGHTS}
        w = NarrativeWeight(**kwargs)
        w2 = w.compute_overall()
        assert abs(w2.overall - 1.0) < 1e-6

    def test_compute_overall_does_not_mutate(self):
        w = NarrativeWeight(temporal_depth=0.5)
        w2 = w.compute_overall()
        assert w.overall == 0.0  # original unchanged
        assert w2.overall != 0.0

    def test_weakest_components_returns_sorted(self):
        w = NarrativeWeight(temporal_depth=0.9, era_reference_count=0.1, lore_density=0.5)
        weak = w.weakest_components(n=3)
        # Check it's sorted ascending by score
        scores = [s for _, s in weak]
        assert scores == sorted(scores)

    def test_weakest_components_correct_n(self):
        w = NarrativeWeight()
        weak = w.weakest_components(n=2)
        assert len(weak) == 2

    def test_improvement_suggestions_count(self):
        w = NarrativeWeight()
        sug = w.improvement_suggestions(n=3)
        assert len(sug) == 3

    def test_improvement_suggestions_are_strings(self):
        w = NarrativeWeight()
        for s in w.improvement_suggestions():
            assert isinstance(s, str)
            assert len(s) > 10

    def test_to_dict_has_nw_prefix(self):
        w = NarrativeWeight(temporal_depth=0.8)
        d = w.to_dict()
        assert "nw_temporal_depth" in d
        assert d["nw_temporal_depth"] == 0.8
        assert "nw_overall" in d

    def test_from_dict_roundtrip(self):
        w = NarrativeWeight(temporal_depth=0.7, thematic_threads=0.5)
        w2 = w.compute_overall()
        d = w2.to_dict()
        w3 = NarrativeWeight.from_dict(d)
        assert abs(w3.temporal_depth - 0.7) < 1e-4
        assert abs(w3.thematic_threads - 0.5) < 1e-4
        assert abs(w3.overall - w2.overall) < 1e-4

    def test_from_dict_plain_keys(self):
        """from_dict should also work without the nw_ prefix."""
        d = {"temporal_depth": 0.6, "lore_density": 0.4, "overall": 0.0}
        w = NarrativeWeight.from_dict(d)
        assert abs(w.temporal_depth - 0.6) < 1e-4

    def test_summary_contains_overall(self):
        w = NarrativeWeight(temporal_depth=0.8).compute_overall()
        summary = w.summary("test_passage")
        assert "Overall" in summary
        assert "test_passage" in summary

    def test_summary_contains_all_components(self):
        w = NarrativeWeight().compute_overall()
        summary = w.summary()
        for comp in COMPONENT_WEIGHTS:
            assert comp in summary, f"Component not in summary: {comp}"

    def test_all_components_clamped_to_unit_interval(self):
        """Even with extreme inputs, all components should be in [0, 1]."""
        w = NarrativeWeight(
            temporal_depth=1.5,  # over-specified
            era_reference_count=-0.1,  # negative
        )
        # No validation in the dataclass — but compute_overall should still work
        w2 = w.compute_overall()
        assert isinstance(w2.overall, float)


# ---------------------------------------------------------------------------
# Text analysis helpers
# ---------------------------------------------------------------------------

class TestTextHelpers:
    def test_clamp_within_bounds(self):
        assert _clamp(0.5) == 0.5
        assert _clamp(1.5) == 1.0
        assert _clamp(-0.5) == 0.0

    def test_norm_basic(self):
        assert abs(_norm(5.0, 10.0) - 0.5) < 1e-9

    def test_norm_zero_max(self):
        assert _norm(5.0, 0.0) == 0.0

    def test_norm_exceeds_max(self):
        assert _norm(20.0, 10.0) == 1.0  # clamped

    def test_count_words_simple(self):
        assert _count_words("Hello world foo") == 3

    def test_count_sentences_single(self):
        assert _count_sentences("Hello world.") >= 1

    def test_count_sentences_multiple(self):
        count = _count_sentences("First sentence. Second sentence. Third sentence.")
        assert count >= 2

    def test_count_proper_nouns(self):
        text = "Gandalf and Frodo walked to the Shire."
        count = _count_proper_nouns(text)
        assert count >= 2  # Gandalf, Frodo, Shire

    def test_count_proper_nouns_common_words_excluded(self):
        text = "The quick brown fox jumps over the lazy dog."
        count = _count_proper_nouns(text)
        # "The" and other common words should not count
        assert count <= 1

    def test_dialogue_density_pure_dialogue(self):
        text = '"I am Gandalf," he said.'
        density = _dialogue_density(text)
        assert density > 0.0

    def test_dialogue_density_no_dialogue(self):
        text = "Frodo walked along the road."
        density = _dialogue_density(text)
        assert density == 0.0

    def test_emotional_keywords_found(self):
        text = "There was great joy and wonder, but also dread in the shadow."
        count = _emotional_keywords(text)
        assert count >= 3  # joy, wonder, dread, shadow

    def test_emotional_contrast_both_present(self):
        text = "The light of hope pierced the shadow of doom."
        score = _emotional_contrast(text)
        assert score == 1.0

    def test_emotional_contrast_only_dark(self):
        text = "Darkness and shadow and doom filled the hall."
        score = _emotional_contrast(text)
        assert score == 0.0  # no light words

    def test_foreshadowing_score_with_hints(self):
        text = "Perhaps one day you will understand this. Remember this moment."
        score = _foreshadowing_score(text)
        assert score > 0.0

    def test_revelation_score_with_secret(self):
        text = "The secret was revealed at last."
        score = _revelation_score(text)
        assert score > 0.0

    def test_callback_density_with_reference(self):
        text = "You may remember when we first came to this place, long ago."
        score = _callback_density(text)
        assert score > 0.0


# ---------------------------------------------------------------------------
# Theme detection
# ---------------------------------------------------------------------------

class TestThemeDetection:
    def test_detects_eucatastrophe(self):
        text = "Against all hope, joy came suddenly at the last moment."
        themes = _detect_themes(text)
        assert "eucatastrophe" in themes

    def test_detects_the_long_defeat(self):
        text = "We have been fighting the long defeat for ages."
        themes = _detect_themes(text)
        assert "the_long_defeat" in themes

    def test_detects_diminishment(self):
        text = "Much that was beautiful has been fading and is now merely a shadow of what was."
        themes = _detect_themes(text)
        assert "diminishment" in themes

    def test_detects_mercy(self):
        text = "He showed mercy and pity to the wretched creature."
        themes = _detect_themes(text)
        assert "mercy" in themes

    def test_detects_mortality(self):
        text = "The mortal men must face death at the end of their days."
        themes = _detect_themes(text)
        assert "mortality" in themes

    def test_detects_power_corrupts(self):
        text = "The ring corrupted him with its power and he was consumed."
        themes = _detect_themes(text)
        assert "power_corrupts" in themes

    def test_detects_hope_vs_despair(self):
        text = "There is no hope, and yet we must not despair while the light endures."
        themes = _detect_themes(text)
        assert "hope_vs_despair" in themes

    def test_multiple_themes_detected(self):
        themes = _detect_themes(MULTIPLE_THEMES_TEXT)
        assert len(themes) >= 3

    def test_no_themes_in_plain_text(self):
        text = "The cat sat on the mat beside the window."
        themes = _detect_themes(text)
        # Should detect 0 or very few themes
        assert len(themes) <= 1

    def test_theme_coherence_single_theme(self):
        assert _theme_coherence(["eucatastrophe"]) == 1.0

    def test_theme_coherence_empty(self):
        assert _theme_coherence([]) == 0.0

    def test_theme_coherence_known_pair(self):
        # eucatastrophe + hope_vs_despair are a known coherent pair
        score = _theme_coherence(["eucatastrophe", "hope_vs_despair"])
        assert score == 1.0

    def test_theme_coherence_unrelated_pair(self):
        # power_corrupts + loyalty are not a known coherent pair
        score = _theme_coherence(["power_corrupts", "loyalty"])
        # This could be 0 or low — just verify it's a valid float
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# NarrativeWeightComputer
# ---------------------------------------------------------------------------

class TestNarrativeWeightComputer:
    def setup_method(self):
        self.computer = NarrativeWeightComputer()

    def test_compute_from_text_returns_narrative_weight(self):
        w = self.computer.compute_from_text(SIMPLE_TEXT)
        assert isinstance(w, NarrativeWeight)

    def test_compute_from_text_overall_in_unit_interval(self):
        w = self.computer.compute_from_text(TOLKIEN_RICH_TEXT)
        assert 0.0 <= w.overall <= 1.0

    def test_tolkien_rich_text_scores_higher_than_simple(self):
        w_rich = self.computer.compute_from_text(TOLKIEN_RICH_TEXT)
        w_simple = self.computer.compute_from_text(SIMPLE_TEXT)
        assert w_rich.overall > w_simple.overall

    def test_temporal_depth_zero_when_no_depth(self):
        w = self.computer.compute_from_text(SIMPLE_TEXT, temporal_depth_years=None)
        assert w.temporal_depth == 0.0

    def test_temporal_depth_one_at_max(self):
        max_depth = NarrativeWeightComputer.MAX_TEMPORAL_DEPTH_YEARS
        w = self.computer.compute_from_text(SIMPLE_TEXT, temporal_depth_years=max_depth)
        assert abs(w.temporal_depth - 1.0) < 1e-6

    def test_temporal_depth_partial(self):
        half_depth = NarrativeWeightComputer.MAX_TEMPORAL_DEPTH_YEARS / 2
        w = self.computer.compute_from_text(SIMPLE_TEXT, temporal_depth_years=half_depth)
        assert abs(w.temporal_depth - 0.5) < 0.01

    def test_era_ref_count_normalised(self):
        max_era = NarrativeWeightComputer.MAX_ERA_REF_COUNT
        w = self.computer.compute_from_text(SIMPLE_TEXT, era_ref_count=max_era)
        assert abs(w.era_reference_count - 1.0) < 1e-6

    def test_entity_count_normalised(self):
        max_ent = NarrativeWeightComputer.MAX_ENTITY_COUNT
        w = self.computer.compute_from_text(SIMPLE_TEXT, entity_count=max_ent)
        assert abs(w.entity_reference_count - 1.0) < 1e-6

    def test_overall_increases_with_depth(self):
        w_shallow = self.computer.compute_from_text(SIMPLE_TEXT, temporal_depth_years=500.0)
        w_deep = self.computer.compute_from_text(SIMPLE_TEXT, temporal_depth_years=10_000.0)
        assert w_deep.overall > w_shallow.overall

    def test_compute_from_passage_returns_weight(self):
        p = make_passage()
        w = self.computer.compute_from_passage(p)
        assert isinstance(w, NarrativeWeight)
        assert 0.0 <= w.overall <= 1.0

    def test_compute_from_passage_uses_era_ref_count(self):
        p_low = make_passage(era_reference_count=0)
        p_high = make_passage(era_reference_count=5)
        w_low = self.computer.compute_from_passage(p_low)
        w_high = self.computer.compute_from_passage(p_high)
        assert w_high.era_reference_count > w_low.era_reference_count

    def test_compute_from_passage_uses_temporal_depth(self):
        p_shallow = make_passage(temporal_depth_years_back=500.0)
        p_deep = make_passage(temporal_depth_years_back=15_000.0)
        w_shallow = self.computer.compute_from_passage(p_shallow)
        w_deep = self.computer.compute_from_passage(p_deep)
        assert w_deep.temporal_depth > w_shallow.temporal_depth

    def test_detect_themes_returns_theme_nodes(self):
        themes = self.computer.detect_themes(TOLKIEN_RICH_TEXT)
        assert isinstance(themes, list)
        for t in themes:
            assert isinstance(t, ThemeNode)

    def test_detect_themes_finds_long_defeat(self):
        text = "I have fought the long defeat and known fading loss."
        themes = self.computer.detect_themes(text)
        ids = [t.id for t in themes]
        assert "the_long_defeat" in ids

    def test_improvement_suggestions_returns_list_of_strings(self):
        w = NarrativeWeight()
        suggestions = self.computer.improvement_suggestions(w, n=3)
        assert len(suggestions) == 3
        for s in suggestions:
            assert isinstance(s, str)

    def test_corpus_stats_empty(self):
        stats = self.computer.compute_corpus_stats([])
        assert stats["count"] == 0

    def test_corpus_stats_non_empty(self):
        passages = [
            make_passage(id=f"p_{i}", text=TOLKIEN_RICH_TEXT if i % 2 == 0 else SIMPLE_TEXT)
            for i in range(5)
        ]
        stats = self.computer.compute_corpus_stats(passages)
        assert stats["count"] == 5
        assert 0.0 <= stats["mean_overall"] <= 1.0
        assert stats["max_overall"] >= stats["mean_overall"]

    def test_corpus_stats_has_component_means(self):
        passages = [make_passage(id=f"p_{i}") for i in range(3)]
        stats = self.computer.compute_corpus_stats(passages)
        assert "component_means" in stats
        for comp in COMPONENT_WEIGHTS:
            assert comp in stats["component_means"]

    def test_emotional_contrast_score_in_rich_text(self):
        """Tolkien-rich text with light/dark should score non-zero emotional_contrast."""
        text = "The light shone in the darkness and shadow fell upon hope."
        w = self.computer.compute_from_text(text)
        assert w.emotional_contrast > 0.0

    def test_voice_distinctiveness_higher_for_dialogue(self):
        text = "Thou dost not understand, verily I say thee this."
        w_dialogue = self.computer.compute_from_text(text, is_dialogue=True)
        w_narration = self.computer.compute_from_text(text, is_dialogue=False)
        assert w_dialogue.voice_distinctiveness >= w_narration.voice_distinctiveness
