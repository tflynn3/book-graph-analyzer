"""Tests for VoiceProfile extensions (Issue #10).

All tests are self-contained — no Neo4j connection required.
Graph-dependent tests are patched/skipped via unittest.mock.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from book_graph_analyzer.voice.profile import (
    CharacterVoiceProfile,
    _compute_formality_score,
    _is_rhetorical,
    _is_imperative,
    _compute_topic_distribution,
    MODERN_ANACHRONISMS,
)
from book_graph_analyzer.voice.dialogue import DialogueLine
from book_graph_analyzer.voice.audience import classify_audience, AUDIENCE_TYPES, CONTEXT_TYPES
from book_graph_analyzer.graph.voice_writer import (
    _extract_text_metrics,
    _compute_similarity,
    VoiceProfileWriter,
)


# ---------------------------------------------------------------------------
# Helpers to build DialogueLine fixtures
# ---------------------------------------------------------------------------

def make_line(
    text: str,
    speaker: str = "Gandalf",
    is_question: bool = False,
    is_exclamation: bool = False,
) -> DialogueLine:
    is_statement = not is_question and not is_exclamation
    return DialogueLine(
        text=text,
        speaker=speaker,
        is_question=is_question,
        is_exclamation=is_exclamation,
        is_statement=is_statement,
    )


# ---------------------
# Tolkien-esque corpus
# ---------------------

GANDALF_LINES = [
    make_line("You shall not pass!", is_exclamation=True),
    make_line("Fly, you fools!", is_exclamation=True),
    make_line("A wizard is never late, nor is he early; he arrives precisely when he means to."),
    make_line("Many that live deserve death. And some that die deserve life."),
    make_line("All we have to decide is what to do with the time that is given us."),
    make_line("Even the very wise cannot see all ends."),
    make_line("The treacherous are ever distrustful."),
    make_line("Verily I say unto thee, hark, for the hour is come."),
    make_line("Thou shalt not enter here without leave."),
    make_line("Behold, ere long the shadow shall fall upon us all."),
    make_line("Go back to the Shadow!", is_exclamation=True),
    make_line("Why should I fear thee, dark servant?", is_question=True),
    make_line("Hark! The hour of doom is nigh!", is_exclamation=True),
]

SAM_LINES = [
    make_line("I'm coming with you, Mr. Frodo."),
    make_line("I can't carry it for you, but I can carry you."),
    make_line("There's some good in this world, and it's worth fighting for."),
    make_line("Don't go where I can't follow."),
    make_line("I don't know half of you half as well as I should like."),
    make_line("Yeah, I know, it's heavy."),
    make_line("Okay, okay, I'll do it then."),
    make_line("What's taters, precious?", is_question=True),
    make_line("Hi there, Mr. Frodo, how are you today?", is_question=True),
    make_line("That Gollum, he's a nasty piece of work."),
    make_line("I don't feel right about this at all."),
    make_line("It's me, Sam, I've come to find you."),
]


def build_profile(name: str, lines: list[DialogueLine]) -> CharacterVoiceProfile:
    return CharacterVoiceProfile.from_dialogue_lines(name, lines)


# ===========================================================================
# TestCharacterVoiceProfile
# ===========================================================================

class TestCharacterVoiceProfile:

    def test_formality_score_gandalf_higher_than_sam(self):
        gandalf = build_profile("Gandalf", GANDALF_LINES)
        sam = build_profile("Sam", SAM_LINES)
        assert gandalf.formality_score > sam.formality_score, (
            f"Expected Gandalf ({gandalf.formality_score:.3f}) > Sam ({sam.formality_score:.3f})"
        )

    def test_formality_score_range(self):
        for name, lines in [("Gandalf", GANDALF_LINES), ("Sam", SAM_LINES)]:
            profile = build_profile(name, lines)
            assert 0.0 <= profile.formality_score <= 1.0, (
                f"{name}: formality_score {profile.formality_score} outside [0,1]"
            )

    def test_archaism_rate_detected(self):
        profile = build_profile("Gandalf", GANDALF_LINES)
        # Gandalf says "thee", "thou", "verily", "ere", "behold", "hark"
        assert profile.archaism_rate > 0.0, "Expected archaism_rate > 0 for archaic text"
        assert profile.archaism_count > 0

    def test_archaism_rate_zero_for_modern_text(self):
        modern_lines = [make_line(t) for t in [
            "Yeah, okay, I'm cool with that.",
            "Basically I wanna go now.",
            "Literally the best thing ever.",
        ]]
        profile = build_profile("Modern", modern_lines)
        assert profile.archaism_rate == 0.0

    def test_archaism_rate_is_proportion(self):
        profile = build_profile("Gandalf", GANDALF_LINES)
        assert 0.0 <= profile.archaism_rate <= 1.0

    def test_imperative_ratio_detected(self):
        imperative_lines = [
            make_line("Go back to the Shadow!", is_exclamation=True),
            make_line("Fly, you fools!", is_exclamation=True),
            make_line("Behold, the hour is come."),
            make_line("Hark! Listen well.", is_exclamation=True),
            make_line("Wait here and do not move."),
        ]
        profile = build_profile("Commander", imperative_lines)
        assert profile.imperative_ratio > 0.0, (
            f"Expected imperative_ratio > 0, got {profile.imperative_ratio}"
        )

    def test_imperative_ratio_range(self):
        profile = build_profile("Gandalf", GANDALF_LINES)
        assert 0.0 <= profile.imperative_ratio <= 1.0

    def test_rhetorical_density_detected(self):
        rhetorical_lines = [
            make_line("Why would you do such a thing?", is_question=True),
            make_line("How dare you speak of such matters?", is_question=True),
            make_line("Is it not clear that the shadow grows?", is_question=True),
            make_line("A simple statement here."),
            make_line("Another statement, not rhetorical."),
        ]
        profile = build_profile("Rhetor", rhetorical_lines)
        assert profile.rhetorical_density > 0.0, (
            f"Expected rhetorical_density > 0, got {profile.rhetorical_density}"
        )

    def test_rhetorical_density_range(self):
        profile = build_profile("Gandalf", GANDALF_LINES)
        assert 0.0 <= profile.rhetorical_density <= 1.0

    def test_topic_distribution_sums_to_one(self):
        profile = build_profile("Gandalf", GANDALF_LINES)
        total = sum(profile.topic_distribution.values())
        assert abs(total - 1.0) < 0.01, (
            f"topic_distribution values should sum to 1.0, got {total}"
        )

    def test_topic_distribution_all_keys_present(self):
        profile = build_profile("Gandalf", GANDALF_LINES)
        expected_topics = {"history", "war", "practical", "wisdom", "nature", "friendship"}
        assert expected_topics == set(profile.topic_distribution.keys())

    def test_audience_variant_metrics_computed(self):
        """With audience_lines supplied, formality_by_audience should be populated."""
        from book_graph_analyzer.voice.audience import classify_audience

        lines = GANDALF_LINES[:6]
        audience_lines = []
        for line in lines:
            aud, ctx = classify_audience(line, "")
            audience_lines.append((line, aud, ctx))

        profile = CharacterVoiceProfile.from_dialogue_lines(
            "Gandalf", lines, audience_lines=audience_lines
        )
        assert isinstance(profile.formality_by_audience, dict)
        assert isinstance(profile.length_by_audience, dict)
        assert isinstance(profile.register_by_audience, dict)
        for aud_type in profile.formality_by_audience:
            assert aud_type in AUDIENCE_TYPES, f"Unexpected audience type: {aud_type}"
            val = profile.formality_by_audience[aud_type]
            assert 0.0 <= val <= 1.0, f"formality_by_audience[{aud_type}] = {val}"

    def test_never_says_is_subset_of_modern_anachronisms(self):
        profile = build_profile("Gandalf", GANDALF_LINES)
        for word in profile.never_says:
            assert word in MODERN_ANACHRONISMS

    def test_never_says_excludes_words_character_actually_used(self):
        # Build a profile with a modern word
        modern_line = make_line("Yeah, okay, I'm totally fine with that.")
        profile = CharacterVoiceProfile.from_dialogue_lines("ModernChar", [modern_line])
        # "yeah" and "okay" are in MODERN_ANACHRONISMS; character DID say them
        # so they should NOT be in never_says
        assert "yeah" not in profile.never_says
        assert "okay" not in profile.never_says

    def test_empty_lines_returns_defaults(self):
        profile = CharacterVoiceProfile.from_dialogue_lines("Nobody", [])
        assert profile.total_lines == 0
        assert profile.formality_score == 0.0
        assert profile.archaism_rate == 0.0
        assert profile.imperative_ratio == 0.0
        assert profile.rhetorical_density == 0.0

    def test_backward_compat_existing_fields_still_present(self):
        profile = build_profile("Gandalf", GANDALF_LINES)
        # Existing fields must still be accessible
        assert hasattr(profile, "total_lines")
        assert hasattr(profile, "question_ratio")
        assert hasattr(profile, "exclamation_ratio")
        assert hasattr(profile, "statement_ratio")
        assert hasattr(profile, "contraction_ratio")
        assert hasattr(profile, "top_words")
        assert hasattr(profile, "distinctive_words")
        assert hasattr(profile, "archaism_count")
        assert hasattr(profile, "archaisms_used")
        assert hasattr(profile, "sample_quotes")
        assert hasattr(profile, "signature_phrases")


# ===========================================================================
# TestComputeFormalityScore
# ===========================================================================

class TestComputeFormalityScore:

    def test_all_archaic_is_formal(self):
        score = _compute_formality_score(0.05, 0.0, 7.0, 0.0)
        assert score >= 0.7

    def test_all_informal_is_low(self):
        score = _compute_formality_score(0.0, 0.20, 3.0, 0.10)
        assert score <= 0.4

    def test_range_is_zero_to_one(self):
        for arch in [0.0, 0.05, 0.10]:
            for contr in [0.0, 0.10, 0.20]:
                for wl in [3.0, 5.0, 8.0]:
                    for fp in [0.0, 0.05, 0.10]:
                        score = _compute_formality_score(arch, contr, wl, fp)
                        assert 0.0 <= score <= 1.0, (
                            f"Score {score} out of range for params "
                            f"(arch={arch}, contr={contr}, wl={wl}, fp={fp})"
                        )

    def test_more_archaisms_raises_score(self):
        low = _compute_formality_score(0.0, 0.05, 5.0, 0.03)
        high = _compute_formality_score(0.05, 0.05, 5.0, 0.03)
        assert high > low

    def test_more_contractions_lowers_score(self):
        formal = _compute_formality_score(0.02, 0.0, 5.0, 0.03)
        informal = _compute_formality_score(0.02, 0.15, 5.0, 0.03)
        assert informal < formal


# ===========================================================================
# TestHelperFunctions
# ===========================================================================

class TestHelperFunctions:

    def test_is_rhetorical_why_would(self):
        assert _is_rhetorical("Why would you do such a thing?")

    def test_is_rhetorical_how_dare(self):
        assert _is_rhetorical("How dare you enter here?")

    def test_is_rhetorical_is_it_not(self):
        assert _is_rhetorical("Is it not clear that the shadow grows?")

    def test_is_not_rhetorical_plain_question(self):
        assert not _is_rhetorical("What is your name?")

    def test_is_not_rhetorical_no_question_mark(self):
        assert not _is_rhetorical("Why would you do such a thing")

    def test_is_imperative_go(self):
        assert _is_imperative("Go back to the Shadow!")

    def test_is_imperative_fly(self):
        assert _is_imperative("Fly, you fools")

    def test_is_imperative_beware(self):
        assert _is_imperative("beware the shadow that follows")

    def test_is_not_imperative_statement(self):
        assert not _is_imperative("He went back to the shadow.")

    def test_topic_distribution_sums_to_one(self):
        words = "battle sword enemy war fight ancient ages kingdom".split()
        dist = _compute_topic_distribution(words)
        total = sum(dist.values())
        assert abs(total - 1.0) < 0.01

    def test_topic_distribution_has_all_keys(self):
        dist = _compute_topic_distribution(["hello"])
        assert set(dist.keys()) == {"history", "war", "practical", "wisdom", "nature", "friendship"}


# ===========================================================================
# TestAudienceClassifier
# ===========================================================================

class TestAudienceClassifier:

    def test_imperative_classified_as_command(self):
        line = make_line("Go back to the Shadow!", is_exclamation=True)
        _, context = classify_audience(line, "")
        assert context == "command"

    def test_farewell_phrase_classified(self):
        line = make_line("Farewell, may your journey be safe.")
        _, context = classify_audience(line, "")
        assert context == "farewell"

    def test_comfort_phrase_classified(self):
        line = make_line("Fear not, all will be well in the end.")
        _, context = classify_audience(line, "")
        assert context == "comfort"

    def test_warning_phrase_classified(self):
        line = make_line("Beware the shadow that creeps from the east.")
        _, context = classify_audience(line, "")
        assert context == "warning"

    def test_unknown_defaults_to_neutral_explanation(self):
        line = make_line("The sky is blue and the stars are bright.")
        aud, ctx = classify_audience(line, "")
        # audience defaults to neutral when no names
        assert aud == "neutral"
        # context defaults to explanation
        assert ctx == "explanation"

    def test_hobbit_name_in_context_classifies_hobbit(self):
        line = make_line("I hope you are well.")
        aud, _ = classify_audience(line, "Frodo looked up at him.")
        assert aud == "hobbit"

    def test_elf_name_in_context_classifies_elf(self):
        line = make_line("There is much wisdom in your words.")
        aud, _ = classify_audience(line, "Galadriel smiled graciously.")
        assert aud == "elf"

    def test_dwarf_name_in_text_classifies_dwarf(self):
        line = make_line("Gimli, come forward.")
        aud, _ = classify_audience(line, "")
        assert aud == "dwarf"

    def test_audience_type_in_valid_set(self):
        for text in [
            "Fear not, friend.",
            "Go! Now!",
            "Farewell and safe travels.",
            "Tell me what you know.",
        ]:
            line = make_line(text)
            aud, ctx = classify_audience(line, "")
            assert aud in AUDIENCE_TYPES, f"Unexpected audience type: {aud}"
            assert ctx in CONTEXT_TYPES, f"Unexpected context type: {ctx}"

    def test_context_type_in_valid_set(self):
        line = make_line("Because the ancient lore tells us so.")
        _, ctx = classify_audience(line, "")
        assert ctx in CONTEXT_TYPES


# ===========================================================================
# TestVoiceSpeakerIdentify (mocked Neo4j)
# ===========================================================================

class TestVoiceSpeakerIdentify:

    def _make_writer_with_profiles(self, profiles: list[dict]) -> VoiceProfileWriter:
        """Return a VoiceProfileWriter whose driver returns canned profile dicts."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Build mock rows from profile dicts
        def make_row(d: dict):
            row = MagicMock()
            row.__getitem__ = lambda self, key: d.get(key)
            row.get = lambda key, default=None: d.get(key, default)
            # Support dict(row) pattern
            row.keys = lambda: d.keys()
            row.values = lambda: d.values()
            row.items = lambda: d.items()
            return row

        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter(profiles))
        mock_session.run.return_value = mock_result

        writer = VoiceProfileWriter(driver=mock_driver)
        return writer

    def test_identify_returns_sorted_by_confidence(self):
        """Matches should be sorted highest confidence first."""
        profiles = [
            {
                "cid": "gandalf",
                "formality_score": 0.85,
                "archaism_rate": 0.08,
                "question_ratio": 0.10,
                "exclamation_ratio": 0.15,
                "imperative_ratio": 0.30,
                "rhetorical_density": 0.05,
                "contraction_usage": 0.01,
                "avg_utterance_length": 10.0,
            },
            {
                "cid": "sam",
                "formality_score": 0.25,
                "archaism_rate": 0.00,
                "question_ratio": 0.20,
                "exclamation_ratio": 0.10,
                "imperative_ratio": 0.05,
                "rhetorical_density": 0.00,
                "contraction_usage": 0.12,
                "avg_utterance_length": 8.0,
            },
        ]
        writer = self._make_writer_with_profiles(profiles)

        # Text that sounds very Gandalf-like
        text = "Thou shalt not enter! Beware the darkness that follows thee!"
        results = writer.identify_speaker(text, top_n=2)

        assert len(results) == 2
        # Results must be sorted descending
        confidences = [c for _, c in results]
        assert confidences == sorted(confidences, reverse=True), (
            f"Results not sorted by confidence: {results}"
        )

    def test_confidence_between_zero_and_one(self):
        """All confidence scores must lie in [0.0, 1.0]."""
        profiles = [
            {
                "cid": "frodo",
                "formality_score": 0.50,
                "archaism_rate": 0.01,
                "question_ratio": 0.15,
                "exclamation_ratio": 0.05,
                "imperative_ratio": 0.05,
                "rhetorical_density": 0.00,
                "contraction_usage": 0.05,
                "avg_utterance_length": 7.0,
            },
        ]
        writer = self._make_writer_with_profiles(profiles)
        results = writer.identify_speaker("I must carry the Ring alone.", top_n=1)

        for _, conf in results:
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of range [0, 1]"

    def test_identify_returns_at_most_top_n(self):
        profiles = [
            {"cid": f"char_{i}", "formality_score": 0.5, "archaism_rate": 0.0,
             "question_ratio": 0.1, "exclamation_ratio": 0.1, "imperative_ratio": 0.1,
             "rhetorical_density": 0.0, "contraction_usage": 0.05, "avg_utterance_length": 8.0}
            for i in range(5)
        ]
        writer = self._make_writer_with_profiles(profiles)
        results = writer.identify_speaker("A simple sentence.", top_n=3)
        assert len(results) <= 3

    def test_identify_empty_profiles_returns_empty(self):
        writer = self._make_writer_with_profiles([])
        results = writer.identify_speaker("You shall not pass!")
        assert results == []


# ===========================================================================
# TestExtractTextMetrics
# ===========================================================================

class TestExtractTextMetrics:

    def test_returns_all_expected_keys(self):
        metrics = _extract_text_metrics("You shall not pass!")
        expected = {
            "formality_score", "archaism_rate", "question_ratio",
            "exclamation_ratio", "imperative_ratio", "rhetorical_density",
            "contraction_usage", "avg_utterance_length",
        }
        assert expected == set(metrics.keys())

    def test_archaic_text_has_high_formality(self):
        text = "Thou shalt not enter here. Behold, ere long the shadow cometh."
        metrics = _extract_text_metrics(text)
        assert metrics["archaism_rate"] > 0.0
        assert metrics["formality_score"] > 0.5

    def test_modern_text_has_low_formality(self):
        text = "Yeah, I'm gonna go now. I don't really wanna stay."
        metrics = _extract_text_metrics(text)
        assert metrics["formality_score"] < 0.5

    def test_all_values_in_expected_range(self):
        text = "Go back! Thou shall not pass. Is it not time to leave?"
        metrics = _extract_text_metrics(text)
        for key, val in metrics.items():
            if key == "avg_utterance_length":
                assert val >= 0.0
            else:
                assert 0.0 <= val <= 1.0, f"{key}={val} out of range"


# ===========================================================================
# TestComputeSimilarity
# ===========================================================================

class TestComputeSimilarity:

    def test_identical_profiles_give_similarity_one(self):
        profile = {
            "formality_score": 0.80,
            "archaism_rate": 0.05,
            "question_ratio": 0.10,
            "exclamation_ratio": 0.10,
            "imperative_ratio": 0.20,
            "rhetorical_density": 0.05,
            "contraction_usage": 0.02,
            "avg_utterance_length": 10.0,
        }
        query = dict(profile)
        sim = _compute_similarity(query, profile)
        assert abs(sim - 1.0) < 1e-6

    def test_opposite_profiles_give_low_similarity(self):
        query = {
            "formality_score": 1.0,
            "archaism_rate": 1.0,
            "question_ratio": 0.0,
            "exclamation_ratio": 0.0,
            "imperative_ratio": 1.0,
            "rhetorical_density": 1.0,
            "contraction_usage": 0.0,
            "avg_utterance_length": 1.0,  # after normalisation
        }
        profile = {
            "formality_score": 0.0,
            "archaism_rate": 0.0,
            "question_ratio": 1.0,
            "exclamation_ratio": 1.0,
            "imperative_ratio": 0.0,
            "rhetorical_density": 0.0,
            "contraction_usage": 1.0,
            "avg_utterance_length": 30.0,  # after normalisation → 1.0
        }
        sim = _compute_similarity(query, profile)
        assert sim < 0.5
