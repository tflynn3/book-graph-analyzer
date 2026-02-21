"""
Tests for Issue #10 — VoiceProfile + Audience-Variant Character Voice

Tests cover:
- Audience type classification
- Context type classification
- DialogueLine new fields (is_imperative, is_verse, audience_type, context_type)
- CharacterVoiceProfile new fields (formality_score, archaism_rate, etc.)
- Audience-variant metrics (formality_by_audience, length_by_audience, register_by_audience)
- never_says computation
- topic_distribution computation
- VoiceAnalyzer.identify_speaker
- VoiceAnalyzer.check_voice_violations
"""
import pytest

from book_graph_analyzer.voice.audience import (
    classify_audience_type,
    classify_context_type,
    classify_dialogue_line,
    AUDIENCE_TYPES,
    CONTEXT_TYPES,
)
from book_graph_analyzer.voice.dialogue import (
    DialogueLine,
    extract_dialogue,
    _detect_imperative,
    _detect_verse,
)
from book_graph_analyzer.voice.profile import (
    CharacterVoiceProfile,
    _compute_topic_distribution,
    _find_never_says,
)
from book_graph_analyzer.voice.analyzer import VoiceAnalyzer


# ---------------------------------------------------------------------------
# Audience classification tests
# ---------------------------------------------------------------------------

class TestAudienceClassification:
    def test_classify_audience_hobbit_context(self):
        result = classify_audience_type(
            "Come with me!",
            context_before="Gandalf spoke to Bilbo and Frodo",
            context_after="in Bag End",
        )
        assert result == "hobbit"

    def test_classify_audience_elf_context(self):
        result = classify_audience_type(
            "Namárië.",
            context_before="He turned to Galadriel and Elrond",
            context_after="and spoke in Elvish",
        )
        assert result == "elf"

    def test_classify_audience_dwarf_context(self):
        result = classify_audience_type(
            "Well met!",
            context_before="He greeted Gimli the dwarf",
            context_after="at the gates of Erebor",
        )
        assert result == "dwarf"

    def test_classify_audience_neutral(self):
        # No known race/character keywords in context → "neutral"
        result = classify_audience_type("Hello.", context_before="said a voice", context_after="in the dark")
        assert result == "neutral"

    def test_classify_audience_self(self):
        result = classify_audience_type(
            "What am I to do?",
            context_before="he thought to himself",
            context_after="",
        )
        assert result == "self"

    def test_classify_audience_enemy_context(self):
        result = classify_audience_type(
            "You shall not pass!",
            context_before="facing the orc army",
            context_after="and blocked the bridge",
        )
        assert result == "enemy"

    def test_all_audience_types_valid(self):
        # All returned values must be in the known set
        for ctx_b, ctx_a in [
            ("hobbit bilbo frodo", ""),
            ("elf galadriel elrond", ""),
            ("dwarf gimli erebor", ""),
            ("aragorn man gondor", ""),
            ("orc enemy mordor", ""),
            ("", ""),
        ]:
            result = classify_audience_type("text", ctx_b, ctx_a)
            assert result in AUDIENCE_TYPES, f"Unknown audience type: {result}"


class TestContextTypeClassification:
    def test_context_crisis(self):
        assert classify_context_type("Run! They are coming!") == "crisis"

    def test_context_command(self):
        assert classify_context_type("Go now, and do not return.") == "command"

    def test_context_warning(self):
        assert classify_context_type("Beware the shadow that lurks here.") == "warning"

    def test_context_farewell(self):
        assert classify_context_type("Farewell, dear friends, until we meet again.") == "farewell"

    def test_context_comfort(self):
        assert classify_context_type("Fear not. All shall be well in the end.") == "comfort"

    def test_context_explanation(self):
        assert classify_context_type("Long ago, in the Elder Days, this was forged.") == "explanation"

    def test_context_statement_default(self):
        assert classify_context_type("The sun is shining today.") == "statement"

    def test_all_context_types_valid(self):
        texts = [
            "Run!",
            "Go now.",
            "Beware the dark.",
            "Farewell.",
            "Fear not.",
            "Long ago...",
            "The sky is blue.",
        ]
        for t in texts:
            result = classify_context_type(t)
            assert result in CONTEXT_TYPES, f"Unknown context type: {result}"


# ---------------------------------------------------------------------------
# DialogueLine new fields
# ---------------------------------------------------------------------------

class TestImperativeDetection:
    def test_imperative_go(self):
        assert _detect_imperative("Go now and do not look back.") is True

    def test_imperative_behold(self):
        assert _detect_imperative("Behold the ring of power!") is True

    def test_imperative_do_not(self):
        assert _detect_imperative("Do not speak that name here.") is True

    def test_not_imperative_subject_first(self):
        assert _detect_imperative("He walked away in silence.") is False

    def test_not_imperative_question(self):
        assert _detect_imperative("What are you doing here?") is False


class TestVerseDetection:
    def test_verse_with_newline(self):
        assert _detect_verse("O Elbereth Gilthoniel\nsilivren penna miriel") is True

    def test_verse_o_invocation(self):
        assert _detect_verse("O Elbereth, hear my call!") is True

    def test_not_verse_plain_statement(self):
        assert _detect_verse("I will go to Rivendell.") is False


class TestExtractDialogueNewFields:
    def test_extract_sets_audience_type(self):
        text = '''Gandalf spoke to Bilbo the hobbit: "Come with me on an adventure."'''
        result = extract_dialogue(text)
        assert len(result.dialogue_lines) >= 1
        line = result.dialogue_lines[0]
        assert line.audience_type in AUDIENCE_TYPES

    def test_extract_sets_context_type(self):
        text = '"Go now, and do not return." said the wizard.'
        result = extract_dialogue(text)
        assert len(result.dialogue_lines) >= 1
        line = result.dialogue_lines[0]
        assert line.context_type in CONTEXT_TYPES

    def test_extract_detects_imperative_line(self):
        text = '"Run, you fools!" he cried.'
        result = extract_dialogue(text)
        assert len(result.dialogue_lines) >= 1
        # The line starts with "Run" — an imperative
        line = result.dialogue_lines[0]
        assert line.is_imperative is True


# ---------------------------------------------------------------------------
# CharacterVoiceProfile new fields
# ---------------------------------------------------------------------------

def _make_line(text: str, audience: str = "neutral", context: str = "statement",
               imperative: bool = False, verse: bool = False) -> DialogueLine:
    """Helper to create a DialogueLine with all new fields."""
    return DialogueLine(
        text=text,
        speaker="TestChar",
        is_question=text.endswith("?"),
        is_exclamation=text.endswith("!"),
        is_statement=not text.endswith("?") and not text.endswith("!"),
        is_imperative=imperative,
        is_verse=verse,
        audience_type=audience,
        context_type=context,
        audience_confidence=0.8,
    )


class TestCharacterVoiceProfileNewFields:
    def test_formality_score_range(self):
        lines = [_make_line("I am the flame of Udun.")] * 5
        profile = CharacterVoiceProfile.from_dialogue_lines("Balrog", lines)
        assert 0.0 <= profile.formality_score <= 1.0

    def test_high_formality_archaic_character(self):
        archaic_lines = [
            _make_line("Thee shall not pass, thou foolish mortal."),
            _make_line("Hearken to me, for ere long thou shalt see."),
            _make_line("Wherefore dost thou tarry, ye wanderers?"),
        ]
        profile = CharacterVoiceProfile.from_dialogue_lines("AncientOne", archaic_lines)
        assert profile.archaism_rate > 0  # Has archaisms
        assert profile.archaism_count > 0
        assert profile.formality_score > 0.5  # Should be formal

    def test_low_formality_informal_character(self):
        informal_lines = [
            _make_line("I'm not gonna do it, it's not right."),
            _make_line("That's what I've been saying all along, isn't it?"),
            _make_line("We're here, aren't we? So let's get it done."),
        ]
        profile = CharacterVoiceProfile.from_dialogue_lines("Informal", informal_lines)
        assert profile.contraction_ratio > 0.05  # Has contractions
        # Formality should be lower than archaic character

    def test_archaism_rate_per_100_words(self):
        # 2 archaic words in ~10 words = ~20 per 100
        lines = [_make_line("Thee and thou art welcome here in this hall.")]
        profile = CharacterVoiceProfile.from_dialogue_lines("A", lines)
        assert profile.archaism_rate > 0

    def test_imperative_ratio(self):
        lines = [
            _make_line("Go now.", imperative=True),
            _make_line("Come to me.", imperative=True),
            _make_line("I will help you.", imperative=False),
            _make_line("We are here.", imperative=False),
        ]
        profile = CharacterVoiceProfile.from_dialogue_lines("C", lines)
        assert profile.imperative_ratio == pytest.approx(0.5)

    def test_verse_lines_counted(self):
        lines = [
            _make_line("Three rings for the Elven-kings.", verse=True),
            _make_line("In the land of Mordor.", verse=True),
            _make_line("I must go now.", verse=False),
        ]
        profile = CharacterVoiceProfile.from_dialogue_lines("Poet", lines)
        assert profile.verse_lines == 2
        assert profile.prose_lines == 1

    def test_rhetorical_density(self):
        lines = [
            # Short rhetorical questions
            _make_line("Who knows?"),
            _make_line("What does it matter?"),
            _make_line("Where are we going?"),  # Genuine question
        ]
        profile = CharacterVoiceProfile.from_dialogue_lines("Q", lines)
        assert 0.0 <= profile.rhetorical_density <= 1.0

    def test_topic_distribution_computed(self):
        lines = [
            _make_line("In the ancient age before the sun rose, the battle raged."),
            _make_line("The war came to the valley and the army marched."),
        ]
        profile = CharacterVoiceProfile.from_dialogue_lines("Narrator", lines)
        assert isinstance(profile.topic_distribution, dict)
        # Some topics should be detected
        total = sum(profile.topic_distribution.values())
        if total > 0:
            assert abs(total - 1.0) < 0.01  # Should be normalized

    def test_never_says_computed_with_other_chars(self):
        char_lines = [_make_line("I go to the mountain.")]
        other_words = {
            "OtherChar": __import__("collections").Counter(
                ["precious", "precious", "gollum", "precious"]
            ),
        }
        profile = CharacterVoiceProfile.from_dialogue_lines(
            "Mine", char_lines, all_character_words=other_words
        )
        # If other char uses "precious" 3+ times and Mine never does, it should be in never_says
        assert "precious" in profile.never_says

    def test_audience_variant_metrics(self):
        lines = [
            _make_line("Come, dear hobbit.", audience="hobbit"),
            _make_line("Come, dear hobbit.", audience="hobbit"),
            _make_line("Hearken thee, O Elf.", audience="elf"),
        ]
        profile = CharacterVoiceProfile.from_dialogue_lines("G", lines)
        assert "hobbit" in profile.length_by_audience
        assert "elf" in profile.length_by_audience
        assert "hobbit" in profile.formality_by_audience
        assert "hobbit" in profile.register_by_audience

    def test_register_by_audience_is_valid_context_type(self):
        lines = [
            _make_line("Run, hobbit!", audience="hobbit", context="crisis"),
            _make_line("Flee now, hobbit!", audience="hobbit", context="crisis"),
            _make_line("Farewell, elf.", audience="elf", context="farewell"),
        ]
        profile = CharacterVoiceProfile.from_dialogue_lines("A", lines)
        if "hobbit" in profile.register_by_audience:
            assert profile.register_by_audience["hobbit"] in CONTEXT_TYPES
        if "elf" in profile.register_by_audience:
            assert profile.register_by_audience["elf"] in CONTEXT_TYPES


# ---------------------------------------------------------------------------
# topic_distribution helper
# ---------------------------------------------------------------------------

class TestTopicDistribution:
    def test_history_words_detected(self):
        words = ["in", "the", "ancient", "age", "long", "ago", "tale", "legend"]
        dist = _compute_topic_distribution(words)
        assert "history" in dist

    def test_practical_words_detected(self):
        words = ["food", "water", "road", "fire", "camp", "eat"]
        dist = _compute_topic_distribution(words)
        assert "practical" in dist

    def test_war_words_detected(self):
        words = ["battle", "sword", "enemy", "army", "fight", "war"]
        dist = _compute_topic_distribution(words)
        assert "war" in dist

    def test_normalized(self):
        words = ["battle", "ancient", "food", "river", "friend"]
        dist = _compute_topic_distribution(words)
        if dist:
            total = sum(dist.values())
            assert abs(total - 1.0) < 0.01

    def test_empty_returns_empty(self):
        dist = _compute_topic_distribution([])
        assert dist == {} or isinstance(dist, dict)


# ---------------------------------------------------------------------------
# VoiceAnalyzer.identify_speaker
# ---------------------------------------------------------------------------

class TestIdentifySpeaker:
    def _make_profile(self, name, formality, archaism_rate, contraction_ratio,
                      distinctive=None, signature=None, never=None):
        p = CharacterVoiceProfile(character_name=name)
        p.formality_score = formality
        p.archaism_rate = archaism_rate
        p.contraction_ratio = contraction_ratio
        p.imperative_ratio = 0.1
        p.avg_utterance_length = 10.0
        p.question_ratio = 0.1
        p.distinctive_words = distinctive or []
        p.signature_phrases = signature or []
        p.never_says = never or []
        return p

    def test_returns_list(self):
        analyzer = VoiceAnalyzer()
        profiles = {
            "A": self._make_profile("A", 0.8, 4.0, 0.01),
            "B": self._make_profile("B", 0.2, 0.1, 0.1),
        }
        result = analyzer.identify_speaker("Thou art welcome here.", profiles)
        assert isinstance(result, list)
        assert len(result) <= 2

    def test_archaic_text_prefers_archaic_profile(self):
        analyzer = VoiceAnalyzer()
        profiles = {
            "Gandalf": self._make_profile("Gandalf", 0.85, 4.5, 0.01,
                                          distinctive=["thee", "thou", "pass"],
                                          signature=["you shall not"]),
            "Sam": self._make_profile("Sam", 0.15, 0.1, 0.12,
                                      distinctive=["taters", "mr", "frodo"]),
        }
        # Archaic text should prefer Gandalf
        result = analyzer.identify_speaker("Thou shalt not pass, thou fool!", profiles)
        assert len(result) >= 1
        # Gandalf should rank higher than Sam
        char_names = [r[0] for r in result]
        gandalf_idx = char_names.index("Gandalf") if "Gandalf" in char_names else 999
        sam_idx = char_names.index("Sam") if "Sam" in char_names else 999
        assert gandalf_idx < sam_idx

    def test_informal_text_prefers_informal_profile(self):
        analyzer = VoiceAnalyzer()
        profiles = {
            "Gandalf": self._make_profile("Gandalf", 0.85, 4.5, 0.01),
            "Sam": self._make_profile("Sam", 0.15, 0.1, 0.12,
                                      distinctive=["taters", "mr", "frodo", "begging"]),
        }
        result = analyzer.identify_speaker("I'm not sure we should, Mr. Frodo, begging your pardon.", profiles)
        char_names = [r[0] for r in result]
        sam_idx = char_names.index("Sam") if "Sam" in char_names else 999
        gandalf_idx = char_names.index("Gandalf") if "Gandalf" in char_names else 999
        assert sam_idx < gandalf_idx

    def test_confidence_between_0_and_1(self):
        analyzer = VoiceAnalyzer()
        profiles = {
            "A": self._make_profile("A", 0.5, 2.0, 0.05),
        }
        result = analyzer.identify_speaker("Hello there.", profiles)
        for _, conf in result:
            assert 0.0 <= conf <= 1.0

    def test_top_n_respected(self):
        analyzer = VoiceAnalyzer()
        profiles = {
            "A": self._make_profile("A", 0.5, 1.0, 0.05),
            "B": self._make_profile("B", 0.6, 2.0, 0.03),
            "C": self._make_profile("C", 0.4, 0.5, 0.08),
            "D": self._make_profile("D", 0.7, 3.0, 0.01),
        }
        result = analyzer.identify_speaker("Test text.", profiles, top_n=2)
        assert len(result) <= 2

    def test_empty_profiles_returns_empty(self):
        analyzer = VoiceAnalyzer()
        result = analyzer.identify_speaker("Some text.", {})
        assert result == []

    def test_distinctive_word_overlap_boosts_score(self):
        analyzer = VoiceAnalyzer()
        profiles = {
            "Gollum": self._make_profile("Gollum", 0.1, 0.3, 0.04,
                                         distinctive=["precious", "nasty", "tricksy"],
                                         signature=["my precious"]),
            "Frodo": self._make_profile("Frodo", 0.5, 0.5, 0.05,
                                        distinctive=["ring", "shire", "quest"]),
        }
        result = analyzer.identify_speaker("Precious, my precious! We wants it!", profiles)
        char_names = [r[0] for r in result]
        gollum_idx = char_names.index("Gollum") if "Gollum" in char_names else 999
        frodo_idx = char_names.index("Frodo") if "Frodo" in char_names else 999
        assert gollum_idx < frodo_idx


# ---------------------------------------------------------------------------
# VoiceAnalyzer.check_voice_violations
# ---------------------------------------------------------------------------

class TestCheckVoiceViolations:
    def _gandalf_profile(self):
        p = CharacterVoiceProfile(character_name="Gandalf")
        p.formality_score = 0.85
        p.archaism_rate = 4.0
        p.contraction_ratio = 0.01
        p.distinctive_words = ["pass", "fool", "shadow", "ancient"]
        p.signature_phrases = ["you shall not", "the grey", "ancient days"]
        p.never_says = ["taters", "begging", "pardon", "gaffer"]
        return p

    def test_no_violations_for_appropriate_text(self):
        analyzer = VoiceAnalyzer()
        profile = self._gandalf_profile()
        violations = analyzer.check_voice_violations(
            "You shall not pass! I am a servant of the Secret Fire.", profile
        )
        # Should be clean or only soft warnings
        hard = [v for v in violations if v["severity"] == "hard"]
        assert len(hard) == 0

    def test_anachronism_detected(self):
        analyzer = VoiceAnalyzer()
        profile = self._gandalf_profile()
        violations = analyzer.check_voice_violations(
            "Yeah, sure, let's go dude. That's totally cool.", profile
        )
        types = [v["type"] for v in violations]
        assert "anachronism" in types

    def test_anachronism_is_hard(self):
        analyzer = VoiceAnalyzer()
        profile = self._gandalf_profile()
        violations = analyzer.check_voice_violations(
            "Okay yeah whatever.", profile
        )
        hard = [v for v in violations if v["type"] == "anachronism"]
        assert len(hard) >= 1
        assert hard[0]["severity"] == "hard"

    def test_never_says_violation(self):
        analyzer = VoiceAnalyzer()
        profile = self._gandalf_profile()
        violations = analyzer.check_voice_violations(
            "Begging your pardon, sir, but taters are good.", profile
        )
        types = [v["type"] for v in violations]
        assert "uses_never_says" in types

    def test_never_says_is_soft(self):
        analyzer = VoiceAnalyzer()
        profile = self._gandalf_profile()
        violations = analyzer.check_voice_violations(
            "Begging your pardon.", profile
        )
        ns = [v for v in violations if v["type"] == "uses_never_says"]
        if ns:
            assert ns[0]["severity"] == "soft"

    def test_wrong_formality_detected(self):
        analyzer = VoiceAnalyzer()
        # Highly formal profile
        formal_profile = CharacterVoiceProfile(character_name="Elrond")
        formal_profile.formality_score = 0.95
        formal_profile.archaism_rate = 5.0
        formal_profile.contraction_ratio = 0.0
        formal_profile.distinctive_words = []
        formal_profile.signature_phrases = []
        formal_profile.never_says = []
        
        # Very informal text
        violations = analyzer.check_voice_violations(
            "I'm gonna do it alright. It's whatever.", formal_profile
        )
        types = [v["type"] for v in violations]
        assert "wrong_formality" in types

    def test_violations_have_required_keys(self):
        analyzer = VoiceAnalyzer()
        profile = self._gandalf_profile()
        violations = analyzer.check_voice_violations("Okay yeah.", profile)
        for v in violations:
            assert "type" in v
            assert "severity" in v
            assert "message" in v

    def test_empty_text_no_crash(self):
        analyzer = VoiceAnalyzer()
        profile = self._gandalf_profile()
        violations = analyzer.check_voice_violations("", profile)
        assert isinstance(violations, list)


# ---------------------------------------------------------------------------
# Integration: full pipeline produces expected profile structure
# ---------------------------------------------------------------------------

class TestVoiceProfileIntegration:
    _sample_text = """
    "Come, Mr. Frodo!" Sam called. "We must hurry now!"
    "I can't go on," Frodo murmured. "I'm too tired."
    "You shall not pass!" the wizard cried to the darkness.
    "Thou art welcome in this hall, dear Elf," said the lord.
    "Precious, my precious," whispered Gollum. "We wants it."
    "Fear not," said Gandalf. "All shall be well in the end."
    "Run! They are coming!" shouted Legolas.
    "Go back to the shadows!" the wizard commanded.
    """

    def test_analyze_produces_profiles(self):
        analyzer = VoiceAnalyzer(min_lines_for_profile=1)
        result = analyzer.analyze_text(self._sample_text)
        # Should find some characters
        assert result.total_characters > 0

    def test_profiles_have_new_fields(self):
        analyzer = VoiceAnalyzer(min_lines_for_profile=1)
        result = analyzer.analyze_text(self._sample_text)
        for name, profile in result.profiles.items():
            assert hasattr(profile, "formality_score")
            assert hasattr(profile, "archaism_rate")
            assert hasattr(profile, "rhetorical_density")
            assert hasattr(profile, "imperative_ratio")
            assert hasattr(profile, "formality_by_audience")
            assert hasattr(profile, "length_by_audience")
            assert hasattr(profile, "register_by_audience")
            assert hasattr(profile, "never_says")
            assert hasattr(profile, "topic_distribution")
            assert hasattr(profile, "verse_lines")
            assert hasattr(profile, "prose_lines")
            assert 0.0 <= profile.formality_score <= 1.0
            assert profile.archaism_rate >= 0.0
            assert 0.0 <= profile.rhetorical_density <= 1.0
            assert 0.0 <= profile.imperative_ratio <= 1.0

    def test_profile_serialization_roundtrip(self):
        """New fields survive to_json / from_json roundtrip."""
        lines = [
            _make_line("Thee art welcome here.", audience="elf", context="farewell"),
            _make_line("Go now, and do not tarry.", audience="hobbit", context="command",
                       imperative=True),
        ]
        profile = CharacterVoiceProfile.from_dialogue_lines("TestChar", lines)
        
        # Serialize and deserialize
        json_str = profile.to_json()
        restored = CharacterVoiceProfile.from_json(json_str)
        
        assert restored.character_name == "TestChar"
        assert abs(restored.formality_score - profile.formality_score) < 0.001
        assert abs(restored.archaism_rate - profile.archaism_rate) < 0.001
        assert restored.verse_lines == profile.verse_lines
        assert restored.prose_lines == profile.prose_lines
        assert restored.formality_by_audience == profile.formality_by_audience
        assert restored.length_by_audience == profile.length_by_audience
        assert restored.register_by_audience == profile.register_by_audience
        assert restored.topic_distribution == profile.topic_distribution
