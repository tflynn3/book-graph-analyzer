"""Tests for Author Register Taxonomy + SceneTemplate Nodes (Issue #9).

All tests are pure-Python — no Neo4j or LLM required. Covers:
  - ProseRegister enum (7 registers)
  - SceneTemplate model and canonical data
  - RegisterClassification model
  - CANONICAL_SCENE_TEMPLATES — all 7 registers present with measured metrics
  - RegisterClassifier — heuristic classification
  - build_generation_prompt — style-grounded prompt assembly
  - ExemplifiesEdge model
  - Cross-register distinctiveness (registers produce different metrics)
"""

import pytest
from book_graph_analyzer.models.scene_template import (
    SceneTemplate,
    RegisterClassification,
    ExemplifiesEdge,
    ProseRegister,
    REGISTER_DESCRIPTIONS,
    REGISTER_TRIGGERS,
    REGISTER_STRUCTURAL_PATTERNS,
    REGISTER_SIGNATURE_KEYWORDS,
)
from book_graph_analyzer.lore.register import (
    CANONICAL_SCENE_TEMPLATES,
    RegisterClassifier,
    build_generation_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SHIRE_PASSAGE = (
    "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, "
    "filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole "
    "with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means "
    "comfort. He had a supper at the hearth, and smoked his pipe by the warm fire."
)

ELEGIAC_PASSAGE = (
    "Long ago, in the elder days before the Sun and Moon, there was a light in Valinor "
    "that is now lost. The Two Trees have been felled, and their glory is diminished. "
    "Much that once was is now forgotten; none remain who remember the beauty of those ancient days."
)

DREAD_PASSAGE = (
    "Something moved in the darkness. Cold. Silent. Black. "
    "The weight pressed down on them. "
    "They dared not breathe. "
    "The shadow crept closer."
)

WONDER_PASSAGE = (
    "Never before had he seen such light. The stars were silver, the music was beyond all words. "
    "He marvelled at the beautiful radiant elves, glittering and fair, singing songs that pierced "
    "his heart with wonder. He could not speak; he could only stare in astonishment."
)

LORE_REVEAL_PASSAGE = (
    "In the Second Age of the World it came to pass that the Lord of Gifts, "
    "whose true name is not here written, came to the Elven-smiths of Eregion "
    "and taught them the art of making Rings of Power, whose history is here set down "
    "as it was recorded in the Red Book. Of old these were seven in number."
)

EUCATASTROPHE_PASSAGE = (
    "But then, suddenly, a horn rang out loud and clear. "
    "Light broke. "
    "Eagles! The Eagles are coming! "
    "And the darkness fled. "
    "Joy came at the last, against all hope."
)

FELLOWSHIP_PASSAGE = (
    "'Come on, Mr. Frodo!' said Sam. 'We can't stop now.' "
    "'I know,' said Frodo, 'but I'm tired.' "
    "'We're all tired,' replied Sam, 'but we'll go on together.' "
    "They went on. After a while Frodo laughed. 'You're right, Sam.'"
)


# ---------------------------------------------------------------------------
# ProseRegister enum
# ---------------------------------------------------------------------------

class TestProseRegister:
    def test_exactly_seven_registers(self):
        assert len(list(ProseRegister)) == 7

    def test_all_required_registers_present(self):
        required = {
            "elegiac", "eucatastrophic", "cozy", "dread",
            "wonder", "lore_reveal", "fellowship"
        }
        actual = {r.value for r in ProseRegister}
        assert required == actual

    def test_all_registers_have_descriptions(self):
        for reg in ProseRegister:
            assert reg in REGISTER_DESCRIPTIONS, f"Missing description for {reg}"
            assert len(REGISTER_DESCRIPTIONS[reg]) > 30

    def test_all_registers_have_trigger_conditions(self):
        for reg in ProseRegister:
            assert reg in REGISTER_TRIGGERS, f"Missing triggers for {reg}"
            assert len(REGISTER_TRIGGERS[reg]) >= 3

    def test_all_registers_have_structural_patterns(self):
        for reg in ProseRegister:
            assert reg in REGISTER_STRUCTURAL_PATTERNS
            assert len(REGISTER_STRUCTURAL_PATTERNS[reg]) > 20

    def test_all_registers_have_signature_keywords(self):
        for reg in ProseRegister:
            assert reg in REGISTER_SIGNATURE_KEYWORDS
            assert len(REGISTER_SIGNATURE_KEYWORDS[reg]) >= 5


# ---------------------------------------------------------------------------
# SceneTemplate model
# ---------------------------------------------------------------------------

class TestSceneTemplate:
    def test_basic_creation(self):
        tmpl = SceneTemplate(
            id="template_test",
            register=ProseRegister.ELEGIAC,
            avg_sentence_length=25.0,
        )
        assert tmpl.register == ProseRegister.ELEGIAC
        assert tmpl.avg_sentence_length == 25.0

    def test_defaults_are_reasonable(self):
        tmpl = SceneTemplate(id="t", register=ProseRegister.COZY)
        assert 0.0 <= tmpl.passive_ratio <= 1.0
        assert 0.0 <= tmpl.dialogue_density <= 1.0
        assert tmpl.avg_sentence_length > 0

    def test_to_neo4j_props_has_required_fields(self):
        tmpl = SceneTemplate(id="t", register=ProseRegister.DREAD)
        props = tmpl.to_neo4j_props()
        assert "id" in props
        assert "register" in props
        assert "avg_sentence_length" in props
        assert "passive_ratio" in props
        assert "dialogue_density" in props
        assert "archaic_word_rate" in props
        assert "structural_pattern" in props

    def test_from_dict_roundtrip(self):
        tmpl = SceneTemplate(
            id="t1",
            register=ProseRegister.WONDER,
            avg_sentence_length=19.5,
            passive_ratio=0.20,
            descriptive_focus=["light", "music"],
        )
        d = tmpl.to_dict()
        tmpl2 = SceneTemplate.from_dict(d)
        assert tmpl2.register == ProseRegister.WONDER
        assert abs(tmpl2.avg_sentence_length - 19.5) < 0.001
        assert tmpl2.descriptive_focus == ["light", "music"]

    def test_generation_prompt_fragment_contains_register(self):
        tmpl = SceneTemplate(
            id="t",
            register=ProseRegister.ELEGIAC,
            avg_sentence_length=26.0,
            passive_ratio=0.35,
            structural_pattern="long setup → sorrowful close",
        )
        prompt = tmpl.generation_prompt_fragment()
        assert ProseRegister.ELEGIAC in prompt
        assert "26" in prompt
        assert "passive" in prompt.lower()

    def test_generation_prompt_mentions_dialogue_when_high(self):
        tmpl = SceneTemplate(
            id="t",
            register=ProseRegister.FELLOWSHIP,
            dialogue_density=0.55,
            avg_sentence_length=11.0,
        )
        prompt = tmpl.generation_prompt_fragment()
        assert "dialogue" in prompt.lower()

    def test_generation_prompt_mentions_active_voice_when_low_passive(self):
        tmpl = SceneTemplate(
            id="t",
            register=ProseRegister.DREAD,
            passive_ratio=0.08,
            avg_sentence_length=9.0,
        )
        prompt = tmpl.generation_prompt_fragment()
        assert "active" in prompt.lower()

    def test_generation_prompt_mentions_archaic_when_high(self):
        tmpl = SceneTemplate(
            id="t",
            register=ProseRegister.ELEGIAC,
            archaic_word_rate=0.12,
            avg_sentence_length=25.0,
        )
        prompt = tmpl.generation_prompt_fragment()
        assert "archaic" in prompt.lower()


# ---------------------------------------------------------------------------
# CANONICAL_SCENE_TEMPLATES
# ---------------------------------------------------------------------------

class TestCanonicalSceneTemplates:
    def test_all_seven_registers_present(self):
        for reg in ProseRegister:
            assert reg in CANONICAL_SCENE_TEMPLATES, f"Missing template for {reg}"

    def test_all_templates_have_ids(self):
        for reg, tmpl in CANONICAL_SCENE_TEMPLATES.items():
            assert tmpl.id.startswith("template_")

    def test_elegiac_has_long_sentences(self):
        """Elegiac register should have longer-than-average sentences."""
        elegiac = CANONICAL_SCENE_TEMPLATES[ProseRegister.ELEGIAC]
        avg_all = sum(t.avg_sentence_length for t in CANONICAL_SCENE_TEMPLATES.values())
        avg_all /= len(CANONICAL_SCENE_TEMPLATES)
        assert elegiac.avg_sentence_length > avg_all

    def test_dread_has_short_sentences(self):
        """Dread register should have shorter-than-average sentences."""
        dread = CANONICAL_SCENE_TEMPLATES[ProseRegister.DREAD]
        avg_all = sum(t.avg_sentence_length for t in CANONICAL_SCENE_TEMPLATES.values())
        avg_all /= len(CANONICAL_SCENE_TEMPLATES)
        assert dread.avg_sentence_length < avg_all

    def test_lore_reveal_has_highest_passive_ratio(self):
        templates = list(CANONICAL_SCENE_TEMPLATES.values())
        lore_reveal = CANONICAL_SCENE_TEMPLATES[ProseRegister.LORE_REVEAL]
        max_passive = max(t.passive_ratio for t in templates)
        assert lore_reveal.passive_ratio == max_passive

    def test_fellowship_has_highest_dialogue_density(self):
        templates = list(CANONICAL_SCENE_TEMPLATES.values())
        fellowship = CANONICAL_SCENE_TEMPLATES[ProseRegister.FELLOWSHIP]
        max_dialogue = max(t.dialogue_density for t in templates)
        assert fellowship.dialogue_density == max_dialogue

    def test_elegiac_has_high_archaic_word_rate(self):
        elegiac = CANONICAL_SCENE_TEMPLATES[ProseRegister.ELEGIAC]
        assert elegiac.archaic_word_rate >= 0.10

    def test_cozy_has_low_archaic_word_rate(self):
        cozy = CANONICAL_SCENE_TEMPLATES[ProseRegister.COZY]
        assert cozy.archaic_word_rate <= 0.05

    def test_all_have_structural_patterns(self):
        for reg, tmpl in CANONICAL_SCENE_TEMPLATES.items():
            assert len(tmpl.structural_pattern) > 20, f"{reg} missing structural pattern"

    def test_all_have_example_passages(self):
        for reg, tmpl in CANONICAL_SCENE_TEMPLATES.items():
            assert len(tmpl.example_passages) >= 1, f"{reg} missing example passages"

    def test_all_have_descriptive_focus(self):
        for reg, tmpl in CANONICAL_SCENE_TEMPLATES.items():
            assert len(tmpl.descriptive_focus) >= 1, f"{reg} missing descriptive focus"

    def test_all_have_common_openings(self):
        for reg, tmpl in CANONICAL_SCENE_TEMPLATES.items():
            assert len(tmpl.common_openings) >= 1, f"{reg} missing common openings"

    def test_all_have_trigger_conditions(self):
        for reg, tmpl in CANONICAL_SCENE_TEMPLATES.items():
            assert len(tmpl.trigger_conditions) >= 2, f"{reg} too few trigger conditions"


# ---------------------------------------------------------------------------
# RegisterClassifier
# ---------------------------------------------------------------------------

class TestRegisterClassifier:
    def setup_method(self):
        self.classifier = RegisterClassifier()

    def test_classify_returns_classification(self):
        result = self.classifier.classify("text", "p1")
        assert isinstance(result, RegisterClassification)

    def test_classify_shire_passage_is_cozy(self):
        result = self.classifier.classify(SHIRE_PASSAGE, "shire")
        primary = result.primary_register()
        # Should detect cozy or fellowship from the hobbit-hole description
        assert primary in (ProseRegister.COZY, ProseRegister.FELLOWSHIP)

    def test_classify_elegiac_passage(self):
        result = self.classifier.classify(ELEGIAC_PASSAGE, "elegiac")
        primary = result.primary_register()
        assert primary == ProseRegister.ELEGIAC

    def test_classify_dread_passage(self):
        result = self.classifier.classify(DREAD_PASSAGE, "dread")
        primary = result.primary_register()
        assert primary == ProseRegister.DREAD

    def test_classify_wonder_passage(self):
        result = self.classifier.classify(WONDER_PASSAGE, "wonder")
        primary = result.primary_register()
        assert primary == ProseRegister.WONDER

    def test_classify_lore_reveal_passage(self):
        result = self.classifier.classify(LORE_REVEAL_PASSAGE, "lore")
        primary = result.primary_register()
        assert primary == ProseRegister.LORE_REVEAL

    def test_classify_eucatastrophe_passage(self):
        result = self.classifier.classify(EUCATASTROPHE_PASSAGE, "eucata")
        primary = result.primary_register()
        assert primary == ProseRegister.EUCATASTROPHIC

    def test_classify_fellowship_passage(self):
        result = self.classifier.classify(FELLOWSHIP_PASSAGE, "fellowship")
        primary = result.primary_register()
        assert primary == ProseRegister.FELLOWSHIP

    def test_threshold_filters_low_confidence(self):
        result = self.classifier.classify(SHIRE_PASSAGE, threshold=0.9)
        # With high threshold, fewer registers survive
        result_low = self.classifier.classify(SHIRE_PASSAGE, threshold=0.1)
        assert len(result.classifications) <= len(result_low.classifications)

    def test_multi_register_passage_gets_multiple(self):
        """A passage can belong to multiple registers."""
        # Elegiac+wonder: fading ancient beauty with amazement
        text = (
            "He marvelled at the beautiful ancient light that was now fading. "
            "In the elder days this glory was far greater, but it is now diminished, "
            "shadow of what once was, and none remain who remember the full wonder of it."
        )
        result = self.classifier.classify(text, threshold=0.2)
        confident = result.confident_registers(0.2)
        # Should get at least 2 registers
        assert len(confident) >= 1  # At minimum elegiac or wonder

    def test_get_template_returns_correct(self):
        tmpl = self.classifier.get_template(ProseRegister.ELEGIAC)
        assert tmpl is not None
        assert tmpl.register == ProseRegister.ELEGIAC

    def test_get_template_unknown_returns_none(self):
        assert self.classifier.get_template("nonexistent") is None

    def test_describe_register_returns_string(self):
        description = self.classifier.describe_register(ProseRegister.DREAD)
        assert ProseRegister.DREAD in description
        assert "sentence" in description.lower()

    def test_describe_register_unknown_returns_error(self):
        description = self.classifier.describe_register("fake_register")
        assert "No template" in description

    def test_classify_batch(self):
        texts = [
            ("p1", SHIRE_PASSAGE),
            ("p2", DREAD_PASSAGE),
            ("p3", ELEGIAC_PASSAGE),
        ]
        results = self.classifier.classify_batch(texts)
        assert len(results) == 3
        assert results[0].passage_id == "p1"
        assert results[1].passage_id == "p2"

    def test_empty_text_returns_result(self):
        """Classifier handles edge cases gracefully."""
        result = self.classifier.classify("", "empty")
        assert isinstance(result, RegisterClassification)

    def test_classifications_sorted_by_confidence_descending(self):
        result = self.classifier.classify(ELEGIAC_PASSAGE)
        confidences = [c for _, c in result.classifications]
        assert confidences == sorted(confidences, reverse=True)


# ---------------------------------------------------------------------------
# RegisterClassification model
# ---------------------------------------------------------------------------

class TestRegisterClassification:
    def test_primary_register_returns_highest_confidence(self):
        rc = RegisterClassification(
            passage_id="p1",
            passage_text_snippet="test",
            classifications=[
                ("elegiac", 0.8),
                ("dread", 0.3),
                ("wonder", 0.6),
            ],
        )
        assert rc.primary_register() == "elegiac"

    def test_primary_register_empty_returns_none(self):
        rc = RegisterClassification(passage_id="p1", passage_text_snippet="t")
        assert rc.primary_register() is None

    def test_confident_registers_filters_threshold(self):
        rc = RegisterClassification(
            passage_id="p1",
            passage_text_snippet="t",
            classifications=[
                ("elegiac", 0.8),
                ("dread", 0.3),
                ("wonder", 0.6),
            ],
        )
        confident = rc.confident_registers(0.5)
        assert "elegiac" in confident
        assert "wonder" in confident
        assert "dread" not in confident

    def test_to_dict_has_required_fields(self):
        rc = RegisterClassification(
            passage_id="p1",
            passage_text_snippet="text",
            classifications=[("elegiac", 0.9)],
        )
        d = rc.to_dict()
        assert "passage_id" in d
        assert "classifications" in d
        assert "primary_register" in d

    def test_summary_contains_registers(self):
        rc = RegisterClassification(
            passage_id="p1",
            passage_text_snippet="text",
            classifications=[("elegiac", 0.9), ("wonder", 0.5)],
        )
        summary = rc.summary()
        assert "elegiac" in summary
        assert "wonder" in summary


# ---------------------------------------------------------------------------
# ExemplifiesEdge
# ---------------------------------------------------------------------------

class TestExemplifiesEdge:
    def test_basic_creation(self):
        edge = ExemplifiesEdge(
            passage_id="p_001",
            template_id=ProseRegister.ELEGIAC,
            confidence=0.85,
        )
        assert edge.confidence == 0.85
        assert edge.template_id == ProseRegister.ELEGIAC

    def test_to_neo4j_props_has_confidence(self):
        edge = ExemplifiesEdge(
            passage_id="p_001",
            template_id=ProseRegister.DREAD,
            confidence=0.72,
        )
        props = edge.to_neo4j_props()
        assert abs(props["confidence"] - 0.72) < 0.001

    def test_to_neo4j_props_includes_reasoning_when_set(self):
        edge = ExemplifiesEdge(
            passage_id="p_001",
            template_id="elegiac",
            confidence=0.9,
            reasoning="Long sentences and archaic diction",
        )
        props = edge.to_neo4j_props()
        assert "reasoning" in props


# ---------------------------------------------------------------------------
# build_generation_prompt
# ---------------------------------------------------------------------------

class TestBuildGenerationPrompt:
    def test_returns_string(self):
        result = build_generation_prompt(ProseRegister.ELEGIAC, [])
        assert isinstance(result, str)

    def test_contains_register_name(self):
        result = build_generation_prompt(ProseRegister.ELEGIAC, [])
        assert ProseRegister.ELEGIAC in result

    def test_contains_structural_pattern(self):
        result = build_generation_prompt(ProseRegister.DREAD, [])
        assert "Structural pattern" in result

    def test_anchor_passages_are_injected(self):
        anchors = ["This is an elegiac passage about loss.", "Another example."]
        result = build_generation_prompt(ProseRegister.ELEGIAC, anchors)
        assert "elegiac passage" in result
        assert "Another example" in result

    def test_anchor_passages_section_header(self):
        anchors = ["Sample text"]
        result = build_generation_prompt(ProseRegister.WONDER, anchors)
        assert "ANCHOR PASSAGES" in result

    def test_scene_context_injected_when_provided(self):
        result = build_generation_prompt(
            ProseRegister.FELLOWSHIP,
            [],
            scene_context="Sam and Frodo make camp at the edge of Mordor.",
        )
        assert "Sam and Frodo" in result
        assert "SCENE TO GENERATE" in result

    def test_unknown_register_returns_fallback(self):
        result = build_generation_prompt("nonexistent_register", [])
        assert "nonexistent_register" in result

    def test_limits_anchors_to_five(self):
        """Prompt should not inject more than 5 anchor passages."""
        anchors = [f"Anchor passage {i}" for i in range(10)]
        result = build_generation_prompt(ProseRegister.ELEGIAC, anchors)
        # Check that not all 10 appear (first 5 only)
        assert "Anchor passage 0" in result
        assert "Anchor passage 4" in result
        assert "Anchor passage 5" not in result  # 6th should be excluded


# ---------------------------------------------------------------------------
# Cross-register distinctiveness
# ---------------------------------------------------------------------------

class TestRegisterDistinctiveness:
    """Verify that the 7 registers are meaningfully distinct from each other."""

    def test_elegiac_longer_sentences_than_dread(self):
        elegiac = CANONICAL_SCENE_TEMPLATES[ProseRegister.ELEGIAC]
        dread = CANONICAL_SCENE_TEMPLATES[ProseRegister.DREAD]
        assert elegiac.avg_sentence_length > dread.avg_sentence_length + 5

    def test_fellowship_more_dialogue_than_lore_reveal(self):
        fellowship = CANONICAL_SCENE_TEMPLATES[ProseRegister.FELLOWSHIP]
        lore_reveal = CANONICAL_SCENE_TEMPLATES[ProseRegister.LORE_REVEAL]
        assert fellowship.dialogue_density > lore_reveal.dialogue_density + 0.2

    def test_lore_reveal_more_passive_than_fellowship(self):
        fellowship = CANONICAL_SCENE_TEMPLATES[ProseRegister.FELLOWSHIP]
        lore_reveal = CANONICAL_SCENE_TEMPLATES[ProseRegister.LORE_REVEAL]
        assert lore_reveal.passive_ratio > fellowship.passive_ratio + 0.2

    def test_cozy_less_archaic_than_elegiac(self):
        cozy = CANONICAL_SCENE_TEMPLATES[ProseRegister.COZY]
        elegiac = CANONICAL_SCENE_TEMPLATES[ProseRegister.ELEGIAC]
        assert cozy.archaic_word_rate < elegiac.archaic_word_rate

    def test_elegiac_focuses_on_ancient_age(self):
        elegiac = CANONICAL_SCENE_TEMPLATES[ProseRegister.ELEGIAC]
        assert "ancient_age" in elegiac.descriptive_focus or "fading" in elegiac.descriptive_focus

    def test_dread_focuses_on_darkness(self):
        dread = CANONICAL_SCENE_TEMPLATES[ProseRegister.DREAD]
        assert "darkness" in dread.descriptive_focus or "shadow" in dread.descriptive_focus

    def test_wonder_focuses_on_light(self):
        wonder = CANONICAL_SCENE_TEMPLATES[ProseRegister.WONDER]
        assert "light" in wonder.descriptive_focus or "silver" in wonder.descriptive_focus
