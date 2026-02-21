"""Tests for LoreRule as Executable Cypher system (Issue #6).

All tests run without Neo4j — covers:
  - LoreRule model
  - LoreValidationResult / LoreViolation
  - TOLKIEN_LORE_RULES taxonomy
  - LoreRuleRegistry
  - LoreRuleValidator (pure-Python path)
  - Text context extraction heuristics
  - WorldBibleRuleMapper
  - classify_hardness
"""

import pytest
from book_graph_analyzer.models.lore_rule import (
    LoreRule,
    LoreViolation,
    LoreValidationResult,
    RULE_CATEGORIES,
)
from book_graph_analyzer.lore.rules import (
    LoreRuleRegistry,
    LoreRuleValidator,
    WorldBibleRuleMapper,
    SceneContext,
    TOLKIEN_LORE_RULES,
    classify_hardness,
    _extract_context_from_text,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_rule(**overrides) -> LoreRule:
    defaults = dict(
        id="test_rule",
        statement="Characters cannot fly without aid",
        category="magic",
        hardness="HARD",
        confidence=1.0,
    )
    defaults.update(overrides)
    return LoreRule(**defaults)


def make_context(**overrides) -> SceneContext:
    defaults = dict(
        scene_id="scene_001",
        character_names=["Gandalf"],
        character_races={"Gandalf": "Maia"},
        place_names=["Rivendell"],
        object_names=[],
        event_types=[],
        story_era="Third Age",
    )
    defaults.update(overrides)
    return SceneContext(**defaults)


# ---------------------------------------------------------------------------
# LoreRule model
# ---------------------------------------------------------------------------

class TestLoreRule:
    def test_basic_creation(self):
        rule = make_rule()
        assert rule.id == "test_rule"
        assert rule.hardness == "HARD"
        assert rule.is_hard is True
        assert rule.is_soft is False

    def test_soft_rule(self):
        rule = make_rule(hardness="SOFT")
        assert rule.is_soft is True
        assert rule.is_hard is False

    def test_optional_fields_default_none(self):
        rule = make_rule()
        assert rule.scope_entity_type is None
        assert rule.scope_era is None
        assert rule.cypher_check is None

    def test_source_passage_ids_default_empty(self):
        rule = make_rule()
        assert rule.source_passage_ids == []

    def test_to_neo4j_props_includes_required_fields(self):
        rule = make_rule(cypher_check="RETURN 'violation'")
        props = rule.to_neo4j_props()
        assert props["id"] == "test_rule"
        assert props["statement"] == "Characters cannot fly without aid"
        assert props["hardness"] == "HARD"
        assert props["category"] == "magic"
        assert props["cypher_check"] == "RETURN 'violation'"

    def test_to_neo4j_props_omits_none_optional(self):
        rule = make_rule()
        props = rule.to_neo4j_props()
        assert "scope_entity_type" not in props
        assert "scope_era" not in props
        assert "cypher_check" not in props

    def test_to_neo4j_props_includes_scope_when_set(self):
        rule = make_rule(scope_entity_type="Elf", scope_era="First Age")
        props = rule.to_neo4j_props()
        assert props["scope_entity_type"] == "Elf"
        assert props["scope_era"] == "First Age"

    def test_from_dict_roundtrip(self):
        rule = make_rule(scope_entity_type="Elf", cypher_check="RETURN 'x'")
        d = rule.to_dict()
        rule2 = LoreRule.from_dict(d)
        assert rule2.id == rule.id
        assert rule2.statement == rule.statement
        assert rule2.hardness == rule.hardness
        assert rule2.scope_entity_type == "Elf"
        assert rule2.cypher_check == "RETURN 'x'"

    def test_from_dict_defaults_on_missing_keys(self):
        rule = LoreRule.from_dict({"id": "x", "statement": "s"})
        assert rule.category == "metaphysics"
        assert rule.hardness == "SOFT"
        assert rule.confidence == 1.0

    def test_confidence_bounds(self):
        rule = make_rule(confidence=0.85)
        assert 0.0 <= rule.confidence <= 1.0


# ---------------------------------------------------------------------------
# LoreViolation
# ---------------------------------------------------------------------------

class TestLoreViolation:
    def test_hard_violation_is_blocking(self):
        v = LoreViolation(
            rule_id="x",
            rule_statement="test",
            hardness="HARD",
            description="violation detected",
            blocking=True,
        )
        assert v.blocking is True

    def test_soft_violation_is_not_blocking(self):
        v = LoreViolation(
            rule_id="x",
            rule_statement="test",
            hardness="SOFT",
            description="warning",
            blocking=False,
        )
        assert v.blocking is False

    def test_str_has_hard_tag(self):
        v = LoreViolation("r1", "rule", "HARD", "desc", True)
        s = str(v)
        assert "[HARD]" in s

    def test_str_has_soft_tag(self):
        v = LoreViolation("r1", "rule", "SOFT", "desc", False)
        s = str(v)
        assert "[SOFT]" in s


# ---------------------------------------------------------------------------
# LoreValidationResult
# ---------------------------------------------------------------------------

class TestLoreValidationResult:
    def test_passed_when_no_hard_violations(self):
        result = LoreValidationResult(scene_id="s1", passed=True, rules_checked=5)
        assert result.passed is True
        assert result.has_hard_violations is False

    def test_failed_when_hard_violations(self):
        v = LoreViolation("r1", "rule", "HARD", "desc", True)
        result = LoreValidationResult(
            scene_id="s1", passed=False, hard_violations=[v], rules_checked=5
        )
        assert result.passed is False
        assert result.has_hard_violations is True

    def test_soft_warnings_dont_affect_passed(self):
        v = LoreViolation("r1", "rule", "SOFT", "desc", False)
        result = LoreValidationResult(
            scene_id="s1", passed=True, soft_warnings=[v], rules_checked=5
        )
        assert result.passed is True
        assert result.has_soft_warnings is True

    def test_all_violations_combines_hard_and_soft(self):
        hard = LoreViolation("r1", "h", "HARD", "h-desc", True)
        soft = LoreViolation("r2", "s", "SOFT", "s-desc", False)
        result = LoreValidationResult(
            scene_id="s1", passed=False,
            hard_violations=[hard], soft_warnings=[soft], rules_checked=5
        )
        assert len(result.all_violations()) == 2

    def test_summary_contains_pass_when_passing(self):
        result = LoreValidationResult(scene_id="test_scene", passed=True, rules_checked=10)
        summary = result.summary()
        assert "PASS" in summary
        assert "test_scene" in summary

    def test_summary_contains_fail_when_failing(self):
        v = LoreViolation("r1", "rule", "HARD", "violation message", True)
        result = LoreValidationResult(
            scene_id="test_scene", passed=False, hard_violations=[v], rules_checked=5
        )
        summary = result.summary()
        assert "FAIL" in summary
        assert "violation message" in summary

    def test_summary_contains_soft_warnings(self):
        v = LoreViolation("r1", "rule", "SOFT", "soft warning", False)
        result = LoreValidationResult(
            scene_id="s1", passed=True, soft_warnings=[v], rules_checked=3
        )
        summary = result.summary()
        assert "soft warning" in summary

    def test_to_dict_has_required_fields(self):
        result = LoreValidationResult(scene_id="s1", passed=True, rules_checked=5)
        d = result.to_dict()
        assert "scene_id" in d
        assert "passed" in d
        assert "rules_checked" in d
        assert "hard_violations" in d
        assert "soft_warnings" in d


# ---------------------------------------------------------------------------
# TOLKIEN_LORE_RULES taxonomy
# ---------------------------------------------------------------------------

class TestTolkienLoreRules:
    def test_at_least_twenty_rules(self):
        assert len(TOLKIEN_LORE_RULES) >= 20, "Should have at least 20 pre-defined rules"

    def test_all_categories_covered(self):
        categories = {r.category for r in TOLKIEN_LORE_RULES}
        expected = {"race", "magic", "cosmology", "geography", "politics", "metaphysics", "objects", "history"}
        assert expected.issubset(categories), f"Missing categories: {expected - categories}"

    def test_both_hard_and_soft_present(self):
        hard = [r for r in TOLKIEN_LORE_RULES if r.is_hard]
        soft = [r for r in TOLKIEN_LORE_RULES if r.is_soft]
        assert len(hard) >= 5, "Need at least 5 HARD rules"
        assert len(soft) >= 5, "Need at least 5 SOFT rules"

    def test_elf_immortal_rule_exists(self):
        ids = {r.id for r in TOLKIEN_LORE_RULES}
        assert "race_elf_immortal" in ids

    def test_ring_corruption_rule_is_hard(self):
        rule = next(r for r in TOLKIEN_LORE_RULES if r.id == "magic_ring_corruption")
        assert rule.is_hard

    def test_ring_destruction_rule_is_hard(self):
        rule = next(r for r in TOLKIEN_LORE_RULES if r.id == "magic_ring_destruction")
        assert rule.is_hard

    def test_hobbit_unadventurous_is_soft(self):
        rule = next(r for r in TOLKIEN_LORE_RULES if r.id == "race_hobbit_unadventurous")
        assert rule.is_soft

    def test_all_rules_have_ids(self):
        for rule in TOLKIEN_LORE_RULES:
            assert rule.id, f"Rule missing ID"

    def test_all_rules_have_statements(self):
        for rule in TOLKIEN_LORE_RULES:
            assert rule.statement, f"Rule {rule.id} missing statement"

    def test_all_rule_categories_valid(self):
        valid_cats = RULE_CATEGORIES | {"history"}  # history added
        for rule in TOLKIEN_LORE_RULES:
            assert rule.category in valid_cats, \
                f"Rule {rule.id} has invalid category: {rule.category}"

    def test_cypher_check_present_for_hard_rules(self):
        """Most HARD rules should have a Cypher check defined."""
        hard_with_cypher = [r for r in TOLKIEN_LORE_RULES if r.is_hard and r.cypher_check]
        hard_total = [r for r in TOLKIEN_LORE_RULES if r.is_hard]
        # At least 50% of hard rules should have cypher_check
        ratio = len(hard_with_cypher) / len(hard_total)
        assert ratio >= 0.5, f"Only {ratio:.0%} of HARD rules have cypher_check"


# ---------------------------------------------------------------------------
# LoreRuleRegistry
# ---------------------------------------------------------------------------

class TestLoreRuleRegistry:
    def test_empty_registry(self):
        reg = LoreRuleRegistry()
        assert len(reg) == 0

    def test_add_rule(self):
        reg = LoreRuleRegistry()
        reg.add(make_rule())
        assert len(reg) == 1

    def test_get_existing_rule(self):
        reg = LoreRuleRegistry()
        rule = make_rule(id="my_rule")
        reg.add(rule)
        assert reg.get("my_rule") is rule

    def test_get_missing_rule_returns_none(self):
        reg = LoreRuleRegistry()
        assert reg.get("nonexistent") is None

    def test_add_many(self):
        reg = LoreRuleRegistry()
        rules = [make_rule(id=f"r{i}") for i in range(5)]
        reg.add_many(rules)
        assert len(reg) == 5

    def test_all_returns_all_rules(self):
        reg = LoreRuleRegistry()
        rules = [make_rule(id=f"r{i}") for i in range(3)]
        reg.add_many(rules)
        assert len(reg.all()) == 3

    def test_hard_rules_filter(self):
        reg = LoreRuleRegistry()
        reg.add(make_rule(id="hard", hardness="HARD"))
        reg.add(make_rule(id="soft", hardness="SOFT"))
        assert len(reg.hard_rules()) == 1
        assert reg.hard_rules()[0].id == "hard"

    def test_soft_rules_filter(self):
        reg = LoreRuleRegistry()
        reg.add(make_rule(id="hard", hardness="HARD"))
        reg.add(make_rule(id="soft", hardness="SOFT"))
        assert len(reg.soft_rules()) == 1

    def test_by_category_filter(self):
        reg = LoreRuleRegistry()
        reg.add(make_rule(id="r1", category="magic"))
        reg.add(make_rule(id="r2", category="race"))
        assert len(reg.by_category("magic")) == 1
        assert reg.by_category("magic")[0].id == "r1"

    def test_by_entity_type_includes_universal(self):
        reg = LoreRuleRegistry()
        reg.add(make_rule(id="universal"))            # scope_entity_type=None
        reg.add(make_rule(id="elf_rule", scope_entity_type="Elf"))
        elf_rules = reg.by_entity_type("Elf")
        ids = {r.id for r in elf_rules}
        assert "universal" in ids
        assert "elf_rule" in ids

    def test_by_entity_type_excludes_wrong_scope(self):
        reg = LoreRuleRegistry()
        reg.add(make_rule(id="dwarf_rule", scope_entity_type="Dwarf"))
        elf_rules = reg.by_entity_type("Elf")
        ids = {r.id for r in elf_rules}
        assert "dwarf_rule" not in ids

    def test_from_tolkien_defaults(self):
        reg = LoreRuleRegistry.from_tolkien_defaults()
        assert len(reg) >= 20
        assert reg.get("race_elf_immortal") is not None
        assert reg.get("magic_ring_corruption") is not None


# ---------------------------------------------------------------------------
# classify_hardness
# ---------------------------------------------------------------------------

class TestClassifyHardness:
    def test_cannot_is_hard(self):
        assert classify_hardness("Elves cannot die of age", "") == "HARD"

    def test_never_is_hard(self):
        assert classify_hardness("This never happens", "") == "HARD"

    def test_can_only_is_hard(self):
        assert classify_hardness("Ring can only be destroyed in Mount Doom", "") == "HARD"

    def test_must_is_hard(self):
        assert classify_hardness("Characters must age", "") == "HARD"

    def test_cultural_norm_is_soft(self):
        assert classify_hardness("Hobbits prefer to stay at home", "") == "SOFT"

    def test_description_also_checked(self):
        assert classify_hardness("Magic", "This is impossible to circumvent") == "HARD"


# ---------------------------------------------------------------------------
# Context extraction from text
# ---------------------------------------------------------------------------

class TestExtractContextFromText:
    def test_detects_elf_race(self):
        text = "Legolas the Elf walked through the forest."
        ctx = _extract_context_from_text(text, "s1", "Third Age")
        assert "Elf" in ctx.character_races.values()

    def test_detects_place_mount_doom(self):
        text = "They journeyed to Mount Doom in the land of Mordor."
        ctx = _extract_context_from_text(text, "s1", None)
        assert "Mount Doom" in ctx.place_names

    def test_detects_numenor(self):
        text = "The ship sailed to Númenor across the sea."
        ctx = _extract_context_from_text(text, "s1", None)
        assert any("menor" in p.lower() for p in ctx.place_names)

    def test_detects_one_ring(self):
        text = "Frodo wore the One Ring and disappeared."
        ctx = _extract_context_from_text(text, "s1", None)
        assert any("ring" in o.lower() for o in ctx.object_names)

    def test_detects_death_event(self):
        text = "Boromir was slain by the Uruk-hai."
        ctx = _extract_context_from_text(text, "s1", None)
        assert "death" in ctx.event_types

    def test_detects_destruction_event(self):
        text = "The Ring was destroyed in the flames."
        ctx = _extract_context_from_text(text, "s1", None)
        assert "destruction" in ctx.event_types

    def test_detects_hobbit_race(self):
        text = "Frodo the Hobbit carried the burden."
        ctx = _extract_context_from_text(text, "s1", None)
        assert "Hobbit" in ctx.character_races.values()

    def test_story_era_preserved(self):
        text = "Gandalf spoke."
        ctx = _extract_context_from_text(text, "s1", "Second Age")
        assert ctx.story_era == "Second Age"


# ---------------------------------------------------------------------------
# LoreRuleValidator — pure-Python validation
# ---------------------------------------------------------------------------

class TestLoreRuleValidator:
    def setup_method(self):
        self.registry = LoreRuleRegistry.from_tolkien_defaults()
        self.validator = LoreRuleValidator(self.registry)

    def test_clean_scene_passes(self):
        ctx = make_context()
        result = self.validator.validate_scene_context(ctx)
        assert isinstance(result, LoreValidationResult)
        assert result.passed is True
        assert result.rules_checked > 0

    def test_numenor_in_third_age_is_hard_violation(self):
        ctx = make_context(
            story_era="Third Age",
            place_names=["Númenor", "Gondor"],
        )
        result = self.validator.validate_scene_context(ctx)
        # hist_numenor_fallen should trigger
        violation_ids = {v.rule_id for v in result.hard_violations}
        assert "hist_numenor_fallen" in violation_ids

    def test_one_ring_destruction_outside_mount_doom_is_violation(self):
        ctx = make_context(
            object_names=["The One Ring"],
            event_types=["destruction"],
            place_names=["Rivendell"],
        )
        result = self.validator.validate_scene_context(ctx)
        violation_ids = {v.rule_id for v in result.hard_violations}
        assert "magic_ring_destruction" in violation_ids

    def test_one_ring_at_mount_doom_does_not_violate(self):
        ctx = make_context(
            object_names=["The One Ring"],
            event_types=["destruction"],
            place_names=["Mount Doom"],
        )
        result = self.validator.validate_scene_context(ctx)
        violation_ids = {v.rule_id for v in result.hard_violations}
        assert "magic_ring_destruction" not in violation_ids

    def test_validate_text_elf_death_by_age(self):
        text = "The Elf Legolas died of old age peacefully in his sleep."
        result = self.validator.validate_text(text, story_era="Third Age")
        # Should pick up death_by_age event and elf character
        violation_ids = {v.rule_id for v in result.hard_violations}
        assert "race_elf_immortal" in violation_ids

    def test_validate_text_clean_passage_passes(self):
        text = "Frodo sat by the fire and thought of the Shire."
        result = self.validator.validate_text(text, story_era="Third Age")
        assert result.passed is True
        assert result.rules_checked > 0

    def test_validate_text_ring_destruction_away_from_doom(self):
        text = "Bilbo destroyed the One Ring in the library of Rivendell."
        result = self.validator.validate_text(text)
        violation_ids = {v.rule_id for v in result.hard_violations}
        assert "magic_ring_destruction" in violation_ids

    def test_category_filter_limits_rules_checked(self):
        ctx = make_context()
        result_all = self.validator.validate_scene_context(ctx)
        result_magic = self.validator.validate_scene_context(ctx, categories=["magic"])
        assert result_magic.rules_checked < result_all.rules_checked

    def test_validation_result_has_scene_id(self):
        ctx = make_context(scene_id="scene_xyz")
        result = self.validator.validate_scene_context(ctx)
        assert result.scene_id == "scene_xyz"

    def test_hard_violations_are_blocking(self):
        ctx = make_context(
            story_era="Third Age",
            place_names=["Númenor"],
        )
        result = self.validator.validate_scene_context(ctx)
        for v in result.hard_violations:
            assert v.blocking is True

    def test_soft_warnings_are_not_blocking(self):
        ctx = make_context()
        # Manufacture a soft result by checking soft-only rules
        soft_registry = LoreRuleRegistry()
        soft_registry.add_many([r for r in self.registry.all() if r.is_soft])
        validator = LoreRuleValidator(soft_registry)
        result = validator.validate_scene_context(ctx)
        for v in result.soft_warnings:
            assert v.blocking is False


# ---------------------------------------------------------------------------
# WorldBibleRuleMapper
# ---------------------------------------------------------------------------

class TestWorldBibleRuleMapper:
    def test_classify_hard_from_cannot(self):
        assert classify_hardness("X cannot do Y", "") == "HARD"

    def test_classify_soft_from_description(self):
        assert classify_hardness("X usually does Y", "") == "SOFT"

    def test_mapper_creates_lore_rule(self):
        """Test that mapper works with a mock WorldRule object."""
        from book_graph_analyzer.lore.rules import WorldBibleRuleMapper
        from unittest.mock import MagicMock

        mapper = WorldBibleRuleMapper()

        # Create a mock WorldRule
        mock_rule = MagicMock()
        mock_rule.category.value = "magic"
        mock_rule.title = "Magic cannot be used without consequence"
        mock_rule.description = "Every use of magical power has a cost"
        mock_rule.confidence = 0.9

        result = mapper.map_rule(mock_rule)

        assert isinstance(result, LoreRule)
        assert result.hardness == "HARD"  # "cannot" triggers HARD
        assert result.category == "magic"
        assert result.confidence == 0.9

    def test_mapper_soft_rule(self):
        """Test soft rule classification."""
        from book_graph_analyzer.lore.rules import WorldBibleRuleMapper
        from unittest.mock import MagicMock

        mapper = WorldBibleRuleMapper()
        mock_rule = MagicMock()
        mock_rule.category.value = "culture"
        mock_rule.title = "Dwarves generally prefer their mountain halls"
        mock_rule.description = "This is a general cultural tendency"
        mock_rule.confidence = 0.8

        result = mapper.map_rule(mock_rule)
        assert result.hardness == "SOFT"

    def test_mapper_generates_valid_id(self):
        """Mapped rule ID should be a valid Python identifier fragment."""
        from book_graph_analyzer.lore.rules import WorldBibleRuleMapper
        from unittest.mock import MagicMock

        mapper = WorldBibleRuleMapper()
        mock_rule = MagicMock()
        mock_rule.category.value = "cosmology"
        mock_rule.title = "The world was made by music"
        mock_rule.description = "Cosmological fact"
        mock_rule.confidence = 1.0

        result = mapper.map_rule(mock_rule)
        assert result.id.startswith("wb_")
        assert " " not in result.id  # No spaces in ID
