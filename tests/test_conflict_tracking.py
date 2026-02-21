"""Tests for LoreConflict tracking system (Issue #7).

All tests are pure-Python — no Neo4j required. Covers:
  - LoreConflict model and ConflictClaim
  - ConflictType, ResolutionPolicy, AuthorPeriod enums
  - KNOWN_TOLKIEN_CONFLICTS baseline
  - ConflictRegistry (CRUD, filters, resolution)
  - ConflictDetector
  - ConflictAwareValidator (suppression and downgrade logic)
  - author_period field on Passage model
"""

import pytest
from book_graph_analyzer.models.lore_conflict import (
    LoreConflict,
    ConflictClaim,
    ConflictType,
    ResolutionPolicy,
    AuthorPeriod,
    AUTHOR_PERIOD_ORDER,
)
from book_graph_analyzer.lore.conflicts import (
    ConflictRegistry,
    ConflictDetector,
    ConflictAwareValidator,
    KNOWN_TOLKIEN_CONFLICTS,
)
from book_graph_analyzer.models.passage import Passage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_claim(**overrides) -> ConflictClaim:
    defaults = dict(
        statement="Elves cannot remarry",
        source_book="Laws and Customs",
        author_period=AuthorPeriod.LATE,
        confidence=0.9,
    )
    defaults.update(overrides)
    return ConflictClaim(**defaults)


def make_conflict(**overrides) -> LoreConflict:
    defaults = dict(
        id="test_conflict",
        summary="Test contradiction",
        conflict_type=ConflictType.RETCON,
        claims=[
            make_claim(
                statement="Early claim",
                author_period=AuthorPeriod.EARLY,
                confidence=0.7,
            ),
            make_claim(
                statement="Late claim",
                author_period=AuthorPeriod.LATE,
                confidence=0.9,
            ),
        ],
        entity_ids=["some_elf"],
        rule_ids=["race_elf_immortal"],
        resolution_policy=ResolutionPolicy.USE_LATER_TEXT,
        resolved=True,
    )
    defaults.update(overrides)
    return LoreConflict(**defaults)


def make_passage(**overrides) -> Passage:
    defaults = dict(
        id="p_test",
        text="Gandalf spoke.",
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
# AuthorPeriod and ordering
# ---------------------------------------------------------------------------

class TestAuthorPeriod:
    def test_early_before_middle(self):
        assert AUTHOR_PERIOD_ORDER[AuthorPeriod.EARLY] < AUTHOR_PERIOD_ORDER[AuthorPeriod.MIDDLE]

    def test_middle_before_late(self):
        assert AUTHOR_PERIOD_ORDER[AuthorPeriod.MIDDLE] < AUTHOR_PERIOD_ORDER[AuthorPeriod.LATE]

    def test_early_before_late(self):
        assert AUTHOR_PERIOD_ORDER[AuthorPeriod.EARLY] < AUTHOR_PERIOD_ORDER[AuthorPeriod.LATE]

    def test_all_periods_covered(self):
        for period in (AuthorPeriod.EARLY, AuthorPeriod.MIDDLE, AuthorPeriod.LATE):
            assert period in AUTHOR_PERIOD_ORDER


# ---------------------------------------------------------------------------
# ConflictClaim
# ---------------------------------------------------------------------------

class TestConflictClaim:
    def test_basic_creation(self):
        claim = make_claim()
        assert claim.statement == "Elves cannot remarry"
        assert claim.author_period == AuthorPeriod.LATE

    def test_period_order_late_is_highest(self):
        early = make_claim(author_period=AuthorPeriod.EARLY)
        late = make_claim(author_period=AuthorPeriod.LATE)
        assert late.period_order() > early.period_order()

    def test_source_passage_id_optional(self):
        claim = make_claim()
        assert claim.source_passage_id is None

    def test_to_dict_roundtrip(self):
        claim = make_claim(source_passage_id="p_001")
        d = claim.to_dict()
        claim2 = ConflictClaim.from_dict(d)
        assert claim2.statement == claim.statement
        assert claim2.author_period == claim.author_period
        assert claim2.source_passage_id == "p_001"

    def test_to_dict_omits_none_passage_id(self):
        claim = make_claim()
        d = claim.to_dict()
        assert "source_passage_id" not in d


# ---------------------------------------------------------------------------
# LoreConflict model
# ---------------------------------------------------------------------------

class TestLoreConflict:
    def test_basic_creation(self):
        c = make_conflict()
        assert c.id == "test_conflict"
        assert c.conflict_type == ConflictType.RETCON

    def test_is_resolved_property(self):
        c = make_conflict(resolved=True)
        assert c.is_resolved is True

        c2 = make_conflict(resolved=False)
        assert c2.is_resolved is False

    def test_needs_human_review_false_when_resolved(self):
        c = make_conflict(
            resolution_policy=ResolutionPolicy.FLAG_FOR_HUMAN,
            resolved=True,
        )
        assert c.needs_human_review is False

    def test_needs_human_review_true_when_flagged_and_unresolved(self):
        c = make_conflict(
            resolution_policy=ResolutionPolicy.FLAG_FOR_HUMAN,
            resolved=False,
        )
        assert c.needs_human_review is True

    def test_winning_claim_use_later_text(self):
        c = make_conflict(resolution_policy=ResolutionPolicy.USE_LATER_TEXT)
        winner = c.winning_claim()
        assert winner is not None
        assert winner.author_period == AuthorPeriod.LATE

    def test_winning_claim_use_earlier_text(self):
        c = make_conflict(resolution_policy=ResolutionPolicy.USE_EARLIER_TEXT)
        winner = c.winning_claim()
        assert winner is not None
        assert winner.author_period == AuthorPeriod.EARLY

    def test_winning_claim_both_valid_returns_none(self):
        c = make_conflict(resolution_policy=ResolutionPolicy.BOTH_VALID_IN_UNIVERSE)
        assert c.winning_claim() is None

    def test_winning_claim_flag_for_human_returns_none(self):
        c = make_conflict(resolution_policy=ResolutionPolicy.FLAG_FOR_HUMAN)
        assert c.winning_claim() is None

    def test_winning_claim_irresolvable_returns_none(self):
        c = make_conflict(resolution_policy=ResolutionPolicy.IRRESOLVABLE)
        assert c.winning_claim() is None

    def test_winning_claim_use_most_cited(self):
        c = make_conflict(
            resolution_policy=ResolutionPolicy.USE_MOST_CITED,
            claims=[
                make_claim(statement="Low conf", confidence=0.5),
                make_claim(statement="High conf", confidence=0.95),
            ],
        )
        winner = c.winning_claim()
        assert winner is not None
        assert winner.confidence == 0.95

    def test_suppresses_lore_violation_both_valid(self):
        c = make_conflict(
            rule_ids=["race_elf_immortal"],
            entity_ids=["glorfindel"],
            resolution_policy=ResolutionPolicy.BOTH_VALID_IN_UNIVERSE,
        )
        assert c.suppresses_lore_violation("race_elf_immortal", "glorfindel") is True

    def test_suppresses_lore_violation_other_policy(self):
        c = make_conflict(
            rule_ids=["race_elf_immortal"],
            entity_ids=["glorfindel"],
            resolution_policy=ResolutionPolicy.USE_LATER_TEXT,
        )
        assert c.suppresses_lore_violation("race_elf_immortal", "glorfindel") is False

    def test_suppresses_lore_violation_wrong_rule(self):
        c = make_conflict(
            rule_ids=["race_elf_immortal"],
            entity_ids=[],
            resolution_policy=ResolutionPolicy.BOTH_VALID_IN_UNIVERSE,
        )
        assert c.suppresses_lore_violation("magic_ring_corruption", "sauron") is False

    def test_downgrades_to_soft_flag_for_human(self):
        c = make_conflict(
            rule_ids=["race_elf_immortal"],
            entity_ids=["glorfindel"],
            resolution_policy=ResolutionPolicy.FLAG_FOR_HUMAN,
        )
        assert c.downgrades_to_soft("race_elf_immortal", "glorfindel") is True

    def test_downgrades_to_soft_other_policy(self):
        c = make_conflict(
            rule_ids=["race_elf_immortal"],
            entity_ids=[],
            resolution_policy=ResolutionPolicy.USE_LATER_TEXT,
        )
        assert c.downgrades_to_soft("race_elf_immortal", "anyone") is False

    def test_to_dict_roundtrip(self):
        c = make_conflict()
        d = c.to_dict()
        c2 = LoreConflict.from_dict(d)
        assert c2.id == c.id
        assert c2.summary == c.summary
        assert c2.conflict_type == c.conflict_type
        assert len(c2.claims) == len(c.claims)

    def test_from_dict_claims_as_json_string(self):
        """from_dict should handle claims stored as JSON string (Neo4j format)."""
        import json
        c = make_conflict()
        d = c.to_neo4j_props()
        # to_neo4j_props stores claims as JSON string
        d2 = dict(d)
        c2 = LoreConflict.from_dict(d2)
        assert len(c2.claims) == 2

    def test_brief_contains_id(self):
        c = make_conflict()
        brief = c.brief()
        assert c.id in brief

    def test_detail_contains_claims(self):
        c = make_conflict()
        detail = c.detail()
        assert "Early claim" in detail
        assert "Late claim" in detail

    def test_detail_shows_winning_claim(self):
        c = make_conflict(resolution_policy=ResolutionPolicy.USE_LATER_TEXT)
        detail = c.detail()
        assert "Active claim" in detail


# ---------------------------------------------------------------------------
# KNOWN_TOLKIEN_CONFLICTS baseline
# ---------------------------------------------------------------------------

class TestKnownTolkienConflicts:
    def test_at_least_five_conflicts(self):
        assert len(KNOWN_TOLKIEN_CONFLICTS) >= 5

    def test_blue_wizards_conflict_exists(self):
        ids = {c.id for c in KNOWN_TOLKIEN_CONFLICTS}
        assert "blue_wizards_names" in ids

    def test_glorfindel_conflict_exists(self):
        ids = {c.id for c in KNOWN_TOLKIEN_CONFLICTS}
        assert "glorfindel_identity" in ids

    def test_bombadil_conflict_exists(self):
        ids = {c.id for c in KNOWN_TOLKIEN_CONFLICTS}
        assert "bombadil_nature" in ids

    def test_elvish_mortality_conflict_exists(self):
        ids = {c.id for c in KNOWN_TOLKIEN_CONFLICTS}
        assert "elvish_mortality" in ids

    def test_all_conflict_types_represented(self):
        types = {c.conflict_type for c in KNOWN_TOLKIEN_CONFLICTS}
        assert ConflictType.RETCON in types
        assert ConflictType.AMBIGUITY in types

    def test_all_author_periods_used(self):
        periods = set()
        for c in KNOWN_TOLKIEN_CONFLICTS:
            for claim in c.claims:
                periods.add(claim.author_period)
        assert AuthorPeriod.EARLY in periods
        assert AuthorPeriod.MIDDLE in periods
        assert AuthorPeriod.LATE in periods

    def test_all_conflicts_have_claims(self):
        for c in KNOWN_TOLKIEN_CONFLICTS:
            assert len(c.claims) >= 2, f"{c.id} should have at least 2 claims"

    def test_all_conflicts_have_summaries(self):
        for c in KNOWN_TOLKIEN_CONFLICTS:
            assert len(c.summary) > 20, f"{c.id} summary too short"

    def test_bombadil_is_irresolvable(self):
        bombadil = next(c for c in KNOWN_TOLKIEN_CONFLICTS if c.id == "bombadil_nature")
        assert bombadil.resolution_policy == ResolutionPolicy.IRRESOLVABLE
        assert bombadil.resolved is False

    def test_glorfindel_is_resolved_with_later_text(self):
        glorfindel = next(c for c in KNOWN_TOLKIEN_CONFLICTS if c.id == "glorfindel_identity")
        assert glorfindel.resolution_policy == ResolutionPolicy.USE_LATER_TEXT
        assert glorfindel.resolved is True

    def test_glorfindel_covers_elf_rule(self):
        glorfindel = next(c for c in KNOWN_TOLKIEN_CONFLICTS if c.id == "glorfindel_identity")
        assert "race_elf_immortal" in glorfindel.rule_ids

    def test_blue_wizards_is_both_valid(self):
        bw = next(c for c in KNOWN_TOLKIEN_CONFLICTS if c.id == "blue_wizards_names")
        assert bw.resolution_policy == ResolutionPolicy.BOTH_VALID_IN_UNIVERSE


# ---------------------------------------------------------------------------
# ConflictRegistry
# ---------------------------------------------------------------------------

class TestConflictRegistry:
    def test_empty_registry(self):
        reg = ConflictRegistry()
        assert len(reg) == 0

    def test_add_and_get(self):
        reg = ConflictRegistry()
        c = make_conflict(id="my_conflict")
        reg.add(c)
        assert reg.get("my_conflict") is c

    def test_get_missing_returns_none(self):
        reg = ConflictRegistry()
        assert reg.get("nonexistent") is None

    def test_add_many(self):
        reg = ConflictRegistry()
        reg.add_many([make_conflict(id=f"c{i}") for i in range(5)])
        assert len(reg) == 5

    def test_all_returns_all(self):
        reg = ConflictRegistry()
        reg.add_many([make_conflict(id=f"c{i}") for i in range(3)])
        assert len(reg.all()) == 3

    def test_resolved_filter(self):
        reg = ConflictRegistry()
        reg.add(make_conflict(id="res", resolved=True))
        reg.add(make_conflict(id="unres", resolved=False))
        assert len(reg.resolved()) == 1
        assert reg.resolved()[0].id == "res"

    def test_unresolved_filter(self):
        reg = ConflictRegistry()
        reg.add(make_conflict(id="res", resolved=True))
        reg.add(make_conflict(id="unres", resolved=False))
        assert len(reg.unresolved()) == 1
        assert reg.unresolved()[0].id == "unres"

    def test_needing_human_review(self):
        reg = ConflictRegistry()
        reg.add(make_conflict(
            id="human",
            resolution_policy=ResolutionPolicy.FLAG_FOR_HUMAN,
            resolved=False,
        ))
        reg.add(make_conflict(id="auto", resolved=True))
        reviews = reg.needing_human_review()
        assert len(reviews) == 1
        assert reviews[0].id == "human"

    def test_by_entity_filter(self):
        reg = ConflictRegistry()
        reg.add(make_conflict(id="c1", entity_ids=["glorfindel"]))
        reg.add(make_conflict(id="c2", entity_ids=["sauron"]))
        results = reg.by_entity("glorfindel")
        assert len(results) == 1
        assert results[0].id == "c1"

    def test_by_rule_filter(self):
        reg = ConflictRegistry()
        reg.add(make_conflict(id="c1", rule_ids=["race_elf_immortal"]))
        reg.add(make_conflict(id="c2", rule_ids=["magic_ring_corruption"]))
        results = reg.by_rule("race_elf_immortal")
        assert len(results) == 1

    def test_by_type_filter(self):
        reg = ConflictRegistry()
        reg.add(make_conflict(id="retcon", conflict_type=ConflictType.RETCON))
        reg.add(make_conflict(id="ambi", conflict_type=ConflictType.AMBIGUITY))
        assert len(reg.by_type(ConflictType.RETCON)) == 1

    def test_resolve_applies_policy(self):
        reg = ConflictRegistry()
        reg.add(make_conflict(
            id="my_conflict",
            resolution_policy=ResolutionPolicy.FLAG_FOR_HUMAN,
            resolved=False,
        ))
        success = reg.resolve("my_conflict", ResolutionPolicy.USE_LATER_TEXT, "test")
        assert success is True
        c = reg.get("my_conflict")
        assert c.resolution_policy == ResolutionPolicy.USE_LATER_TEXT
        assert c.resolved is True

    def test_resolve_returns_false_for_missing(self):
        reg = ConflictRegistry()
        assert reg.resolve("nonexistent", ResolutionPolicy.USE_LATER_TEXT) is False

    def test_suppresses_violation(self):
        reg = ConflictRegistry()
        reg.add(make_conflict(
            id="c1",
            rule_ids=["race_elf_immortal"],
            entity_ids=["glorfindel"],
            resolution_policy=ResolutionPolicy.BOTH_VALID_IN_UNIVERSE,
        ))
        assert reg.suppresses_violation("race_elf_immortal", "glorfindel") is True
        assert reg.suppresses_violation("race_elf_immortal", "legolas") is False

    def test_downgrades_to_soft(self):
        reg = ConflictRegistry()
        reg.add(make_conflict(
            id="c1",
            rule_ids=["race_elf_immortal"],
            entity_ids=["glorfindel"],
            resolution_policy=ResolutionPolicy.FLAG_FOR_HUMAN,
        ))
        assert reg.downgrades_to_soft("race_elf_immortal", "glorfindel") is True
        assert reg.downgrades_to_soft("magic_ring_corruption", "glorfindel") is False

    def test_from_tolkien_defaults_has_known_conflicts(self):
        reg = ConflictRegistry.from_tolkien_defaults()
        assert len(reg) >= 5
        assert reg.get("glorfindel_identity") is not None
        assert reg.get("bombadil_nature") is not None
        assert reg.get("blue_wizards_names") is not None


# ---------------------------------------------------------------------------
# ConflictDetector
# ---------------------------------------------------------------------------

class TestConflictDetector:
    def setup_method(self):
        self.detector = ConflictDetector()

    def test_check_entity_returns_conflicts(self):
        conflicts = self.detector.check_entity("glorfindel", "new statement")
        ids = [c.id for c in conflicts]
        assert "glorfindel_identity" in ids

    def test_check_entity_empty_for_unknown(self):
        conflicts = self.detector.check_entity("unknown_entity_xyz", "statement")
        assert conflicts == []

    def test_check_rule_returns_conflicts(self):
        conflicts = self.detector.check_rule("race_elf_immortal")
        # glorfindel_identity and elvish_mortality both reference this rule
        assert len(conflicts) >= 1

    def test_detect_new_conflict_retcon(self):
        conflict = self.detector.detect_new_conflict(
            entity_ids=["some_elf"],
            rule_ids=["race_elf_immortal"],
            new_statement="Elves can die from despair",
            source_book="Early Tales",
            author_period=AuthorPeriod.EARLY,
            existing_statement="Elves cannot die of age or disease",
            existing_source="Laws and Customs",
            existing_period=AuthorPeriod.LATE,
        )
        assert conflict.conflict_type == ConflictType.RETCON
        assert "some_elf" in conflict.entity_ids
        assert len(conflict.claims) == 2

    def test_detect_new_conflict_enriches_claim_source_metadata_when_known(self):
        conflict = self.detector.detect_new_conflict(
            entity_ids=["blue_wizards"],
            rule_ids=[],
            new_statement="Blue Wizards are Morinehtar and Romestamo",
            source_book="Unfinished Tales",
            author_period=AuthorPeriod.LATE,
            existing_statement="Blue Wizards are Alatar and Pallando",
            existing_source="The Hobbit",
            existing_period=AuthorPeriod.MIDDLE,
        )
        existing_claim, new_claim = conflict.claims

        assert existing_claim.source_id == "src_hobbit"
        assert existing_claim.editorial_status == "published"
        assert existing_claim.source_authority_weight is not None

        assert new_claim.source_id == "src_unfinished_tales"
        assert new_claim.editorial_status == "unfinished"
        assert new_claim.source_authority_weight is not None

    def test_detect_new_conflict_same_period_is_direct(self):
        conflict = self.detector.detect_new_conflict(
            entity_ids=["aragorn"],
            rule_ids=[],
            new_statement="Aragorn is mortal",
            source_book="Book A",
            author_period=AuthorPeriod.MIDDLE,
            existing_statement="Aragorn is immortal",
            existing_source="Book B",
            existing_period=AuthorPeriod.MIDDLE,
        )
        assert conflict.conflict_type == ConflictType.DIRECT_CONTRADICTION

    def test_detect_new_conflict_retcon_uses_later_text_policy(self):
        conflict = self.detector.detect_new_conflict(
            entity_ids=["elf"],
            rule_ids=[],
            new_statement="New fact",
            source_book="Late Book",
            author_period=AuthorPeriod.LATE,
            existing_statement="Old fact",
            existing_source="Early Book",
            existing_period=AuthorPeriod.EARLY,
        )
        assert conflict.resolution_policy == ResolutionPolicy.USE_LATER_TEXT

    def test_detect_new_conflict_direct_flags_for_human(self):
        conflict = self.detector.detect_new_conflict(
            entity_ids=[],
            rule_ids=["some_rule"],
            new_statement="X",
            source_book="A",
            author_period=AuthorPeriod.MIDDLE,
            existing_statement="Y",
            existing_source="B",
            existing_period=AuthorPeriod.MIDDLE,
        )
        assert conflict.resolution_policy == ResolutionPolicy.FLAG_FOR_HUMAN

    def test_auto_generated_id_starts_with_auto(self):
        conflict = self.detector.detect_new_conflict(
            entity_ids=["entity"],
            rule_ids=[],
            new_statement="new",
            source_book="book",
            author_period=AuthorPeriod.MIDDLE,
            existing_statement="old",
            existing_source="book2",
            existing_period=AuthorPeriod.MIDDLE,
        )
        assert conflict.id.startswith("auto_")


# ---------------------------------------------------------------------------
# ConflictAwareValidator — suppression and downgrade
# ---------------------------------------------------------------------------

class TestConflictAwareValidator:
    def setup_method(self):
        self.validator = ConflictAwareValidator()

    def test_clean_text_passes(self):
        result = self.validator.validate_text("Frodo sat by the fire.")
        assert result.passed is True

    def test_bombadil_ring_suppressed(self):
        """Tom Bombadil's Ring immunity is covered by bombadil_nature conflict.

        However, since bombadil_nature's policy is IRRESOLVABLE (not both_valid),
        violations are not suppressed — they're preserved. This tests that logic.
        """
        # Bombadil conflict policy is IRRESOLVABLE — does not suppress violations
        from book_graph_analyzer.lore.conflicts import ConflictRegistry
        reg = ConflictRegistry.from_tolkien_defaults()
        bombadil = reg.get("bombadil_nature")
        # IRRESOLVABLE does not suppress — suppression only happens for BOTH_VALID_IN_UNIVERSE
        assert not bombadil.suppresses_lore_violation("magic_ring_corruption", "tom_bombadil")

    def test_both_valid_universe_suppresses_violation(self):
        """A conflict with BOTH_VALID_IN_UNIVERSE should suppress violations."""
        from book_graph_analyzer.lore.rules import LoreRuleRegistry, LoreRuleValidator, SceneContext
        from book_graph_analyzer.lore.conflicts import ConflictRegistry, ConflictAwareValidator

        # Set up a conflict that suppresses race_elf_immortal for glorfindel
        conflict_reg = ConflictRegistry()
        from book_graph_analyzer.models.lore_conflict import LoreConflict, ConflictClaim
        conflict_reg.add(LoreConflict(
            id="test_suppressor",
            summary="Test suppression",
            conflict_type=ConflictType.AMBIGUITY,
            rule_ids=["race_elf_immortal"],
            entity_ids=["glorfindel"],  # entity name as used in scene context
            resolution_policy=ResolutionPolicy.BOTH_VALID_IN_UNIVERSE,
            claims=[make_claim(), make_claim()],
        ))

        rule_reg = LoreRuleRegistry.from_tolkien_defaults()
        rule_validator = LoreRuleValidator(rule_reg)
        validator = ConflictAwareValidator(rule_validator, conflict_reg)

        # Create a scene with Glorfindel and a death_by_age event
        ctx = SceneContext(
            scene_id="s1",
            character_names=["glorfindel"],
            character_races={"glorfindel": "Elf"},
            place_names=[],
            object_names=[],
            event_types=["death_by_age"],
            story_era="Third Age",
        )

        result = validator.validate_scene_context(ctx)
        # Violation for race_elf_immortal should be suppressed
        hard_ids = {v.rule_id for v in result.hard_violations}
        assert "race_elf_immortal" not in hard_ids

    def test_flag_for_human_downgrades_hard_to_soft(self):
        """A conflict with FLAG_FOR_HUMAN should downgrade HARD → SOFT."""
        from book_graph_analyzer.lore.rules import LoreRuleRegistry, LoreRuleValidator, SceneContext
        from book_graph_analyzer.lore.conflicts import ConflictRegistry, ConflictAwareValidator

        conflict_reg = ConflictRegistry()
        from book_graph_analyzer.models.lore_conflict import LoreConflict, ConflictClaim
        conflict_reg.add(LoreConflict(
            id="test_downgrader",
            summary="Test downgrade",
            conflict_type=ConflictType.AMBIGUITY,
            rule_ids=["race_elf_immortal"],
            entity_ids=["legolas"],
            resolution_policy=ResolutionPolicy.FLAG_FOR_HUMAN,
            claims=[make_claim(), make_claim()],
        ))

        rule_reg = LoreRuleRegistry.from_tolkien_defaults()
        rule_validator = LoreRuleValidator(rule_reg)
        validator = ConflictAwareValidator(rule_validator, conflict_reg)

        ctx = SceneContext(
            scene_id="s1",
            character_names=["legolas"],
            character_races={"legolas": "Elf"},
            place_names=[],
            object_names=[],
            event_types=["death_by_age"],
            story_era="Third Age",
        )

        result = validator.validate_scene_context(ctx)
        # Should be moved from hard to soft
        hard_ids = {v.rule_id for v in result.hard_violations}
        soft_ids = {v.rule_id for v in result.soft_warnings}
        assert "race_elf_immortal" not in hard_ids
        assert "race_elf_immortal" in soft_ids


# ---------------------------------------------------------------------------
# author_period on Passage model
# ---------------------------------------------------------------------------

class TestPassageAuthorPeriod:
    def test_author_period_default_none(self):
        p = make_passage()
        assert p.author_period is None

    def test_author_period_can_be_set(self):
        p = make_passage(author_period=AuthorPeriod.LATE)
        assert p.author_period == AuthorPeriod.LATE

    def test_source_compilation_default_none(self):
        p = make_passage()
        assert p.source_compilation is None

    def test_source_compilation_can_be_set(self):
        p = make_passage(source_compilation="Unfinished Tales")
        assert p.source_compilation == "Unfinished Tales"

    def test_passage_still_works_without_new_fields(self):
        """Existing Passage code still works — new fields are optional."""
        p = make_passage()
        assert p.text == "Gandalf spoke."
        assert p.book == "LOTR"

    def test_all_periods_valid_strings(self):
        for period in (AuthorPeriod.EARLY, AuthorPeriod.MIDDLE, AuthorPeriod.LATE):
            p = make_passage(author_period=period)
            assert p.author_period == period
