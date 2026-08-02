"""Tests for Character Emotional Arc + Relationship Sentiment Valence (Issue #8).

All tests are pure-Python — no Neo4j or LLM required. Covers:
  - EmotionalState model
  - TolkienRegister and RelationshipSentiment enums
  - ArcCheckpoint and CharacterArc
  - TOLKIEN_CHARACTER_ARCS baseline (Frodo's arc specifically)
  - TOLKIEN_RELATIONSHIP_SENTIMENTS (asymmetric relational sentiment)
  - EmotionalArcValidator
  - EmotionalStateExtractor (heuristic text extraction)
  - FeltEdge and RelationalSentimentEdge models
"""

from book_graph_analyzer.models.emotional_arc import (
    EmotionalState,
    ArcCheckpoint,
    CharacterArc,
    FeltEdge,
    RelationalSentimentEdge,
    TolkienRegister,
    RelationshipSentiment,
    REGISTER_ANCHORS,
    SENTIMENT_VALENCE,
)
from book_graph_analyzer.lore.emotional_arc import (
    EmotionalArcValidator,
    TOLKIEN_CHARACTER_ARCS,
    TOLKIEN_RELATIONSHIP_SENTIMENTS,
    extract_emotional_state_from_text,
    text_to_emotional_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_state(
    state_id: str = "s1",
    valence: float = 0.0,
    agency: float = 0.0,
    register: str = TolkienRegister.RESOLUTE,
    description: str = "test state",
) -> EmotionalState:
    return EmotionalState(
        id=state_id,
        valence=valence,
        agency=agency,
        dominant_register=register,
        description=description,
    )


def make_checkpoint(
    label: str = "test_cp",
    year: int = 3019,
    valid_registers: list[str] | None = None,
    invalid_registers: list[str] | None = None,
    hardness: str = "SOFT",
    year_end: int | None = None,
) -> ArcCheckpoint:
    return ArcCheckpoint(
        label=label,
        story_year=year,
        story_year_end=year_end,
        emotional_state=make_state(),
        description="Test checkpoint",
        hardness=hardness,
        valid_registers=(
            valid_registers if valid_registers is not None
            else [TolkienRegister.RESOLUTE, TolkienRegister.HOPE]
        ),
        invalid_registers=(
            invalid_registers if invalid_registers is not None
            else [TolkienRegister.COZY]
        ),
    )


# ---------------------------------------------------------------------------
# TolkienRegister
# ---------------------------------------------------------------------------

class TestTolkienRegister:
    def test_all_registers_have_anchor(self):
        for register in TolkienRegister:
            assert register in REGISTER_ANCHORS, f"Missing anchor for {register}"

    def test_anchors_in_valid_range(self):
        for register, (valence, agency) in REGISTER_ANCHORS.items():
            assert -1.0 <= valence <= 1.0, f"Valence out of range for {register}"
            assert -1.0 <= agency <= 1.0, f"Agency out of range for {register}"

    def test_cozy_is_positive_high_agency(self):
        v, a = REGISTER_ANCHORS[TolkienRegister.COZY]
        assert v > 0.5
        assert a > 0.5

    def test_dread_is_negative_low_agency(self):
        v, a = REGISTER_ANCHORS[TolkienRegister.DREAD]
        assert v < -0.5
        assert a < 0.0

    def test_burden_is_negative_low_agency(self):
        v, a = REGISTER_ANCHORS[TolkienRegister.BURDEN]
        assert v < -0.3
        assert a < -0.4

    def test_eucatastrophic_is_very_positive(self):
        v, a = REGISTER_ANCHORS[TolkienRegister.EUCATASTROPHIC]
        assert v > 0.8


# ---------------------------------------------------------------------------
# RelationshipSentiment
# ---------------------------------------------------------------------------

class TestRelationshipSentiment:
    def test_all_sentiments_have_valence(self):
        for sent in RelationshipSentiment:
            assert sent in SENTIMENT_VALENCE, f"Missing valence for {sent}"

    def test_loyal_is_very_positive(self):
        assert SENTIMENT_VALENCE[RelationshipSentiment.LOYAL] > 0.8

    def test_enemy_is_very_negative(self):
        assert SENTIMENT_VALENCE[RelationshipSentiment.ENEMY] < -0.5

    def test_fear_is_negative(self):
        assert SENTIMENT_VALENCE[RelationshipSentiment.FEAR] < -0.5

    def test_love_is_most_positive(self):
        assert SENTIMENT_VALENCE[RelationshipSentiment.LOVE] >= 0.9


# ---------------------------------------------------------------------------
# EmotionalState
# ---------------------------------------------------------------------------

class TestEmotionalState:
    def test_basic_creation(self):
        state = make_state(valence=0.7, agency=0.5)
        assert state.valence == 0.7
        assert state.agency == 0.5

    def test_to_neo4j_props(self):
        state = make_state(state_id="s1", valence=0.7)
        props = state.to_neo4j_props()
        assert props["id"] == "s1"
        assert props["valence"] == 0.7
        assert "dominant_register" in props

    def test_from_dict_roundtrip(self):
        state = make_state(state_id="s1", valence=0.5, agency=-0.3)
        d = state.to_dict()
        state2 = EmotionalState.from_dict(d)
        assert state2.id == state.id
        assert abs(state2.valence - state.valence) < 1e-6

    def test_distance_from_same_state(self):
        s = make_state(valence=0.5, agency=0.3)
        assert s.distance_from(s) == 0.0

    def test_distance_from_opposite(self):
        s1 = make_state(valence=1.0, agency=1.0)
        s2 = make_state(valence=-1.0, agency=-1.0)
        dist = s1.distance_from(s2)
        assert dist > 2.0

    def test_compatible_with_close_state(self):
        s1 = make_state(valence=0.5, agency=0.3)
        s2 = make_state(valence=0.55, agency=0.35)
        assert s1.compatible_with(s2)

    def test_not_compatible_with_far_state(self):
        s1 = make_state(valence=0.9, agency=0.9)   # cozy/joyful
        s2 = make_state(valence=-0.9, agency=-0.9) # despair/powerless
        assert not s1.compatible_with(s2)

    def test_is_violation_when_far(self):
        s_expected = make_state(valence=-0.7, agency=-0.6)  # burden
        s_proposed = make_state(valence=0.9, agency=0.8)   # cozy joy
        assert s_proposed.is_violation(s_expected)

    def test_not_violation_when_close(self):
        s_expected = make_state(valence=0.3, agency=0.5)
        s_proposed = make_state(valence=0.35, agency=0.45)
        assert not s_proposed.is_violation(s_expected)


# ---------------------------------------------------------------------------
# ArcCheckpoint
# ---------------------------------------------------------------------------

class TestArcCheckpoint:
    def test_covers_year_exact(self):
        cp = make_checkpoint(year=3019)
        assert cp.covers_year(3019)

    def test_covers_year_with_range(self):
        cp = make_checkpoint(year=3018, year_end=3019)
        assert cp.covers_year(3018)
        assert cp.covers_year(3019)
        assert not cp.covers_year(3017)
        assert not cp.covers_year(3020)

    def test_covers_year_tolerance_window(self):
        """Single-year checkpoints have ±15 year tolerance."""
        cp = make_checkpoint(year=3019)
        assert cp.covers_year(3010)  # within 15 years
        assert cp.covers_year(3030)  # within 15 years
        assert not cp.covers_year(2990)  # >15 years away

    def test_is_valid_register_in_valid_list(self):
        cp = make_checkpoint(valid_registers=[TolkienRegister.RESOLUTE, TolkienRegister.HOPE])
        assert cp.is_valid_register(TolkienRegister.RESOLUTE)
        assert cp.is_valid_register(TolkienRegister.HOPE)

    def test_is_invalid_register_not_in_valid_list(self):
        cp = make_checkpoint(
            valid_registers=[TolkienRegister.RESOLUTE],
            invalid_registers=[],
        )
        assert not cp.is_valid_register(TolkienRegister.COZY)

    def test_is_invalid_register_explicitly_excluded(self):
        cp = make_checkpoint(
            valid_registers=[],  # No specific valid set
            invalid_registers=[TolkienRegister.COZY],
        )
        assert not cp.is_valid_register(TolkienRegister.COZY)

    def test_any_register_valid_when_no_constraints(self):
        cp = make_checkpoint(valid_registers=[], invalid_registers=[])
        assert cp.is_valid_register(TolkienRegister.COZY)
        assert cp.is_valid_register(TolkienRegister.DREAD)


# ---------------------------------------------------------------------------
# CharacterArc
# ---------------------------------------------------------------------------

class TestCharacterArc:
    def test_get_checkpoint_for_covered_year(self):
        arc = CharacterArc(
            character_id="test",
            character_name="Test Character",
            checkpoints=[make_checkpoint(year=3019, year_end=3019)],
        )
        cp = arc.get_checkpoint_for_year(3019)
        assert cp is not None

    def test_get_checkpoint_returns_none_for_uncovered_year(self):
        arc = CharacterArc(
            character_id="test",
            character_name="Test",
            checkpoints=[make_checkpoint(year=3019, year_end=3019)],
        )
        cp = arc.get_checkpoint_for_year(2500)
        assert cp is None

    def test_validate_state_valid(self):
        cp = make_checkpoint(
            year=3019, year_end=3019,
            valid_registers=[TolkienRegister.RESOLUTE],
            invalid_registers=[TolkienRegister.COZY],
        )
        arc = CharacterArc("test", "Test", [cp])
        is_valid, explanation = arc.validate_state(TolkienRegister.RESOLUTE, 3019)
        assert is_valid is True

    def test_validate_state_violation(self):
        cp = make_checkpoint(
            year=3019, year_end=3019,
            valid_registers=[TolkienRegister.RESOLUTE],
            invalid_registers=[TolkienRegister.COZY],
        )
        arc = CharacterArc("test", "Test", [cp])
        is_valid, explanation = arc.validate_state(TolkienRegister.COZY, 3019)
        assert is_valid is False
        assert "VIOLATION" in explanation

    def test_validate_state_no_checkpoint_passes(self):
        arc = CharacterArc("test", "Test", [])
        is_valid, _ = arc.validate_state(TolkienRegister.COZY, 9999)
        assert is_valid is True


# ---------------------------------------------------------------------------
# TOLKIEN_CHARACTER_ARCS — Frodo's arc in detail
# ---------------------------------------------------------------------------

class TestFrodoArc:
    def setup_method(self):
        self.arc = TOLKIEN_CHARACTER_ARCS["frodo_baggins"]

    def test_arc_exists(self):
        assert self.arc is not None
        assert self.arc.character_name == "Frodo Baggins"

    def test_multiple_checkpoints(self):
        assert len(self.arc.checkpoints) >= 5

    def test_shire_is_cozy(self):
        # Use 3005 — clearly closer to shire_idyll (3001) than flight_to_rivendell (3018)
        cp = self.arc.get_checkpoint_for_year(3005)
        assert cp is not None
        assert TolkienRegister.COZY in cp.valid_registers

    def test_shire_rejects_dread(self):
        cp = self.arc.get_checkpoint_for_year(3005)
        assert cp is not None
        assert TolkienRegister.DREAD in cp.invalid_registers

    def test_mordor_rejects_cozy(self):
        """Frodo near Mordor (TA 3019) — cozy is a hard violation."""
        cp = self.arc.get_checkpoint_for_year(3019)
        assert cp is not None
        assert TolkienRegister.COZY in cp.invalid_registers

    def test_mordor_rejects_hope(self):
        """Frodo near Mordor — hopeful and energetic is a violation."""
        cp = self.arc.get_checkpoint_for_year(3019)
        assert TolkienRegister.HOPE in cp.invalid_registers or \
               TolkienRegister.EUCATASTROPHIC in cp.invalid_registers

    def test_mordor_allows_burden(self):
        cp = self.arc.get_checkpoint_for_year(3019)
        assert TolkienRegister.BURDEN in cp.valid_registers

    def test_mordor_checkpoint_is_hard(self):
        # The Cirith Ungol / Mordor checkpoint should be HARD
        hard_cps = [cp for cp in self.arc.checkpoints if cp.hardness == "HARD"]
        hard_years = [cp.story_year for cp in hard_cps]
        assert 3019 in hard_years

    def test_validates_violation_cozy_at_mordor(self):
        """Spec example: 'hopeful, energetic' at Cirith Ungol → violation."""
        is_valid, explanation = self.arc.validate_state(TolkienRegister.COZY, 3019)
        assert is_valid is False

    def test_validates_valid_burden_at_mordor(self):
        is_valid, _ = self.arc.validate_state(TolkienRegister.BURDEN, 3019)
        assert is_valid is True

    def test_arc_progression_valence_decreases_toward_mordor(self):
        """Frodo's valence should be lower near Mordor than in the Shire."""
        shire_cp = self.arc.get_checkpoint_for_year(3010)
        mordor_cp = self.arc.get_checkpoint_for_year(3019)
        if shire_cp and mordor_cp and shire_cp.emotional_state and mordor_cp.emotional_state:
            assert mordor_cp.emotional_state.valence < shire_cp.emotional_state.valence


# ---------------------------------------------------------------------------
# Other characters
# ---------------------------------------------------------------------------

class TestSamArc:
    def setup_method(self):
        self.arc = TOLKIEN_CHARACTER_ARCS["samwise_gamgee"]

    def test_arc_exists(self):
        assert self.arc is not None

    def test_mordor_sam_is_resolute(self):
        cp = self.arc.get_checkpoint_for_year(3019)
        assert cp is not None
        assert TolkienRegister.RESOLUTE in cp.valid_registers

    def test_mordor_sam_is_hard(self):
        hard_cps = [cp for cp in self.arc.checkpoints if cp.hardness == "HARD"]
        assert len(hard_cps) >= 1


class TestGandalfArc:
    def test_gandalf_white_is_hard(self):
        arc = TOLKIEN_CHARACTER_ARCS["gandalf"]
        hard_cps = [cp for cp in arc.checkpoints if cp.hardness == "HARD"]
        assert any(cp.label == "gandalf_white" for cp in hard_cps)

    def test_gandalf_white_rejects_burden(self):
        arc = TOLKIEN_CHARACTER_ARCS["gandalf"]
        cp = arc.get_checkpoint_for_year(3019)
        if cp:
            assert TolkienRegister.BURDEN in cp.invalid_registers


# ---------------------------------------------------------------------------
# EmotionalArcValidator
# ---------------------------------------------------------------------------

class TestEmotionalArcValidator:
    def setup_method(self):
        self.validator = EmotionalArcValidator()

    def test_get_arc_by_canonical_id(self):
        arc = self.validator.get_arc("frodo_baggins")
        assert arc is not None

    def test_get_arc_by_common_name(self):
        arc = self.validator.get_arc("Frodo")
        assert arc is not None
        assert arc.character_id == "frodo_baggins"

    def test_get_arc_by_alias_sam(self):
        arc = self.validator.get_arc("Sam")
        assert arc is not None
        assert arc.character_id == "samwise_gamgee"

    def test_get_arc_by_alias_strider(self):
        arc = self.validator.get_arc("Strider")
        assert arc is not None
        assert arc.character_id == "aragorn"

    def test_get_arc_unknown_returns_none(self):
        arc = self.validator.get_arc("Tom Sawyer")
        assert arc is None

    def test_validate_arc_violation_frodo_mordor_cozy(self):
        """Spec case: Frodo at TA 3019 feeling 'cozy' = violation."""
        is_valid, explanation = self.validator.validate_arc(
            character="Frodo",
            story_year=3019,
            proposed_register=TolkienRegister.COZY,
        )
        assert is_valid is False
        assert "VIOLATION" in explanation

    def test_validate_arc_valid_frodo_mordor_burden(self):
        is_valid, explanation = self.validator.validate_arc(
            character="Frodo",
            story_year=3019,
            proposed_register=TolkienRegister.BURDEN,
        )
        assert is_valid is True

    def test_validate_arc_valid_sam_mordor_resolute(self):
        is_valid, _ = self.validator.validate_arc(
            character="Sam",
            story_year=3019,
            proposed_register=TolkienRegister.RESOLUTE,
        )
        assert is_valid is True

    def test_validate_arc_unknown_character_passes(self):
        """Unknown character has no constraints — always passes."""
        is_valid, explanation = self.validator.validate_arc(
            character="Tom Bombadil",
            story_year=3019,
            proposed_register=TolkienRegister.COZY,
        )
        assert is_valid is True

    def test_validate_arc_from_text_detects_cozy(self):
        text = "Frodo felt warm and comfortable by the hearth in the cozy hobbit-hole."
        is_valid, register, explanation = self.validator.validate_arc_from_text(
            character="Frodo", story_year=3019, text=text
        )
        assert register == TolkienRegister.COZY
        assert is_valid is False

    def test_validate_arc_from_text_burden_passes(self):
        text = "The weight was crushing, the burden unbearable, he could barely drag himself forward."
        is_valid, register, explanation = self.validator.validate_arc_from_text(
            character="Frodo", story_year=3019, text=text
        )
        assert register == TolkienRegister.BURDEN
        assert is_valid is True

    def test_expected_state_returns_checkpoint(self):
        cp = self.validator.expected_state("Frodo", 3019)
        assert cp is not None

    def test_expected_state_none_for_unknown_year(self):
        cp = self.validator.expected_state("Frodo", 1000)
        assert cp is None

    def test_all_characters_returns_list(self):
        chars = self.validator.all_characters()
        assert "Frodo Baggins" in chars
        assert "Samwise Gamgee" in chars

    def test_list_checkpoints_returns_all(self):
        cps = self.validator.list_checkpoints("Frodo")
        assert len(cps) >= 5


# ---------------------------------------------------------------------------
# Heuristic text extraction
# ---------------------------------------------------------------------------

class TestEmotionalStateExtractor:
    def test_extract_cozy_from_shire_text(self):
        text = "Bilbo sat by the warm hearth in his comfortable hobbit-hole and had supper."
        register, confidence = extract_emotional_state_from_text(text)
        assert register == TolkienRegister.COZY
        assert confidence > 0.0

    def test_extract_dread_from_mordor_text(self):
        text = "The dread darkness pressed upon them; shadow and terror lay on every side."
        register, confidence = extract_emotional_state_from_text(text)
        assert register == TolkienRegister.DREAD
        assert confidence > 0.0

    def test_extract_burden_from_ring_bearer_text(self):
        text = "The weight was crushing and unbearable; dragging him down, exhausted beyond words."
        register, confidence = extract_emotional_state_from_text(text)
        assert register == TolkienRegister.BURDEN
        assert confidence > 0.0

    def test_extract_wonder_from_elves_text(self):
        text = "He marvelled at the beautiful radiant elves, wondrous and magnificent beyond words."
        register, confidence = extract_emotional_state_from_text(text)
        assert register == TolkienRegister.WONDER
        assert confidence > 0.0

    def test_extract_resolute_from_determination_text(self):
        text = "He was determined and resolved; his duty was clear; he stood firm and would not yield."
        register, confidence = extract_emotional_state_from_text(text)
        assert register == TolkienRegister.RESOLUTE
        assert confidence > 0.0

    def test_returns_default_for_neutral_text(self):
        text = "He walked along the path."
        register, confidence = extract_emotional_state_from_text(text)
        assert isinstance(register, str)
        assert 0.0 <= confidence <= 1.0

    def test_text_to_emotional_state_returns_object(self):
        text = "He sat by the fire, warm and safe."
        state = text_to_emotional_state(text, state_id="test")
        assert isinstance(state, EmotionalState)
        assert state.id == "test"
        assert state.dominant_register == TolkienRegister.COZY


# ---------------------------------------------------------------------------
# TOLKIEN_RELATIONSHIP_SENTIMENTS — asymmetric relational sentiment
# ---------------------------------------------------------------------------

class TestRelationalSentiments:
    def test_at_least_five_edges(self):
        assert len(TOLKIEN_RELATIONSHIP_SENTIMENTS) >= 5

    def test_sam_frodo_is_loyal(self):
        sam_frodo = next(
            e for e in TOLKIEN_RELATIONSHIP_SENTIMENTS
            if e.from_character_id == "samwise_gamgee"
            and e.to_character_id == "frodo_baggins"
        )
        assert sam_frodo.sentiment == RelationshipSentiment.LOYAL
        assert sam_frodo.valence > 0.8

    def test_sam_gollum_is_wary_negative(self):
        sam_gollum = next(
            e for e in TOLKIEN_RELATIONSHIP_SENTIMENTS
            if e.from_character_id == "samwise_gamgee"
            and e.to_character_id == "gollum"
        )
        assert sam_gollum.sentiment == RelationshipSentiment.WARY
        assert sam_gollum.valence < 0.0

    def test_gollum_sam_is_fear(self):
        gollum_sam = next(
            e for e in TOLKIEN_RELATIONSHIP_SENTIMENTS
            if e.from_character_id == "gollum"
            and e.to_character_id == "samwise_gamgee"
        )
        assert gollum_sam.sentiment == RelationshipSentiment.FEAR

    def test_asymmetry_sam_gollum_vs_gollum_sam(self):
        """Sam → Gollum ≠ Gollum → Sam — demonstrates asymmetric sentiment."""
        sam_gollum = next(
            e for e in TOLKIEN_RELATIONSHIP_SENTIMENTS
            if e.from_character_id == "samwise_gamgee"
            and e.to_character_id == "gollum"
        )
        gollum_sam = next(
            e for e in TOLKIEN_RELATIONSHIP_SENTIMENTS
            if e.from_character_id == "gollum"
            and e.to_character_id == "samwise_gamgee"
        )
        # Different sentiments, both negative
        assert sam_gollum.sentiment != gollum_sam.sentiment

    def test_frodo_gollum_is_pity(self):
        frodo_gollum = next(
            (e for e in TOLKIEN_RELATIONSHIP_SENTIMENTS
             if e.from_character_id == "frodo_baggins"
             and e.to_character_id == "gollum"),
            None
        )
        if frodo_gollum:
            assert frodo_gollum.sentiment == RelationshipSentiment.PITY

    def test_to_neo4j_props_has_required_fields(self):
        edge = TOLKIEN_RELATIONSHIP_SENTIMENTS[0]
        props = edge.to_neo4j_props()
        assert "sentiment" in props
        assert "valence" in props
        assert "valence_trajectory" in props

    def test_valence_trajectories_valid(self):
        valid_trajectories = {"improving", "deteriorating", "stable", "volatile"}
        for e in TOLKIEN_RELATIONSHIP_SENTIMENTS:
            assert e.valence_trajectory in valid_trajectories, \
                f"Invalid trajectory '{e.valence_trajectory}' for {e.from_character_id}"


# ---------------------------------------------------------------------------
# FeltEdge and RelationalSentimentEdge models
# ---------------------------------------------------------------------------

class TestFeltEdge:
    def test_basic_creation(self):
        edge = FeltEdge(
            character_id="frodo",
            emotional_state_id="state_mordor",
            era="Third Age",
            year=3019,
            passage_id="p_001",
        )
        assert edge.character_id == "frodo"
        assert edge.year == 3019

    def test_to_neo4j_props_includes_era(self):
        edge = FeltEdge(
            character_id="frodo",
            emotional_state_id="state_mordor",
            era="Third Age",
        )
        props = edge.to_neo4j_props()
        assert props["era"] == "Third Age"

    def test_to_neo4j_props_omits_none_values(self):
        edge = FeltEdge(
            character_id="frodo",
            emotional_state_id="s1",
            era="Third Age",
        )
        props = edge.to_neo4j_props()
        assert "year" not in props
        assert "passage_id" not in props
        assert "toward_entity_id" not in props


class TestRelationalSentimentEdge:
    def test_basic_creation(self):
        edge = RelationalSentimentEdge(
            from_character_id="sam",
            to_character_id="frodo",
            sentiment=RelationshipSentiment.LOYAL,
            valence=0.95,
            valence_trajectory="stable",
        )
        assert edge.sentiment == RelationshipSentiment.LOYAL
        assert edge.valence == 0.95

    def test_source_passage_ids_default_empty(self):
        edge = RelationalSentimentEdge(
            from_character_id="sam",
            to_character_id="frodo",
            sentiment=RelationshipSentiment.LOYAL,
            valence=0.9,
            valence_trajectory="stable",
        )
        assert edge.source_passage_ids == []
