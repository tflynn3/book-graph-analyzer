"""Emotional arc system — pre-defined Tolkien arcs, validator, extractor, and Neo4j writer.

Contains:
  TOLKIEN_CHARACTER_ARCS  — canonical arcs for Frodo, Sam, Gandalf, Aragorn, etc.
  TOLKIEN_RELATIONSHIP_SENTIMENTS — relational sentiment edges
  EmotionalArcValidator   — validates proposed states against canonical arcs
  EmotionalStateExtractor — heuristic extraction from text (no LLM required)
  EmotionalArcNeo4jWriter — writes EmotionalState nodes + FELT / KNOWS edges
"""

from __future__ import annotations

import re
from typing import Optional

from ..models.emotional_arc import (
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


# ---------------------------------------------------------------------------
# Helpers for building arc checkpoints
# ---------------------------------------------------------------------------

def _state(
    label: str,
    valence: float,
    agency: float,
    register: str,
    description: str,
) -> EmotionalState:
    return EmotionalState(
        id=f"state_{label.replace(' ', '_')}",
        valence=valence,
        agency=agency,
        dominant_register=register,
        description=description,
    )


def _cp(
    label: str,
    year: int,
    valence: float,
    agency: float,
    register: str,
    description: str,
    valid_registers: list[str],
    invalid_registers: list[str],
    hardness: str = "SOFT",
    year_end: Optional[int] = None,
) -> ArcCheckpoint:
    return ArcCheckpoint(
        label=label,
        story_year=year,
        story_year_end=year_end,
        emotional_state=_state(label, valence, agency, register, description),
        description=description,
        hardness=hardness,
        valid_registers=valid_registers,
        invalid_registers=invalid_registers,
    )


# ---------------------------------------------------------------------------
# Frodo Baggins — the most detailed arc
# ---------------------------------------------------------------------------

_FRODO_ARC = CharacterArc(
    character_id="frodo_baggins",
    character_name="Frodo Baggins",
    checkpoints=[
        _cp(
            label="shire_idyll",
            year=3001, year_end=3018,
            valence=0.8, agency=0.7,
            register=TolkienRegister.COZY,
            description=(
                "Frodo in the Shire — content, curious, comfortable. "
                "Slightly melancholy after Bilbo's departure but fundamentally at peace."
            ),
            valid_registers=[
                TolkienRegister.COZY, TolkienRegister.WONDER,
                TolkienRegister.ELEGIAC, TolkienRegister.HOPE,
            ],
            invalid_registers=[
                TolkienRegister.DREAD, TolkienRegister.RAGE,
                TolkienRegister.BURDEN,
            ],
        ),
        _cp(
            label="flight_to_rivendell",
            year=3018,
            valence=0.2, agency=0.3,
            register=TolkienRegister.DREAD,
            description=(
                "Fleeing the Shire with the Black Riders on his heels. "
                "Frightened but pressing on — courage overlaid with fear."
            ),
            valid_registers=[
                TolkienRegister.DREAD, TolkienRegister.RESOLUTE,
                TolkienRegister.WONDER, TolkienRegister.HOPE,
            ],
            invalid_registers=[
                TolkienRegister.COZY, TolkienRegister.EUCATASTROPHIC,
            ],
        ),
        _cp(
            label="fellowship_of_the_ring",
            year=3018, year_end=3019,
            valence=0.4, agency=0.5,
            register=TolkienRegister.RESOLUTE,
            description=(
                "Frodo bearing the Ring through the Fellowship. "
                "Wonder at the wider world, growing weight, but still purposeful and hopeful."
            ),
            valid_registers=[
                TolkienRegister.RESOLUTE, TolkienRegister.WONDER,
                TolkienRegister.HOPE, TolkienRegister.BURDEN,
            ],
            invalid_registers=[
                TolkienRegister.COZY, TolkienRegister.EUCATASTROPHIC,
                TolkienRegister.RAGE,
            ],
        ),
        _cp(
            label="emyn_muil_ithilien",
            year=3019,
            valence=-0.1, agency=0.2,
            register=TolkienRegister.BURDEN,
            description=(
                "After the breaking of the Fellowship. "
                "Frodo alone (with Sam) feels the Ring's weight increasing. "
                "Determined but increasingly burdened, wary of Gollum."
            ),
            valid_registers=[
                TolkienRegister.BURDEN, TolkienRegister.RESOLUTE,
                TolkienRegister.DREAD, TolkienRegister.GRIEF,
            ],
            invalid_registers=[
                TolkienRegister.COZY, TolkienRegister.WONDER,
                TolkienRegister.HOPE, TolkienRegister.EUCATASTROPHIC,
            ],
            hardness="HARD",
        ),
        _cp(
            label="cirith_ungol_mordor",
            year=3019,
            valence=-0.7, agency=-0.6,
            register=TolkienRegister.BURDEN,
            description=(
                "Cirith Ungol and the plains of Mordor. "
                "Frodo consumed by the Ring's weight — despair, exhaustion, "
                "near-total loss of agency. Sometimes hostile toward Sam."
            ),
            valid_registers=[
                TolkienRegister.BURDEN, TolkienRegister.DREAD,
                TolkienRegister.GRIEF,
            ],
            invalid_registers=[
                TolkienRegister.COZY, TolkienRegister.WONDER,
                TolkienRegister.HOPE, TolkienRegister.EUCATASTROPHIC,
                TolkienRegister.RESOLUTE, TolkienRegister.RAGE,
            ],
            hardness="HARD",
        ),
        _cp(
            label="mount_doom_eucatastrophe",
            year=3019,
            valence=0.7, agency=0.1,
            register=TolkienRegister.TRANSCENDENT,
            description=(
                "The eucatastrophe at the Crack of Doom. "
                "The burden is lifted through Gollum's mercy — overwhelming relief, "
                "grace, wonder, transcendence beyond ordinary feeling."
            ),
            valid_registers=[
                TolkienRegister.TRANSCENDENT, TolkienRegister.EUCATASTROPHIC,
                TolkienRegister.GRIEF, TolkienRegister.WONDER,
            ],
            invalid_registers=[
                TolkienRegister.COZY, TolkienRegister.RAGE,
                TolkienRegister.DREAD,
            ],
        ),
        _cp(
            label="scouring_of_the_shire",
            year=3019, year_end=3020,
            valence=0.1, agency=0.6,
            register=TolkienRegister.RESOLUTE,
            description=(
                "Post-Ring Frodo — not quite himself, touched by Mordor, "
                "but finding purpose in healing the Shire. Elegiac undertone."
            ),
            valid_registers=[
                TolkienRegister.RESOLUTE, TolkienRegister.ELEGIAC,
                TolkienRegister.GRIEF, TolkienRegister.PITY,
            ],
            invalid_registers=[
                TolkienRegister.COZY, TolkienRegister.EUCATASTROPHIC,
                TolkienRegister.WONDER,
            ],
        ),
        _cp(
            label="grey_havens_departure",
            year=3021,
            valence=0.5, agency=0.4,
            register=TolkienRegister.ELEGIAC,
            description=(
                "Frodo's departure at the Grey Havens — bittersweet peace. "
                "Sadness at leaving, acceptance of what he has become, "
                "quiet joy at the healing ahead."
            ),
            valid_registers=[
                TolkienRegister.ELEGIAC, TolkienRegister.TRANSCENDENT,
                TolkienRegister.PITY, TolkienRegister.WONDER,
            ],
            invalid_registers=[
                TolkienRegister.COZY, TolkienRegister.RAGE,
                TolkienRegister.DREAD, TolkienRegister.BURDEN,
            ],
        ),
    ]
)


# ---------------------------------------------------------------------------
# Samwise Gamgee
# ---------------------------------------------------------------------------

_SAM_ARC = CharacterArc(
    character_id="samwise_gamgee",
    character_name="Samwise Gamgee",
    checkpoints=[
        _cp(
            label="shire_gardener",
            year=3001, year_end=3018,
            valence=0.85, agency=0.7,
            register=TolkienRegister.COZY,
            description="Sam in the Shire — simple contentment, loyal service, dreams of Elves.",
            valid_registers=[TolkienRegister.COZY, TolkienRegister.WONDER, TolkienRegister.HOPE],
            invalid_registers=[TolkienRegister.DREAD, TolkienRegister.BURDEN],
        ),
        _cp(
            label="on_the_quest",
            year=3018, year_end=3019,
            valence=0.5, agency=0.6,
            register=TolkienRegister.RESOLUTE,
            description="Sam on the Quest — steadfast, faithful, occasionally homesick but never wavering.",
            valid_registers=[
                TolkienRegister.RESOLUTE, TolkienRegister.COZY, TolkienRegister.WONDER,
                TolkienRegister.GRIEF, TolkienRegister.HOPE,
            ],
            invalid_registers=[TolkienRegister.DREAD, TolkienRegister.BURDEN],
        ),
        _cp(
            label="carrying_frodo_mordor",
            year=3019,
            valence=0.3, agency=0.7,
            register=TolkienRegister.RESOLUTE,
            description=(
                "Sam carrying Frodo up Mount Doom — utter determination, love, "
                "personal agency through service. The emotional opposite of Frodo's despair."
            ),
            valid_registers=[
                TolkienRegister.RESOLUTE, TolkienRegister.HOPE,
                TolkienRegister.GRIEF, TolkienRegister.PITY,
            ],
            invalid_registers=[TolkienRegister.DREAD, TolkienRegister.BURDEN, TolkienRegister.COZY],
            hardness="HARD",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Gandalf
# ---------------------------------------------------------------------------

_GANDALF_ARC = CharacterArc(
    character_id="gandalf",
    character_name="Gandalf",
    checkpoints=[
        _cp(
            label="gandalf_grey",
            year=3001, year_end=3018,
            valence=0.5, agency=0.7,
            register=TolkienRegister.RESOLUTE,
            description=(
                "Gandalf the Grey — watchful, purposeful, slightly playful. "
                "Carries the weight of long knowledge but with wit."
            ),
            valid_registers=[
                TolkienRegister.RESOLUTE, TolkienRegister.WONDER,
                TolkienRegister.ELEGIAC, TolkienRegister.HOPE,
            ],
            invalid_registers=[TolkienRegister.COZY, TolkienRegister.BURDEN],
        ),
        _cp(
            label="gandalf_white",
            year=3019,
            valence=0.6, agency=0.9,
            register=TolkienRegister.RESOLUTE,
            description=(
                "Gandalf the White — returned with greater power and purpose. "
                "More austere, less playful; entirely focused on the final war."
            ),
            valid_registers=[
                TolkienRegister.RESOLUTE, TolkienRegister.HOPE,
                TolkienRegister.ELEGIAC, TolkienRegister.TRANSCENDENT,
            ],
            invalid_registers=[TolkienRegister.COZY, TolkienRegister.DREAD, TolkienRegister.BURDEN],
            hardness="HARD",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Aragorn
# ---------------------------------------------------------------------------

_ARAGORN_ARC = CharacterArc(
    character_id="aragorn",
    character_name="Aragorn",
    checkpoints=[
        _cp(
            label="strider_incognito",
            year=3001, year_end=3018,
            valence=0.3, agency=0.7,
            register=TolkienRegister.RESOLUTE,
            description=(
                "Aragorn as Strider — carrying the burden of his lineage in secret, "
                "purposeful, a little melancholy, fiercely determined."
            ),
            valid_registers=[
                TolkienRegister.RESOLUTE, TolkienRegister.ELEGIAC,
                TolkienRegister.HOPE, TolkienRegister.WONDER,
            ],
            invalid_registers=[TolkienRegister.COZY, TolkienRegister.DREAD],
        ),
        _cp(
            label="war_of_the_ring",
            year=3018, year_end=3019,
            valence=0.5, agency=0.85,
            register=TolkienRegister.RESOLUTE,
            description=(
                "Aragorn assuming his destiny — resolute, kingly, hope-filled but grave. "
                "The long-waited moment arriving."
            ),
            valid_registers=[
                TolkienRegister.RESOLUTE, TolkienRegister.HOPE,
                TolkienRegister.GRIEF, TolkienRegister.TRANSCENDENT,
            ],
            invalid_registers=[TolkienRegister.DREAD, TolkienRegister.BURDEN, TolkienRegister.COZY],
        ),
    ]
)


# ---------------------------------------------------------------------------
# Gollum / Sméagol
# ---------------------------------------------------------------------------

_GOLLUM_ARC = CharacterArc(
    character_id="gollum",
    character_name="Gollum",
    checkpoints=[
        _cp(
            label="gollum_guiding",
            year=3019,
            valence=-0.3, agency=0.3,
            register=TolkienRegister.DREAD,
            description=(
                "Gollum guiding Frodo and Sam — torn between Sméagol's faint hope "
                "and Gollum's treachery. Dread of Sauron, obsession with the Ring."
            ),
            valid_registers=[
                TolkienRegister.DREAD, TolkienRegister.BURDEN,
                TolkienRegister.GRIEF, TolkienRegister.PITY,
            ],
            invalid_registers=[
                TolkienRegister.COZY, TolkienRegister.WONDER,
                TolkienRegister.RESOLUTE, TolkienRegister.EUCATASTROPHIC,
            ],
        ),
    ]
)


# Registry of all canonical character arcs
TOLKIEN_CHARACTER_ARCS: dict[str, CharacterArc] = {
    "frodo_baggins": _FRODO_ARC,
    "samwise_gamgee": _SAM_ARC,
    "gandalf": _GANDALF_ARC,
    "aragorn": _ARAGORN_ARC,
    "gollum": _GOLLUM_ARC,
}

# Canonical character name aliases
CHARACTER_NAME_MAP: dict[str, str] = {
    "frodo": "frodo_baggins",
    "frodo baggins": "frodo_baggins",
    "sam": "samwise_gamgee",
    "samwise": "samwise_gamgee",
    "samwise gamgee": "samwise_gamgee",
    "gandalf": "gandalf",
    "gandalf the grey": "gandalf",
    "gandalf the white": "gandalf",
    "aragorn": "aragorn",
    "strider": "aragorn",
    "gollum": "gollum",
    "smeagol": "gollum",
    "sméagol": "gollum",
}


# ---------------------------------------------------------------------------
# Pre-defined relational sentiment edges
# ---------------------------------------------------------------------------

TOLKIEN_RELATIONSHIP_SENTIMENTS: list[RelationalSentimentEdge] = [
    # Sam → Frodo: devoted loyalty
    RelationalSentimentEdge(
        from_character_id="samwise_gamgee",
        to_character_id="frodo_baggins",
        sentiment=RelationshipSentiment.LOYAL,
        valence=0.95,
        valence_trajectory="stable",
        era="Third Age",
    ),
    # Frodo → Sam: deep trust and love (declining slightly under Ring influence near end)
    RelationalSentimentEdge(
        from_character_id="frodo_baggins",
        to_character_id="samwise_gamgee",
        sentiment=RelationshipSentiment.LOVE,
        valence=0.9,
        valence_trajectory="volatile",  # Ring distorts it near Mordor
        era="Third Age",
    ),
    # Sam → Gollum: wary distrust
    RelationalSentimentEdge(
        from_character_id="samwise_gamgee",
        to_character_id="gollum",
        sentiment=RelationshipSentiment.WARY,
        valence=-0.7,
        valence_trajectory="deteriorating",
        era="Third Age",
    ),
    # Gollum → Sam: fear and hatred
    RelationalSentimentEdge(
        from_character_id="gollum",
        to_character_id="samwise_gamgee",
        sentiment=RelationshipSentiment.FEAR,
        valence=-0.8,
        valence_trajectory="stable",
        era="Third Age",
    ),
    # Gollum → Frodo: pity mixed with obsession
    RelationalSentimentEdge(
        from_character_id="gollum",
        to_character_id="frodo_baggins",
        sentiment=RelationshipSentiment.PITY,
        valence=0.1,
        valence_trajectory="volatile",
        era="Third Age",
    ),
    # Frodo → Gollum: pity
    RelationalSentimentEdge(
        from_character_id="frodo_baggins",
        to_character_id="gollum",
        sentiment=RelationshipSentiment.PITY,
        valence=0.2,
        valence_trajectory="deteriorating",
        era="Third Age",
    ),
    # Aragorn → Gandalf: respect/trust
    RelationalSentimentEdge(
        from_character_id="aragorn",
        to_character_id="gandalf",
        sentiment=RelationshipSentiment.RESPECT,
        valence=0.8,
        valence_trajectory="stable",
        era="Third Age",
    ),
]


# ---------------------------------------------------------------------------
# EmotionalStateExtractor — heuristic extraction from text (no LLM)
# ---------------------------------------------------------------------------

# Keyword → register mapping for text-based extraction
_REGISTER_KEYWORDS: dict[str, list[str]] = {
    TolkienRegister.ELEGIAC: [
        "fading", "gone", "lost", "mourned", "wept", "tears", "memory", "lament",
        "passed", "no more", "ancient", "shadow of what was",
    ],
    TolkienRegister.EUCATASTROPHIC: [
        "suddenly", "joy", "saved", "victory", "hope fulfilled", "against all",
        "eucatastrophe", "eagles", "light broke",
    ],
    TolkienRegister.DREAD: [
        "dread", "terror", "shadow", "fear", "darkness", "cold", "black",
        "nazgul", "ringwraith", "horror",
    ],
    TolkienRegister.WONDER: [
        "marvelled", "beautiful", "wondrous", "glory", "radiant", "astonished",
        "amazed", "magnificent", "breathtaking",
    ],
    TolkienRegister.COZY: [
        "comfortable", "warm", "hearth", "supper", "pipe-weed", "home",
        "shire", "hobbit-hole", "cozy", "safe",
    ],
    TolkienRegister.RESOLUTE: [
        "determined", "resolved", "must", "duty", "courage", "stood firm",
        "would not yield", "forward", "endure",
    ],
    TolkienRegister.BURDEN: [
        "burden", "weight", "heavy", "exhausted", "could not go on",
        "crushing", "overwhelmed", "dragging",
    ],
    TolkienRegister.GRIEF: [
        "grief", "mourning", "sorrow", "loss", "dead", "gone forever",
        "wept bitterly",
    ],
    TolkienRegister.HOPE: [
        "hope", "dawn", "light", "will endure", "not all is lost",
        "trust", "faith",
    ],
    TolkienRegister.RAGE: [
        "anger", "fury", "wrath", "raged", "burning anger", "snarled",
        "fighting fury",
    ],
    TolkienRegister.PITY: [
        "pity", "mercy", "compassion", "could not bring himself", "spare",
        "gentle", "kind",
    ],
    TolkienRegister.TRANSCENDENT: [
        "beyond", "mystical", "strange peace", "unnatural calm", "otherworldly",
        "beyond grief", "beyond joy",
    ],
}


def extract_emotional_state_from_text(
    text: str,
    character_name: Optional[str] = None,
) -> tuple[str, float]:
    """Heuristically extract the dominant emotional register from passage text.

    Returns:
        (dominant_register, confidence) — confidence 0.0-1.0
    """
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for register, keywords in _REGISTER_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[register] = score

    if not scores:
        return TolkienRegister.RESOLUTE, 0.1

    dominant = max(scores, key=scores.__getitem__)
    total = sum(scores.values())
    confidence = min(1.0, scores[dominant] / max(1, total) + 0.3)
    return dominant, round(confidence, 2)


def text_to_emotional_state(
    text: str,
    state_id: str = "extracted",
    character_name: Optional[str] = None,
) -> EmotionalState:
    """Extract an EmotionalState from text heuristically."""
    register, confidence = extract_emotional_state_from_text(text, character_name)
    valence, agency = REGISTER_ANCHORS.get(register, (0.0, 0.0))
    return EmotionalState(
        id=state_id,
        valence=valence,
        agency=agency,
        dominant_register=register,
        description=f"Extracted from text (confidence={confidence:.0%})",
    )


# ---------------------------------------------------------------------------
# EmotionalArcValidator
# ---------------------------------------------------------------------------

class EmotionalArcValidator:
    """Validates proposed character emotional states against canonical arcs."""

    def __init__(
        self,
        arcs: Optional[dict[str, CharacterArc]] = None,
    ) -> None:
        self._arcs = arcs or TOLKIEN_CHARACTER_ARCS

    def get_arc(self, character_id_or_name: str) -> Optional[CharacterArc]:
        """Look up arc by ID or name (case-insensitive)."""
        name_lower = character_id_or_name.lower()
        # Try direct ID lookup
        if character_id_or_name in self._arcs:
            return self._arcs[character_id_or_name]
        # Try alias map
        canonical_id = CHARACTER_NAME_MAP.get(name_lower)
        if canonical_id and canonical_id in self._arcs:
            return self._arcs[canonical_id]
        return None

    def validate_arc(
        self,
        character: str,
        story_year: int,
        proposed_register: str,
    ) -> tuple[bool, str]:
        """Validate a proposed emotional register for a character at a story year.

        Args:
            character: Character name or ID (e.g. 'Frodo', 'frodo_baggins')
            story_year: Story year in Third Age (e.g. 3019)
            proposed_register: TolkienRegister value being proposed

        Returns:
            (is_valid, explanation)
        """
        arc = self.get_arc(character)
        if not arc:
            return True, f"No canonical arc registered for '{character}'"

        return arc.validate_state(proposed_register, story_year)

    def validate_arc_from_text(
        self,
        character: str,
        story_year: int,
        text: str,
    ) -> tuple[bool, str, str]:
        """Validate emotional state extracted from text.

        Returns:
            (is_valid, detected_register, explanation)
        """
        register, confidence = extract_emotional_state_from_text(text, character)
        is_valid, explanation = self.validate_arc(character, story_year, register)
        full_explanation = (
            f"Detected register: '{register}' (confidence={confidence:.0%})\n"
            f"{explanation}"
        )
        return is_valid, register, full_explanation

    def list_checkpoints(self, character: str) -> list[ArcCheckpoint]:
        """Return all checkpoints for a character."""
        arc = self.get_arc(character)
        if arc is None:
            return []
        return arc.checkpoints

    def expected_state(
        self, character: str, story_year: int
    ) -> Optional[ArcCheckpoint]:
        """Return the expected checkpoint for a character at a given year."""
        arc = self.get_arc(character)
        if arc is None:
            return None
        return arc.get_checkpoint_for_year(story_year)

    def all_characters(self) -> list[str]:
        """Return all character names with registered arcs."""
        return [arc.character_name for arc in self._arcs.values()]


# ---------------------------------------------------------------------------
# EmotionalArcNeo4jWriter
# ---------------------------------------------------------------------------

class EmotionalArcNeo4jWriter:
    """Write EmotionalState nodes and FELT / KNOWS-sentiment edges to Neo4j."""

    def __init__(self, driver=None) -> None:
        self._driver = driver

    @property
    def driver(self):
        if self._driver is None:
            from ..graph.connection import get_driver
            self._driver = get_driver()
        return self._driver

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def upsert_emotional_state(self, state: EmotionalState) -> None:
        """Create or update an EmotionalState node."""
        with self.driver.session() as session:
            session.run(
                "MERGE (e:EmotionalState {id: $id}) SET e += $props",
                id=state.id,
                props=state.to_neo4j_props(),
            )

    def upsert_felt_edge(self, edge: FeltEdge) -> None:
        """Create a FELT edge from Character to EmotionalState."""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (c:Character {id: $char_id})
                MATCH (e:EmotionalState {id: $state_id})
                MERGE (c)-[r:FELT {era: $era}]->(e)
                SET r += $props
                """,
                char_id=edge.character_id,
                state_id=edge.emotional_state_id,
                era=edge.era,
                props=edge.to_neo4j_props(),
            )

    def upsert_relationship_sentiment(self, edge: RelationalSentimentEdge) -> None:
        """Extend a KNOWS edge with sentiment data (asymmetric)."""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (a:Character {id: $from_id})
                MATCH (b:Character {id: $to_id})
                MERGE (a)-[r:KNOWS]->(b)
                SET r += $props
                """,
                from_id=edge.from_character_id,
                to_id=edge.to_character_id,
                props=edge.to_neo4j_props(),
            )

    def seed_tolkien_sentiments(self) -> int:
        """Write all pre-defined Tolkien relationship sentiments to Neo4j."""
        count = 0
        for edge in TOLKIEN_RELATIONSHIP_SENTIMENTS:
            self.upsert_relationship_sentiment(edge)
            count += 1
        return count

    def query_character_arc(self, character_id: str) -> list[dict]:
        """Query all FELT edges for a character from Neo4j, ordered by year."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Character {id: $cid})-[r:FELT]->(e:EmotionalState)
                RETURN c.id AS character_id, e.dominant_register AS register,
                       e.valence AS valence, e.agency AS agency,
                       e.description AS description,
                       r.era AS era, r.year AS year, r.passage_id AS passage_id
                ORDER BY r.year
                """,
                cid=character_id,
            )
            return [dict(row) for row in result]
