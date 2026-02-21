"""Emotional arc models — EmotionalState, ArcCheckpoint, CharacterArc, RelationshipSentiment.

Represents the canonical emotional trajectory of characters over story time.
This is part of the lore contract — generating a character in the wrong
emotional state at a given story moment is a lore violation.

Node: (:EmotionalState { valence, agency, dominant_register, description })
Edge: (Character)-[:FELT { era, year, passage_id, toward_entity_id, triggered_by }]->(EmotionalState)
Edge extension: (Character)-[:KNOWS { sentiment, valence, valence_trajectory, ... }]->(Character)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Tolkien emotional registers
# ---------------------------------------------------------------------------

class TolkienRegister(str, Enum):
    """Specific emotional/narrative registers found in Tolkien's prose."""
    ELEGIAC        = "elegiac"        # Sorrowful, lamenting; things passing away
    EUCATASTROPHIC = "eucatastrophic" # Sudden joyous turn, unexpected salvation
    DREAD          = "dread"          # Deep fear, terror of the dark
    WONDER         = "wonder"         # Awe, marveling at beauty or strangeness
    COZY           = "cozy"           # Domestic comfort, Shire-warmth, safety
    RESOLUTE       = "resolute"       # Grim determination, stoic endurance
    BURDEN         = "burden"         # Crushing weight, unwilled duty
    GRIEF          = "grief"          # Active mourning, loss
    HOPE           = "hope"           # Forward-looking trust despite adversity
    RAGE           = "rage"           # Anger, fierce combat-fury
    PITY           = "pity"           # Compassion, merciful tenderness
    TRANSCENDENT   = "transcendent"   # Mystical, beyond ordinary feeling


# Register valence / agency anchors (for validation heuristics)
# Tuple: (valence_center, agency_center)
REGISTER_ANCHORS: dict[str, tuple[float, float]] = {
    TolkienRegister.ELEGIAC:        (-0.3,  0.0),
    TolkienRegister.EUCATASTROPHIC: ( 0.9,  0.5),
    TolkienRegister.DREAD:          (-0.8, -0.5),
    TolkienRegister.WONDER:         ( 0.7,  0.3),
    TolkienRegister.COZY:           ( 0.8,  0.6),
    TolkienRegister.RESOLUTE:       ( 0.1,  0.7),
    TolkienRegister.BURDEN:         (-0.5, -0.6),
    TolkienRegister.GRIEF:          (-0.6, -0.2),
    TolkienRegister.HOPE:           ( 0.7,  0.4),
    TolkienRegister.RAGE:           (-0.2,  0.8),
    TolkienRegister.PITY:           ( 0.3,  0.2),
    TolkienRegister.TRANSCENDENT:   ( 0.6,  0.1),
}


# ---------------------------------------------------------------------------
# Relationship sentiment values
# ---------------------------------------------------------------------------

class RelationshipSentiment(str, Enum):
    """How one character feels about another in a relational edge."""
    ALLY    = "ally"    # Positive cooperation, shared purpose
    ENEMY   = "enemy"   # Active opposition, hostility
    WARY    = "wary"    # Cautious, not trusting, watching
    LOYAL   = "loyal"   # Deep personal devotion
    LOVE    = "love"    # Romantic or deep familial/companionate love
    PITY    = "pity"    # Compassion mixed with sorrow
    FEAR    = "fear"    # Driven by terror of the other
    RESPECT = "respect" # Admiration without full trust or alliance
    GRIEF   = "grief"   # Sorrow centered on the other (loss, mourning)


# Sentiment → rough valence
SENTIMENT_VALENCE: dict[str, float] = {
    RelationshipSentiment.ALLY:    0.7,
    RelationshipSentiment.ENEMY:  -0.8,
    RelationshipSentiment.WARY:   -0.2,
    RelationshipSentiment.LOYAL:   0.9,
    RelationshipSentiment.LOVE:    0.95,
    RelationshipSentiment.PITY:    0.2,
    RelationshipSentiment.FEAR:   -0.7,
    RelationshipSentiment.RESPECT: 0.5,
    RelationshipSentiment.GRIEF:  -0.3,
}


# ---------------------------------------------------------------------------
# EmotionalState
# ---------------------------------------------------------------------------

@dataclass
class EmotionalState:
    """A snapshot of a character's emotional state at a point in story time.

    Stored as (:EmotionalState) in Neo4j.
    Connected via (Character)-[:FELT { ... }]->(EmotionalState).
    """
    id: str
    valence: float                         # -1.0 (despair/dread) to 1.0 (joy/hope)
    agency: float                          # -1.0 (powerless/burdened) to 1.0 (in command/free)
    dominant_register: str                 # TolkienRegister value
    description: str = ""                  # Human-readable e.g. "grim determination under crushing burden"

    def to_neo4j_props(self) -> dict:
        return {
            "id": self.id,
            "valence": round(self.valence, 3),
            "agency": round(self.agency, 3),
            "dominant_register": self.dominant_register,
            "description": self.description,
        }

    def to_dict(self) -> dict:
        return self.to_neo4j_props()

    @classmethod
    def from_dict(cls, d: dict) -> "EmotionalState":
        return cls(
            id=d.get("id", ""),
            valence=float(d.get("valence", 0.0)),
            agency=float(d.get("agency", 0.0)),
            dominant_register=d.get("dominant_register", TolkienRegister.RESOLUTE),
            description=d.get("description", ""),
        )

    def distance_from(self, other: "EmotionalState") -> float:
        """Euclidean distance in (valence, agency) space."""
        return ((self.valence - other.valence) ** 2 + (self.agency - other.agency) ** 2) ** 0.5

    def compatible_with(self, other: "EmotionalState", tolerance: float = 0.4) -> bool:
        """True if this state is within tolerance of another in (valence, agency) space."""
        return self.distance_from(other) <= tolerance

    def is_violation(self, expected: "EmotionalState", hard_threshold: float = 0.6) -> bool:
        """True if this state is far enough from expected to constitute a lore violation."""
        return self.distance_from(expected) > hard_threshold


# ---------------------------------------------------------------------------
# FELT edge (arc entry)
# ---------------------------------------------------------------------------

@dataclass
class FeltEdge:
    """Represents a (Character)-[:FELT { ... }]->(EmotionalState) edge."""

    character_id: str
    emotional_state_id: str
    era: str
    year: Optional[int] = None
    passage_id: Optional[str] = None
    toward_entity_id: Optional[str] = None   # How they feel ABOUT a specific entity
    triggered_by_event_id: Optional[str] = None

    def to_neo4j_props(self) -> dict:
        props: dict = {"era": self.era}
        if self.year is not None:
            props["year"] = self.year
        if self.passage_id:
            props["passage_id"] = self.passage_id
        if self.toward_entity_id:
            props["toward_entity_id"] = self.toward_entity_id
        if self.triggered_by_event_id:
            props["triggered_by_event_id"] = self.triggered_by_event_id
        return props


# ---------------------------------------------------------------------------
# ArcCheckpoint — a named story moment in a character's arc
# ---------------------------------------------------------------------------

@dataclass
class ArcCheckpoint:
    """A canonical emotional checkpoint in a character's story arc.

    Represents the expected emotional state at a specific narrative moment.
    Used as a validation target — generated scenes must not violate these.
    """
    label: str                             # e.g. 'near_mount_doom'
    story_year: int                        # Approximate story year (TA)
    story_year_end: Optional[int] = None   # Inclusive range end (for spans)
    emotional_state: Optional[EmotionalState] = None
    description: str = ""                  # Why the character feels this way
    # How strictly this must be respected
    hardness: str = "SOFT"                 # 'HARD' | 'SOFT'
    # Keywords that would describe VALID states at this moment
    valid_registers: list[str] = field(default_factory=list)
    # Keywords that would describe INVALID states
    invalid_registers: list[str] = field(default_factory=list)

    def covers_year(self, year: int) -> bool:
        """True if this checkpoint applies to the given story year."""
        if self.story_year_end is not None:
            return self.story_year <= year <= self.story_year_end
        # Allow ±15 years tolerance for single-year checkpoints
        return abs(year - self.story_year) <= 15

    def is_valid_register(self, register: str) -> bool:
        """True if the given register is valid for this checkpoint."""
        if self.valid_registers and register not in self.valid_registers:
            return False
        if register in self.invalid_registers:
            return False
        return True


# ---------------------------------------------------------------------------
# CharacterArc — full emotional arc for one character
# ---------------------------------------------------------------------------

@dataclass
class CharacterArc:
    """The complete canonical emotional arc of a character."""

    character_id: str
    character_name: str
    checkpoints: list[ArcCheckpoint] = field(default_factory=list)

    def get_checkpoint_for_year(self, year: int) -> Optional[ArcCheckpoint]:
        """Return the most specific checkpoint covering the given year."""
        candidates = [cp for cp in self.checkpoints if cp.covers_year(year)]
        if not candidates:
            return None
        # Prefer checkpoints with narrower year ranges (more specific)
        candidates.sort(
            key=lambda cp: (
                abs(cp.story_year - year),
                (cp.story_year_end - cp.story_year) if cp.story_year_end else 0,
            )
        )
        return candidates[0]

    def validate_state(
        self,
        proposed_register: str,
        story_year: int,
    ) -> tuple[bool, str]:
        """Validate a proposed emotional register against the canonical arc.

        Returns:
            (is_valid, explanation) — is_valid=False means lore violation.
        """
        checkpoint = self.get_checkpoint_for_year(story_year)
        if not checkpoint:
            return True, f"No arc constraint for {self.character_name} at TA {story_year}"

        if checkpoint.is_valid_register(proposed_register):
            return True, (
                f"{self.character_name} at TA {story_year}: "
                f"'{proposed_register}' is consistent with '{checkpoint.label}'"
            )

        explanation = (
            f"VIOLATION: {self.character_name} at TA {story_year} "
            f"(checkpoint: '{checkpoint.label}') — "
            f"'{proposed_register}' is incompatible. "
            f"Expected: {checkpoint.valid_registers}. "
            f"Explicitly invalid: {checkpoint.invalid_registers}. "
            f"Context: {checkpoint.description}"
        )
        return False, explanation


# ---------------------------------------------------------------------------
# RelationalSentimentEdge — KNOWS extension
# ---------------------------------------------------------------------------

@dataclass
class RelationalSentimentEdge:
    """Represents sentiment data on a (Character)-[:KNOWS]->(Character) edge.

    Sentiment is asymmetric: how A feels about B ≠ how B feels about A.
    """
    from_character_id: str
    to_character_id: str
    sentiment: str               # RelationshipSentiment value
    valence: float               # -1.0 to 1.0
    valence_trajectory: str      # 'improving' | 'deteriorating' | 'stable' | 'volatile'
    era: Optional[str] = None
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    source_passage_ids: list[str] = field(default_factory=list)

    def to_neo4j_props(self) -> dict:
        props: dict = {
            "sentiment": self.sentiment,
            "valence": round(self.valence, 3),
            "valence_trajectory": self.valence_trajectory,
            "source_passage_ids": self.source_passage_ids,
        }
        if self.era:
            props["era"] = self.era
        if self.year_start is not None:
            props["year_start"] = self.year_start
        if self.year_end is not None:
            props["year_end"] = self.year_end
        return props
