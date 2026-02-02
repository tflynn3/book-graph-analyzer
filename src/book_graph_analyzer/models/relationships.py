"""Relationship models for the knowledge graph."""

from enum import Enum
from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    """Types of relationships between entities."""

    # Character interactions
    SPOKE_WITH = "SPOKE_WITH"
    SPOKE_TO = "SPOKE_TO"
    TRAVELED_WITH = "TRAVELED_WITH"
    FOUGHT = "FOUGHT"
    FOUGHT_AGAINST = "FOUGHT_AGAINST"
    ALLIED_WITH = "ALLIED_WITH"
    BETRAYED = "BETRAYED"
    HELPED = "HELPED"
    CAPTURED = "CAPTURED"
    FREED = "FREED"
    KILLED = "KILLED"
    MET = "MET"

    # Family/social
    PARENT_OF = "PARENT_OF"
    CHILD_OF = "CHILD_OF"
    SIBLING_OF = "SIBLING_OF"
    MARRIED_TO = "MARRIED_TO"
    FRIEND_OF = "FRIEND_OF"
    ENEMY_OF = "ENEMY_OF"
    SERVES = "SERVES"
    LEADS = "LEADS"

    # Object interactions
    POSSESSES = "POSSESSES"
    POSSESSED = "POSSESSED"
    GAVE = "GAVE"
    RECEIVED = "RECEIVED"
    FOUND = "FOUND"
    LOST = "LOST"
    CREATED = "CREATED"
    DESTROYED = "DESTROYED"
    USED = "USED"
    STOLE = "STOLE"

    # Location interactions
    TRAVELED_TO = "TRAVELED_TO"
    TRAVELED_FROM = "TRAVELED_FROM"
    LIVES_IN = "LIVES_IN"
    VISITED = "VISITED"
    ENTERED = "ENTERED"
    LEFT = "LEFT"
    RULES = "RULES"
    GUARDS = "GUARDS"

    # Place relationships
    LOCATED_IN = "LOCATED_IN"
    NEAR = "NEAR"

    # Events
    PARTICIPATED_IN = "PARTICIPATED_IN"
    WITNESSED = "WITNESSED"
    CAUSED = "CAUSED"

    # Generic
    MENTIONED_WITH = "MENTIONED_WITH"
    RELATED_TO = "RELATED_TO"


class ExtractedRelationship(BaseModel):
    """A relationship extracted from text."""

    subject_text: str
    subject_id: str | None = None
    subject_type: str | None = None

    predicate: RelationshipType
    predicate_raw: str  # The original verb/phrase

    object_text: str
    object_id: str | None = None
    object_type: str | None = None

    # Optional third party (e.g., "Bilbo gave the Ring TO Gandalf")
    indirect_object_text: str | None = None
    indirect_object_id: str | None = None
    indirect_object_type: str | None = None

    # Context
    passage_id: str
    passage_text: str
    confidence: float = 1.0
    extraction_method: str = "llm"  # llm, dependency, pattern

    def to_triple(self) -> str:
        """Return a human-readable triple."""
        subj = self.subject_id or self.subject_text
        obj = self.object_id or self.object_text
        return f"({subj})-[{self.predicate.value}]->({obj})"


class RelationshipTriple(BaseModel):
    """A resolved relationship ready for graph insertion."""

    source_id: str
    source_type: str
    target_id: str
    target_type: str
    relationship_type: RelationshipType
    properties: dict = Field(default_factory=dict)
    passage_ids: list[str] = Field(default_factory=list)
    mention_count: int = 1

    def merge_with(self, other: "RelationshipTriple") -> "RelationshipTriple":
        """Merge with another triple of the same type."""
        return RelationshipTriple(
            source_id=self.source_id,
            source_type=self.source_type,
            target_id=self.target_id,
            target_type=self.target_type,
            relationship_type=self.relationship_type,
            properties={**self.properties, **other.properties},
            passage_ids=list(set(self.passage_ids + other.passage_ids)),
            mention_count=self.mention_count + other.mention_count,
        )
