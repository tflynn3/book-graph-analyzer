"""Models for lore-depth extraction (Issue #50)."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class LoreArtifactType(str, Enum):
    """First-class artifact categories for deep lore references."""

    SONG = "song"
    POEM = "poem"
    ARTIFACT = "artifact"


class LoreArtifact(BaseModel):
    """A lore artifact mention extracted from text."""

    id: str
    name: str
    artifact_type: LoreArtifactType
    description: str | None = None
    source_book: str | None = None
    passage_id: str | None = None
    referenced_entities: list[str] = Field(default_factory=list)
    confidence: float = 0.7


class ReferenceCandidate(BaseModel):
    """Candidate canonical entity for an unresolved reference mention."""

    canonical_id: str
    surface: str | None = None
    confidence: float = 0.0
    source: str = "resolver"


class BrokenReference(BaseModel):
    """A likely unresolved/broken reference needing author review."""

    id: str
    mention_text: str
    context_text: str | None = None
    context_before: str | None = None
    context_after: str | None = None
    expected_type: str | None = None
    source_book: str | None = None
    passage_id: str | None = None
    resolved_entity_id: str | None = None
    confidence: float = 0.6

    # Slice 2 additions
    candidates: list[ReferenceCandidate] = Field(default_factory=list)
    provenance_notes: list[str] = Field(default_factory=list)
    conflict_weight: float = 0.0


class LoreDepthExtractionResult(BaseModel):
    """Container for a single extraction run."""

    artifacts: list[LoreArtifact] = Field(default_factory=list)
    broken_references: list[BrokenReference] = Field(default_factory=list)

    @property
    def unresolved_queue(self) -> list[BrokenReference]:
        """Queue prioritized for downstream generation/review."""
        return sorted(
            [r for r in self.broken_references if not r.resolved_entity_id],
            key=lambda r: (r.conflict_weight + r.confidence),
            reverse=True,
        )
