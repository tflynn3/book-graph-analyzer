"""Data models for the spatiotemporal engine.

All models are additive — they extend but don't break the existing schema.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ConflictType(str, Enum):
    """Types of spatiotemporal conflicts."""
    TEMPORAL_OVERLAP = "temporal_overlap"
    TRAVEL_INFEASIBLE = "travel_infeasible"
    CAUSAL_PARADOX = "causal_paradox"
    ERA_MISMATCH = "era_mismatch"


class NormalizedTime(BaseModel):
    """Normalized event time with uncertainty bounds.

    Supports both precise dates (year=3019, confidence=1.0) and
    fuzzy ranges ("sometime in the late Third Age").
    """

    era: str | None = None
    year_start: int | None = None
    year_end: int | None = None
    confidence: float = 0.5
    source_passage_id: str | None = None
    raw_text: str | None = None

    @property
    def is_precise(self) -> bool:
        return (
            self.year_start is not None
            and self.year_start == self.year_end
            and self.confidence >= 0.8
        )

    @property
    def midpoint(self) -> float | None:
        if self.year_start is not None and self.year_end is not None:
            return (self.year_start + self.year_end) / 2.0
        return self.year_start or self.year_end

    def overlaps(self, other: NormalizedTime) -> bool:
        if self.era and other.era and self.era != other.era:
            return False
        if self.year_start is None or self.year_end is None:
            return True
        if other.year_start is None or other.year_end is None:
            return True
        return self.year_start <= other.year_end and other.year_start <= self.year_end

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "era": self.era,
            "year_start": self.year_start,
            "year_end": self.year_end,
            "confidence": self.confidence,
            "source_passage_id": self.source_passage_id,
            "raw_text": self.raw_text,
        }.items() if v is not None}


class SpatiotemporalEvent(BaseModel):
    """An event anchored in both space and time."""

    id: str
    entity_id: str
    entity_name: str | None = None
    location_id: str | None = None
    location_name: str | None = None
    time: NormalizedTime = Field(default_factory=NormalizedTime)
    description: str = ""
    event_type: str = "presence"
    source_book: str | None = None
    source_passage_id: str | None = None
    structural_stratum: str | None = None
    editorial_status: str | None = None
    source_authority_weight: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "location_id": self.location_id,
            "location_name": self.location_name,
            "time": self.time.to_dict(),
            "description": self.description,
            "event_type": self.event_type,
            "source_book": self.source_book,
            "source_passage_id": self.source_passage_id,
            "structural_stratum": self.structural_stratum,
            "editorial_status": self.editorial_status,
            "source_authority_weight": self.source_authority_weight,
        }


class LocationNode(BaseModel):
    """A location in the world map with coordinates for distance estimation.

    TODO(#48): Support actual Tolkien map coordinates from canonical sources.
    """

    id: str
    name: str
    region: str | None = None
    x: float = 0.0
    y: float = 0.0
    aliases: list[str] = Field(default_factory=list)

    def distance_to(self, other: LocationNode) -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class LocationEdge(BaseModel):
    """A travel route between two locations."""

    source_id: str
    target_id: str
    travel_days: float = 1.0
    mode: str = "foot"
    difficulty: str = "normal"
    bidirectional: bool = True


class CausalLink(BaseModel):
    """A declared causal relationship: cause_event causes effect_event.

    Used by the paradox detector to find impossible causal orderings
    where the effect temporally precedes the cause.

    TODO(#48): Extract causal links from LLM event extraction pipeline.
    """

    cause_event_id: str
    effect_event_id: str
    description: str = ""
    confidence: float = 0.7


class TimelineConflict(BaseModel):
    """A detected inconsistency in the spatiotemporal graph."""

    id: str
    conflict_type: ConflictType
    severity: str = "warning"
    description: str = ""
    event_a_id: str | None = None
    event_b_id: str | None = None
    entity_id: str | None = None
    suggestion: str | None = None
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "conflict_type": self.conflict_type.value,
            "severity": self.severity,
            "description": self.description,
            "event_a_id": self.event_a_id,
            "event_b_id": self.event_b_id,
            "entity_id": self.entity_id,
            "suggestion": self.suggestion,
            "confidence": self.confidence,
        }
