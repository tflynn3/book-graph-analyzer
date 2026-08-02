"""Temporal validity utilities for the knowledge graph.

Every relationship in the graph should carry era_start / era_end / year_start / year_end
so that point-in-time queries can answer: "What did character X know at story-time T?"

Era ordering (canonical):
    Before Time < Years of the Lamps < Years of the Trees
    < First Age < Second Age < Third Age < Fourth Age < Unknown
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Era ordering — lower index = earlier in history
# ---------------------------------------------------------------------------

ERA_ORDER: dict[str, int] = {
    "Before Time":       0,
    "Ainulindalë":       0,   # alias
    "Years of the Lamps": 1,
    "Years of the Trees": 2,
    "First Age":         3,
    "FA":                3,   # alias
    "Second Age":        4,
    "SA":                4,
    "Third Age":         5,
    "TA":                5,
    "Fourth Age":        6,
    "FA4":               6,
    "Unknown":           99,
}

# Canonical display names for era aliases
ERA_CANONICAL: dict[str, str] = {
    "Ainulindalë":        "Before Time",
    "FA":                 "First Age",
    "SA":                 "Second Age",
    "TA":                 "Third Age",
    "FA4":                "Fourth Age",
}


def canonicalize_era(era: str | None) -> str | None:
    """Return the canonical display name for an era string."""
    if era is None:
        return None
    return ERA_CANONICAL.get(era, era)


def era_to_order(era: str | None) -> int:
    """Return the sort order for an era (higher = later). Unknown = 99."""
    if era is None:
        return 99
    return ERA_ORDER.get(era, 99)


def era_before_or_equal(a: str | None, b: str | None) -> bool:
    """Return True if era a comes before or is equal to era b."""
    return era_to_order(a) <= era_to_order(b)


def era_after_or_equal(a: str | None, b: str | None) -> bool:
    """Return True if era a comes after or is equal to era b."""
    return era_to_order(a) >= era_to_order(b)


# ---------------------------------------------------------------------------
# TemporalValidity dataclass — attached to every relationship
# ---------------------------------------------------------------------------

@dataclass
class TemporalValidity:
    """When a relationship is valid in the story timeline."""

    era_start: Optional[str] = None    # None = "since forever / unknown"
    era_end: Optional[str] = None      # None = "ongoing / unknown"
    year_start: Optional[int] = None   # Specific year within era_start
    year_end: Optional[int] = None     # Specific year within era_end
    source_passage_id: Optional[str] = None
    confidence: float = 1.0

    def is_valid_at(self, era: str, year: Optional[int] = None) -> bool:
        """Check if this relationship is valid at a given point in time.

        Args:
            era: The era to check (e.g. 'Third Age')
            year: Optional year within the era

        Returns:
            True if the relationship exists at this point in time.
        """
        # Check era_start bound
        if self.era_start is not None:
            if not era_after_or_equal(era, self.era_start):
                return False
            # If same era, check year
            if era == self.era_start and self.year_start is not None and year is not None:
                if year < self.year_start:
                    return False

        # Check era_end bound
        if self.era_end is not None:
            if not era_before_or_equal(era, self.era_end):
                return False
            # If same era, check year
            if era == self.era_end and self.year_end is not None and year is not None:
                if year > self.year_end:
                    return False

        return True

    def to_dict(self) -> dict:
        """Serialise to a dict for Neo4j property storage."""
        return {
            k: v for k, v in {
                "era_start": canonicalize_era(self.era_start),
                "era_end":   canonicalize_era(self.era_end),
                "year_start": self.year_start,
                "year_end":   self.year_end,
                "source_passage_id": self.source_passage_id,
                "temporal_confidence": self.confidence,
            }.items()
            if v is not None
        }

    @classmethod
    def always(cls) -> "TemporalValidity":
        """A relationship that is valid for all time (no constraints)."""
        return cls()

    @classmethod
    def from_era(cls, era: str, year: Optional[int] = None,
                 passage_id: Optional[str] = None) -> "TemporalValidity":
        """Relationship starts at this era/year and continues indefinitely."""
        return cls(era_start=era, year_start=year, source_passage_id=passage_id)

    @classmethod
    def from_dict(cls, d: dict) -> "TemporalValidity":
        """Deserialise from a Neo4j property dict."""
        return cls(
            era_start=d.get("era_start"),
            era_end=d.get("era_end"),
            year_start=d.get("year_start"),
            year_end=d.get("year_end"),
            source_passage_id=d.get("source_passage_id"),
            confidence=float(d.get("temporal_confidence", 1.0)),
        )


# ---------------------------------------------------------------------------
# Cypher helpers for point-in-time queries
# ---------------------------------------------------------------------------

# Map of era name to its integer order — injected into Cypher as a literal
_ERA_ORDER_CYPHER = (
    "{"
    + ", ".join(f"`{k}`: {v}" for k, v in ERA_ORDER.items())
    + "}"
)


def point_in_time_cypher_where(
    rel_alias: str = "r",
    era_param: str = "$era",
    year_param: str = "$year",
) -> str:
    """Return a Cypher WHERE clause fragment that filters a relationship
    by era/year validity.

    Usage:
        MATCH (a)-[r:KNOWS]->(b)
        WHERE {fragment}
        RETURN a, r, b

    Parameters bound: $era (str), $year (int, nullable)
    """
    order = _ERA_ORDER_CYPHER
    return f"""(
      {rel_alias}.era_start IS NULL OR
      coalesce({order}[{rel_alias}.era_start], 99) <= coalesce({order}[{era_param}], 99)
    ) AND (
      {rel_alias}.era_end IS NULL OR
      coalesce({order}[{rel_alias}.era_end], 99) >= coalesce({order}[{era_param}], 99)
    ) AND (
      {rel_alias}.year_start IS NULL OR {year_param} IS NULL OR
      CASE WHEN {rel_alias}.era_start = {era_param}
           THEN coalesce({rel_alias}.year_start, 0) <= coalesce({year_param}, 0)
           ELSE true END
    ) AND (
      {rel_alias}.year_end IS NULL OR {year_param} IS NULL OR
      CASE WHEN {rel_alias}.era_end = {era_param}
           THEN coalesce({rel_alias}.year_end, 999999) >= coalesce({year_param}, 999999)
           ELSE true END
    )"""
