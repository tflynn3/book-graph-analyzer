"""EraReference model — represents a REFERENCES_ERA edge in the knowledge graph.

A Passage can simultaneously reference multiple eras.  For example, a single
dialogue in The Lord of the Rings might:
  - Occur in Third Age 3018 (story-time)
  - Mention the forging of the Rings (Second Age 1600)
  - Allude to the Music of the Ainur (Before Time)

Each such reference is stored as a separate REFERENCES_ERA edge in Neo4j,
keyed by (passage_id, era).
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel


# Valid reference types — what *kind* of temporal reference is this?
ReferenceType = Literal["mentions", "quotes", "alludes_to", "sets_scene_in"]


class EraReference(BaseModel):
    """Represents a directed temporal reference from a Passage to an Era node.

    Stored as a REFERENCES_ERA relationship in Neo4j:
        (Passage)-[:REFERENCES_ERA {...}]->(Era)
    """

    # The passage that contains the reference
    passage_id: str

    # The target era being referenced
    era: str

    # How the passage references the era
    reference_type: ReferenceType = "mentions"

    # Optional: which specific entity from that era is referenced
    entity_referenced_id: Optional[str] = None

    # Optional: which specific event from that era is referenced
    event_referenced_id: Optional[str] = None

    # How far back this reference reaches, in approximate years before
    # the passage's story_year.  None if unknown.
    years_before_story_time: Optional[float] = None

    def to_neo4j_props(self) -> dict:
        """Serialise to a dict suitable for Neo4j relationship properties."""
        props: dict = {"reference_type": self.reference_type}
        if self.entity_referenced_id is not None:
            props["entity_referenced_id"] = self.entity_referenced_id
        if self.event_referenced_id is not None:
            props["event_referenced_id"] = self.event_referenced_id
        if self.years_before_story_time is not None:
            props["years_before_story_time"] = float(self.years_before_story_time)
        return props


class TemporalZoomResult(BaseModel):
    """Result of a temporal-zoom query on a single passage."""

    passage_id: str
    passage_text: str
    story_era: Optional[str] = None
    story_year: Optional[int] = None
    temporal_depth_era: Optional[str] = None
    temporal_depth_years_back: Optional[float] = None
    era_reference_count: int = 0
    temporal_zoom: Optional[float] = None  # passage depth / corpus average
    references: list[EraReference] = []

    def summary(self) -> str:
        lines = [
            f"Passage: {self.passage_id}",
            f'  "{self.passage_text[:120]}..."' if len(self.passage_text) > 120 else f'  "{self.passage_text}"',
        ]
        if self.story_era:
            yr = f" {self.story_year}" if self.story_year else ""
            lines.append(f"  Story-time:    {self.story_era}{yr}")
        if self.temporal_depth_era:
            yb = f" (~{self.temporal_depth_years_back:,.0f} yrs back)" if self.temporal_depth_years_back else ""
            lines.append(f"  Temporal depth: {self.temporal_depth_era}{yb}")
        if self.era_reference_count:
            lines.append(f"  Era refs: {self.era_reference_count}")
        if self.temporal_zoom is not None:
            lines.append(f"  Temporal zoom: {self.temporal_zoom:.2f}x corpus average")
        return "\n".join(lines)
