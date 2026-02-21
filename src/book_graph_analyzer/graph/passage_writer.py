"""Passage temporal data — Neo4j write and query helpers.

Implements the Passage Temporal Zoom model from Issue #4:

  - Write/update Passage nodes with story_era, temporal_depth_era, era_reference_count, etc.
  - Create (Passage)-[:REFERENCES_ERA {…}]->(Era) edges
  - Compute temporal_zoom per passage (depth / corpus average)
  - Query passages filtered by temporal depth
  - Visualisation helper: return depth as a node colour weight
"""

from __future__ import annotations

import statistics
from typing import Optional

from .connection import get_driver
from .temporal import ERA_ORDER, canonicalize_era, era_to_order
from ..models.passage import Passage
from ..models.era_reference import EraReference, TemporalZoomResult


# ---------------------------------------------------------------------------
# Approximate "years ago" mapping for each era, relative to Third Age 3018
# (the dominant story-time of The Lord of the Rings corpus).
#
# These are intentionally rough — they are used only for the temporal_zoom
# heuristic, not for precise timeline queries.
# ---------------------------------------------------------------------------

ERA_APPROX_YEARS_AGO: dict[str, float] = {
    "Before Time":        20_000.0,  # effectively infinite / unknowable
    "Ainulindalë":        20_000.0,
    "Years of the Lamps": 15_000.0,
    "Years of the Trees": 10_000.0,
    "First Age":           6_000.0,
    "FA":                  6_000.0,
    "Second Age":          3_400.0,  # SA 1–3441; midpoint ~3020 years before TA 3018
    "SA":                  3_400.0,
    "Third Age":               500.0,  # mid-TA is ~1500 years before TA 3018
    "TA":                      500.0,
    "Fourth Age":                0.0,
    "FA4":                       0.0,
}


def era_approx_years_back(era: str | None, story_year: int | None = None) -> float | None:
    """Return rough years-before-story-time for a given era.

    Args:
        era: Era name (canonical or alias).
        story_year: Passage story_year, used to refine Third Age estimates.

    Returns:
        Approximate float years, or None if era is None.
    """
    if era is None:
        return None
    canonical = canonicalize_era(era) or era
    base = ERA_APPROX_YEARS_AGO.get(era) or ERA_APPROX_YEARS_AGO.get(canonical)
    if base is None:
        return None
    # For Third Age references, we can be more precise if we have years
    if canonical == "Third Age" and story_year:
        return max(0.0, float(story_year) - 0.0)  # in-era = 0 offset by default
    return base


class PassageTemporalWriter:
    """Write and query passage temporal data in Neo4j."""

    def __init__(self, driver=None):
        self._driver = driver

    @property
    def driver(self):
        if self._driver is None:
            self._driver = get_driver()
            if self._driver is None:
                raise ConnectionError("Cannot connect to Neo4j")
        return self._driver

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def upsert_passage(self, passage: Passage) -> None:
        """Create or update a Passage node with all temporal fields.

        This is idempotent — calling it multiple times is safe.
        """
        props = {
            "id": passage.id,
            "text": passage.text,
            "book": passage.book,
            "chapter": passage.chapter,
            "chapter_num": passage.chapter_num,
            "paragraph_num": passage.paragraph_num,
            "sentence_num": passage.sentence_num,
            "char_offset": passage.char_offset,
        }

        # Optional temporal fields
        if passage.story_era is not None:
            props["story_era"] = canonicalize_era(passage.story_era) or passage.story_era
        if passage.story_year is not None:
            props["story_year"] = passage.story_year
        if passage.temporal_depth_era is not None:
            props["temporal_depth_era"] = canonicalize_era(passage.temporal_depth_era) or passage.temporal_depth_era
        if passage.temporal_depth_years_back is not None:
            props["temporal_depth_years_back"] = float(passage.temporal_depth_years_back)
        if passage.era_reference_count:
            props["era_reference_count"] = passage.era_reference_count

        # Tolkien-specific
        if passage.scene_type is not None:
            props["scene_type"] = passage.scene_type
        if passage.tolkien_register is not None:
            props["tolkien_register"] = passage.tolkien_register
        if passage.pov_character_id is not None:
            props["pov_character_id"] = passage.pov_character_id

        props["is_dialogue"] = passage.is_dialogue
        if passage.speaker_ids:
            props["speaker_ids"] = passage.speaker_ids

        # Style metrics
        if passage.sentence_count:
            props["sentence_count"] = passage.sentence_count
        if passage.avg_sentence_length:
            props["avg_sentence_length"] = passage.avg_sentence_length
        if passage.passive_ratio:
            props["passive_ratio"] = passage.passive_ratio
        if passage.dialogue_density:
            props["dialogue_density"] = passage.dialogue_density
        if passage.archaic_word_count:
            props["archaic_word_count"] = passage.archaic_word_count

        with self.driver.session() as session:
            session.run(
                "MERGE (p:Passage {id: $id}) SET p += $props",
                id=passage.id,
                props=props,
            )

    def upsert_era_reference(self, ref: EraReference) -> None:
        """Create or update a REFERENCES_ERA edge in Neo4j.

        Ensures the Era node exists (MERGE) and creates the relationship
        from the Passage node.
        """
        era_canonical = canonicalize_era(ref.era) or ref.era
        props = ref.to_neo4j_props()
        props["era"] = era_canonical

        with self.driver.session() as session:
            session.run(
                """
                MERGE (e:Era {name: $era})
                WITH e
                MATCH (p:Passage {id: $passage_id})
                MERGE (p)-[r:REFERENCES_ERA {era: $era}]->(e)
                SET r += $props
                """,
                era=era_canonical,
                passage_id=ref.passage_id,
                props=props,
            )

    def upsert_passage_with_references(
        self,
        passage: Passage,
        references: list[EraReference],
    ) -> None:
        """Convenience: write a passage and all its era references together.

        Also auto-fills temporal_depth_era / temporal_depth_years_back
        from the references list if not already set on the passage.
        """
        # Auto-derive temporal depth fields from references
        if references:
            # Find the oldest referenced era
            oldest_era: str | None = None
            oldest_order = 9999
            max_years_back: float | None = None

            for ref in references:
                era_canonical = canonicalize_era(ref.era) or ref.era
                order = era_to_order(era_canonical)
                if order < oldest_order:
                    oldest_order = order
                    oldest_era = era_canonical
                # Track max years_before_story_time
                if ref.years_before_story_time is not None:
                    if max_years_back is None or ref.years_before_story_time > max_years_back:
                        max_years_back = ref.years_before_story_time

            # Fill in auto-derived years_back using our approximation table
            if max_years_back is None and oldest_era is not None:
                max_years_back = era_approx_years_back(oldest_era, passage.story_year)

            if passage.temporal_depth_era is None and oldest_era is not None:
                passage = passage.model_copy(update={
                    "temporal_depth_era": oldest_era,
                    "temporal_depth_years_back": max_years_back,
                    "era_reference_count": len(set(
                        canonicalize_era(r.era) or r.era for r in references
                    )),
                })
            elif passage.temporal_depth_years_back is None and max_years_back is not None:
                passage = passage.model_copy(update={
                    "temporal_depth_years_back": max_years_back,
                    "era_reference_count": len(set(
                        canonicalize_era(r.era) or r.era for r in references
                    )),
                })

        self.upsert_passage(passage)
        for ref in references:
            self.upsert_era_reference(ref)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def compute_corpus_avg_depth(self) -> float | None:
        """Compute average temporal_depth_years_back across all passages in Neo4j.

        Returns None if there are no passages with temporal depth data.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Passage)
                WHERE p.temporal_depth_years_back IS NOT NULL
                RETURN avg(p.temporal_depth_years_back) AS avg_depth, count(p) AS cnt
                """
            )
            row = result.single()
            if not row or row["cnt"] == 0:
                return None
            return float(row["avg_depth"])

    def query_passages_by_temporal_depth(
        self,
        min_era: str,
        limit: int = 50,
        include_references: bool = True,
    ) -> list[TemporalZoomResult]:
        """Return passages whose temporal depth reaches at least `min_era`.

        Args:
            min_era: Era name — passages must reference this era or earlier.
                     e.g. 'First Age' returns passages with depth <= First Age.
            limit: Maximum number of passages to return.
            include_references: If True, fetch REFERENCES_ERA edges too.

        Returns:
            List of TemporalZoomResult, sorted deepest-first.
        """
        min_era_canonical = canonicalize_era(min_era) or min_era
        min_order = era_to_order(min_era_canonical)

        # Build a literal era-order map for Cypher
        era_order_map = (
            "{"
            + ", ".join(f'`{k}`: {v}' for k, v in ERA_ORDER.items())
            + "}"
        )

        cypher = f"""
        MATCH (p:Passage)
        WHERE p.temporal_depth_era IS NOT NULL
          AND coalesce({era_order_map}[p.temporal_depth_era], 99) <= $min_order
        RETURN p
        ORDER BY coalesce({era_order_map}[p.temporal_depth_era], 99) ASC,
                 p.temporal_depth_years_back DESC
        LIMIT $limit
        """

        results: list[TemporalZoomResult] = []
        with self.driver.session() as session:
            rows = session.run(cypher, min_order=min_order, limit=limit)
            passage_nodes = [dict(row["p"]) for row in rows]

        # Compute corpus average for zoom metric
        corpus_avg = self.compute_corpus_avg_depth()

        for node in passage_nodes:
            pid = node.get("id", "")
            depth_years = node.get("temporal_depth_years_back")

            zoom: float | None = None
            if depth_years is not None and corpus_avg and corpus_avg > 0:
                zoom = depth_years / corpus_avg

            refs: list[EraReference] = []
            if include_references:
                refs = self._fetch_references(pid)

            results.append(TemporalZoomResult(
                passage_id=pid,
                passage_text=node.get("text", ""),
                story_era=node.get("story_era"),
                story_year=node.get("story_year"),
                temporal_depth_era=node.get("temporal_depth_era"),
                temporal_depth_years_back=depth_years,
                era_reference_count=node.get("era_reference_count", 0),
                temporal_zoom=zoom,
                references=refs,
            ))

        return results

    def _fetch_references(self, passage_id: str) -> list[EraReference]:
        """Fetch all REFERENCES_ERA edges for a passage from Neo4j."""
        refs: list[EraReference] = []
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (p:Passage {id: $pid})-[r:REFERENCES_ERA]->(e:Era)
                RETURN e.name AS era, r
                """,
                pid=passage_id,
            )
            for row in rows:
                rel_props = dict(row["r"])
                refs.append(EraReference(
                    passage_id=passage_id,
                    era=row["era"],
                    reference_type=rel_props.get("reference_type", "mentions"),
                    entity_referenced_id=rel_props.get("entity_referenced_id"),
                    event_referenced_id=rel_props.get("event_referenced_id"),
                    years_before_story_time=rel_props.get("years_before_story_time"),
                ))
        return refs

    def get_passage_temporal_zoom(self, passage_id: str) -> TemporalZoomResult | None:
        """Compute the temporal zoom score for a single passage."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (p:Passage {id: $pid}) RETURN p",
                pid=passage_id,
            )
            row = result.single()
            if not row:
                return None
            node = dict(row["p"])

        corpus_avg = self.compute_corpus_avg_depth()
        depth_years = node.get("temporal_depth_years_back")
        zoom: float | None = None
        if depth_years is not None and corpus_avg and corpus_avg > 0:
            zoom = depth_years / corpus_avg

        refs = self._fetch_references(passage_id)

        return TemporalZoomResult(
            passage_id=passage_id,
            passage_text=node.get("text", ""),
            story_era=node.get("story_era"),
            story_year=node.get("story_year"),
            temporal_depth_era=node.get("temporal_depth_era"),
            temporal_depth_years_back=depth_years,
            era_reference_count=node.get("era_reference_count", 0),
            temporal_zoom=zoom,
            references=refs,
        )

    def temporal_depth_visualization_color(self, passage: Passage) -> str:
        """Return a hex colour for visualisation: darker = older era.

        Suitable for passing to graph visualisation libraries.
        Returns a hex colour string like '#1a237e' (deep blue = ancient)
        or '#e3f2fd' (pale blue = recent).
        """
        era = passage.temporal_depth_era
        if era is None:
            return "#f5f5f5"  # grey = no temporal data

        order = era_to_order(era)
        # order 0 (Before Time) = darkest, order 6 (Fourth Age) = lightest
        # Map to a 0..1 darkness scale
        max_order = 6
        darkness = max(0.0, 1.0 - (order / max_order))

        # Interpolate from pale (#e3f2fd) to deep blue (#0d47a1)
        light = (227, 242, 253)
        dark = (13, 71, 161)

        r = int(light[0] + (dark[0] - light[0]) * darkness)
        g = int(light[1] + (dark[1] - light[1]) * darkness)
        b = int(light[2] + (dark[2] - light[2]) * darkness)
        return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Pure-Python temporal zoom calculation (no Neo4j needed)
# ---------------------------------------------------------------------------

def compute_temporal_zoom_batch(
    passages: list[Passage],
) -> dict[str, float]:
    """Compute temporal_zoom for a batch of passages without Neo4j.

    temporal_zoom = passage.temporal_depth_years_back / avg(corpus)

    Args:
        passages: List of Passage objects (must have temporal_depth_years_back set).

    Returns:
        Dict mapping passage.id -> temporal_zoom float.
        Passages without temporal_depth_years_back are omitted.
    """
    depths = [
        p.temporal_depth_years_back
        for p in passages
        if p.temporal_depth_years_back is not None
    ]
    if not depths:
        return {}

    avg_depth = statistics.mean(depths)
    if avg_depth == 0:
        return {p.id: 0.0 for p in passages if p.temporal_depth_years_back is not None}

    return {
        p.id: p.temporal_depth_years_back / avg_depth
        for p in passages
        if p.temporal_depth_years_back is not None
    }
