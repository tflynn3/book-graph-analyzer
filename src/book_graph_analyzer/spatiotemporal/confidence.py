"""Confidence calibration with source authority weights.

Adjusts confidence scores on events, causal links, and conflicts using
editorial source provenance (authority_weight from EditorialLayer).

Backward-compatible: when source metadata is absent, all weights default to 1.0.

TODO(#48): Integrate with ingest pipeline for automatic source tagging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .models import CausalLink, SpatiotemporalEvent, TimelineConflict

logger = logging.getLogger(__name__)

# Default authority weight when source metadata is absent
_DEFAULT_AUTHORITY = 1.0


@dataclass
class SourceAuthorityRegistry:
    """Maps source_book identifiers to authority weights.

    Weights come from EditorialLayer.authority_weight in the worldbuilding models.
    """

    weights: dict[str, float] = field(default_factory=dict)

    def get(self, source_book: str | None) -> float:
        """Get authority weight for a source. Returns default if unknown."""
        if source_book is None:
            return _DEFAULT_AUTHORITY
        return self.weights.get(source_book, _DEFAULT_AUTHORITY)

    @classmethod
    def from_editorial_layers(cls, layers: list) -> SourceAuthorityRegistry:
        """Build from a list of EditorialLayer objects.

        Accepts anything with .source_id/.source_title and .authority_weight attrs.
        Maps both source_id and source_title (lowered) for flexible lookup.
        """
        weights: dict[str, float] = {}
        for layer in layers:
            w = getattr(layer, "authority_weight", _DEFAULT_AUTHORITY)
            sid = getattr(layer, "source_id", None)
            title = getattr(layer, "source_title", None)
            if sid:
                weights[sid] = w
            if title:
                weights[title.lower()] = w
                # Also map slug form: "The Hobbit" -> "the_hobbit"
                slug = title.lower().replace(" ", "_").replace("'", "")
                weights[slug] = w
        return cls(weights=weights)

    @classmethod
    def default_tolkien(cls) -> SourceAuthorityRegistry:
        """Build from the canonical TOLKIEN_SOURCES registry."""
        from book_graph_analyzer.models.worldbuilding import TOLKIEN_SOURCES
        return cls.from_editorial_layers(TOLKIEN_SOURCES)


@dataclass
class CalibrationResult:
    """Summary of confidence calibration applied."""
    events_calibrated: int = 0
    links_calibrated: int = 0
    conflicts_calibrated: int = 0
    avg_authority_weight: float = 1.0

    def to_dict(self) -> dict:
        return {
            "events_calibrated": self.events_calibrated,
            "links_calibrated": self.links_calibrated,
            "conflicts_calibrated": self.conflicts_calibrated,
            "avg_authority_weight": round(self.avg_authority_weight, 3),
        }


def calibrate_event_confidence(
    events: list[SpatiotemporalEvent],
    registry: SourceAuthorityRegistry | None = None,
) -> CalibrationResult:
    """Adjust event time.confidence by source authority weight.

    Mutates events in-place. New confidence = original * authority_weight.
    """
    reg = registry or SourceAuthorityRegistry()
    total_weight = 0.0
    count = 0
    for ev in events:
        w = reg.get(ev.source_book)
        ev.time.confidence = round(ev.time.confidence * w, 4)
        total_weight += w
        count += 1
    return CalibrationResult(
        events_calibrated=count,
        avg_authority_weight=total_weight / count if count else 1.0,
    )


def calibrate_causal_link_confidence(
    links: list[CausalLink],
    events: list[SpatiotemporalEvent],
    registry: SourceAuthorityRegistry | None = None,
) -> CalibrationResult:
    """Adjust causal link confidence using the authority of referenced events.

    For each link, the calibrated confidence = original * avg(authority of cause, authority of effect).
    """
    reg = registry or SourceAuthorityRegistry()
    ev_map = {e.id: e for e in events}
    count = 0
    total_weight = 0.0
    for link in links:
        cause = ev_map.get(link.cause_event_id)
        effect = ev_map.get(link.effect_event_id)
        w_cause = reg.get(cause.source_book if cause else None)
        w_effect = reg.get(effect.source_book if effect else None)
        avg_w = (w_cause + w_effect) / 2.0
        link.confidence = round(link.confidence * avg_w, 4)
        total_weight += avg_w
        count += 1
    return CalibrationResult(
        links_calibrated=count,
        avg_authority_weight=total_weight / count if count else 1.0,
    )


def calibrate_conflict_confidence(
    conflicts: list[TimelineConflict],
    events: list[SpatiotemporalEvent],
    registry: SourceAuthorityRegistry | None = None,
) -> CalibrationResult:
    """Adjust conflict confidence using the authority of referenced events.

    Higher authority sources make conflicts more credible (harder to dismiss).
    """
    reg = registry or SourceAuthorityRegistry()
    ev_map = {e.id: e for e in events}
    count = 0
    total_weight = 0.0
    for conflict in conflicts:
        ev_a = ev_map.get(conflict.event_a_id or "")
        ev_b = ev_map.get(conflict.event_b_id or "")
        w_a = reg.get(ev_a.source_book if ev_a else None)
        w_b = reg.get(ev_b.source_book if ev_b else None)
        avg_w = (w_a + w_b) / 2.0
        conflict.confidence = round(conflict.confidence * avg_w, 4)
        total_weight += avg_w
        count += 1
    return CalibrationResult(
        conflicts_calibrated=count,
        avg_authority_weight=total_weight / count if count else 1.0,
    )
