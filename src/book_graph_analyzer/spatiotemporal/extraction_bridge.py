"""Bridge between lore event extraction and spatiotemporal normalization.

Converts raw extracted Events (from lore.events) into SpatiotemporalEvents
with normalized time, tracking extraction-vs-normalized confidence delta.

This is the integration path: extract -> normalize -> detect conflicts.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from ..graph.temporal import canonicalize_era
from ..lore.events import Event
from ..models.worldbuilding import infer_editorial_layer
from .models import NormalizedTime, SpatiotemporalEvent
from .normalizer import TimeNormalizer
from .grounding import (
    METRICS_VERSION,
    backfill_temporal_grounding,
    compute_temporal_grounding_metrics,
)


def _era_display_name(era_val: str) -> str:
    """Convert era enum values like 'third_age' to display names like 'Third Age'."""
    from ..graph.temporal import ERA_ORDER
    # If it's already a recognized canonical name, return it
    if era_val in ERA_ORDER:
        return era_val
    # Convert snake_case to Title Case and check
    title = era_val.replace("_", " ").title()
    if title in ERA_ORDER:
        return title
    # Try canonicalize for abbreviations
    canonical = canonicalize_era(era_val)
    if canonical and canonical in ERA_ORDER:
        return canonical
    return title


@dataclass
class NormalizationResult:
    """Result of normalizing a single extracted event."""

    event: SpatiotemporalEvent
    extraction_confidence: float
    normalization_confidence: float
    raw_era_text: str | None = None
    normalized_era: str | None = None
    era_changed: bool = False

    @property
    def confidence_delta(self) -> float:
        """Difference between extraction and normalization confidence.

        Positive means extraction was more confident than normalization
        could support; negative means normalization increased confidence.
        """
        return self.extraction_confidence - self.normalization_confidence

    @property
    def confidence_category(self) -> str:
        """Categorize the confidence relationship."""
        delta = abs(self.confidence_delta)
        if delta < 0.1:
            return "aligned"
        elif self.confidence_delta > 0:
            return "extraction_overconfident"
        else:
            return "normalization_boosted"

    def to_dict(self) -> dict:
        return {
            "event": self.event.to_dict(),
            "extraction_confidence": self.extraction_confidence,
            "normalization_confidence": self.normalization_confidence,
            "confidence_delta": self.confidence_delta,
            "confidence_category": self.confidence_category,
            "raw_era_text": self.raw_era_text,
            "normalized_era": self.normalized_era,
            "era_changed": self.era_changed,
        }


@dataclass
class BridgeReport:
    """Aggregate report from bridging a batch of extracted events."""

    results: list[NormalizationResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def aligned_count(self) -> int:
        return sum(1 for r in self.results if r.confidence_category == "aligned")

    @property
    def overconfident_count(self) -> int:
        return sum(1 for r in self.results
                   if r.confidence_category == "extraction_overconfident")

    @property
    def boosted_count(self) -> int:
        return sum(1 for r in self.results
                   if r.confidence_category == "normalization_boosted")

    @property
    def era_changed_count(self) -> int:
        return sum(1 for r in self.results if r.era_changed)

    @property
    def avg_confidence_delta(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.confidence_delta for r in self.results) / len(self.results)

    @property
    def events(self) -> list[SpatiotemporalEvent]:
        return [r.event for r in self.results]

    def summary_text(self) -> str:
        if not self.results:
            return "No events processed."
        lines = [
            f"Events bridged: {self.total}",
            f"  Confidence aligned:       {self.aligned_count}",
            f"  Extraction overconfident:  {self.overconfident_count}",
            f"  Normalization boosted:     {self.boosted_count}",
            f"  Era changed during norm:   {self.era_changed_count}",
            f"  Avg confidence delta:      {self.avg_confidence_delta:+.3f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        events = self.events
        by_book: dict[str, list[SpatiotemporalEvent]] = {}
        for ev in events:
            by_book.setdefault(ev.source_book or "unknown", []).append(ev)

        return {
            "metrics_version": METRICS_VERSION,
            "total": self.total,
            "aligned": self.aligned_count,
            "overconfident": self.overconfident_count,
            "boosted": self.boosted_count,
            "era_changed": self.era_changed_count,
            "avg_confidence_delta": self.avg_confidence_delta,
            "temporal_grounding": compute_temporal_grounding_metrics(events).to_dict(),
            "temporal_grounding_by_book": {
                book: compute_temporal_grounding_metrics(book_events).to_dict()
                for book, book_events in by_book.items()
            },
            "results": [r.to_dict() for r in self.results],
        }


class ExtractionBridge:
    """Bridge extracted Events to SpatiotemporalEvents with normalization."""

    def __init__(self, normalizer: TimeNormalizer | None = None):
        self.normalizer = normalizer or TimeNormalizer()

    def bridge_event(self, event: Event, source_book: str | None = None) -> NormalizationResult:
        """Convert a single extracted Event to a SpatiotemporalEvent."""
        extraction_conf = event.confidence

        # Get raw era text from event
        raw_era_text: str | None = None
        if event.era is not None:
            raw_era_text = event.era.value if hasattr(event.era, "value") else str(event.era)
        elif event.year_text:
            raw_era_text = event.year_text

        # Convert era enum value to display name for normalizer
        era_display: str | None = None
        if raw_era_text:
            era_display = _era_display_name(raw_era_text)

        # Normalize time
        parsed_year: int | None = None
        if isinstance(event.year, int):
            parsed_year = event.year
        elif isinstance(event.year, str):
            try:
                parsed_year = int(event.year)
            except ValueError:
                parsed_year = None

        if event.year_text:
            norm_time = self.normalizer.normalize(event.year_text)
        elif parsed_year is None and event.year is not None:
            norm_time = self.normalizer.normalize(str(event.year))
        else:
            norm_time = self.normalizer.normalize_event_time(
                raw_text=event.year_text,
                era=era_display,
                year=parsed_year,
            )

        # Check if era changed during normalization
        era_changed = False
        if raw_era_text and norm_time.era:
            era_changed = raw_era_text.lower().replace(" ", "") != norm_time.era.lower().replace(" ", "")

        source_name = source_book or event.source_book or None
        layer = infer_editorial_layer(source_name) if source_name else None

        # Build SpatiotemporalEvent
        st_event = SpatiotemporalEvent(
            id=f"st_{event.id}",
            entity_id=event.agent or "unknown",
            entity_name=event.agent,
            location_name=event.patient,  # best approximation from event
            time=norm_time,
            description=event.description,
            event_type="extracted",
            source_book=source_name,
            source_passage_id=None,
            source_id=getattr(layer, "source_id", None),
            structural_stratum=getattr(getattr(layer, "author_period", None), "value", None),
            editorial_status=getattr(getattr(layer, "editorial_status", None), "value", None),
            source_authority_weight=float(getattr(layer, "authority_weight", 1.0)) if layer else None,
        )

        return NormalizationResult(
            event=st_event,
            extraction_confidence=extraction_conf,
            normalization_confidence=norm_time.confidence,
            raw_era_text=raw_era_text,
            normalized_era=norm_time.era,
            era_changed=era_changed,
        )

    def bridge_events(
        self,
        events: list[Event],
        source_book: str | None = None,
        apply_backfill: bool = True,
    ) -> BridgeReport:
        """Convert a batch of extracted Events."""
        report = BridgeReport()
        for event in events:
            result = self.bridge_event(event, source_book=source_book)
            report.results.append(result)

        if apply_backfill:
            backfill_temporal_grounding(report.events)
        return report
