"""Temporal grounding quality metrics, backfill, and gates (Issue #89)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .models import SpatiotemporalEvent

METRICS_VERSION = "temporal-grounding-v1"


@dataclass
class TemporalGroundingMetrics:
    total_events: int = 0
    grounded_events: int = 0
    era_grounded_events: int = 0
    year_or_interval_grounded_events: int = 0

    @property
    def grounded_ratio(self) -> float:
        return self.grounded_events / self.total_events if self.total_events else 0.0

    @property
    def era_ratio(self) -> float:
        return self.era_grounded_events / self.total_events if self.total_events else 0.0

    @property
    def year_or_interval_ratio(self) -> float:
        return self.year_or_interval_grounded_events / self.total_events if self.total_events else 0.0

    def to_dict(self) -> dict:
        return {
            "metrics_version": METRICS_VERSION,
            "total_events": self.total_events,
            "grounded_events": self.grounded_events,
            "era_grounded_events": self.era_grounded_events,
            "year_or_interval_grounded_events": self.year_or_interval_grounded_events,
            "grounded_ratio": round(self.grounded_ratio, 4),
            "era_ratio": round(self.era_ratio, 4),
            "year_or_interval_ratio": round(self.year_or_interval_ratio, 4),
        }


@dataclass
class TemporalGroundingGateResult:
    passed: bool
    metrics: TemporalGroundingMetrics
    min_grounded_ratio: float
    min_era_ratio: float
    min_year_or_interval_ratio: float

    @property
    def failures(self) -> list[str]:
        failures: list[str] = []
        if self.metrics.grounded_ratio < self.min_grounded_ratio:
            failures.append(
                f"grounded_ratio {self.metrics.grounded_ratio:.3f} < {self.min_grounded_ratio:.3f}"
            )
        if self.metrics.era_ratio < self.min_era_ratio:
            failures.append(f"era_ratio {self.metrics.era_ratio:.3f} < {self.min_era_ratio:.3f}")
        if self.metrics.year_or_interval_ratio < self.min_year_or_interval_ratio:
            failures.append(
                f"year_or_interval_ratio {self.metrics.year_or_interval_ratio:.3f} "
                f"< {self.min_year_or_interval_ratio:.3f}"
            )
        return failures

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "thresholds": {
                "min_grounded_ratio": self.min_grounded_ratio,
                "min_era_ratio": self.min_era_ratio,
                "min_year_or_interval_ratio": self.min_year_or_interval_ratio,
            },
            "metrics": self.metrics.to_dict(),
            "failures": self.failures,
        }


class TemporalGroundingGate:
    """Hard gate for temporal grounding quality."""

    def __init__(
        self,
        *,
        min_grounded_ratio: float = 0.85,
        min_era_ratio: float = 0.80,
        min_year_or_interval_ratio: float = 0.05,
    ):
        self.min_grounded_ratio = min_grounded_ratio
        self.min_era_ratio = min_era_ratio
        self.min_year_or_interval_ratio = min_year_or_interval_ratio

    def evaluate(self, events: list[SpatiotemporalEvent]) -> TemporalGroundingGateResult:
        metrics = compute_temporal_grounding_metrics(events)
        passed = (
            metrics.grounded_ratio >= self.min_grounded_ratio
            and metrics.era_ratio >= self.min_era_ratio
            and metrics.year_or_interval_ratio >= self.min_year_or_interval_ratio
        )
        return TemporalGroundingGateResult(
            passed=passed,
            metrics=metrics,
            min_grounded_ratio=self.min_grounded_ratio,
            min_era_ratio=self.min_era_ratio,
            min_year_or_interval_ratio=self.min_year_or_interval_ratio,
        )


def compute_temporal_grounding_metrics(events: list[SpatiotemporalEvent]) -> TemporalGroundingMetrics:
    total = len(events)
    era = 0
    year_or_interval = 0
    grounded = 0

    for ev in events:
        has_era = bool((ev.time.era or "").strip())
        has_year = ev.time.year_start is not None or ev.time.year_end is not None
        if has_era:
            era += 1
        if has_year:
            year_or_interval += 1
        if has_era or has_year:
            grounded += 1

    return TemporalGroundingMetrics(
        total_events=total,
        grounded_events=grounded,
        era_grounded_events=era,
        year_or_interval_grounded_events=year_or_interval,
    )


def _dominant_era(events: list[SpatiotemporalEvent]) -> str | None:
    eras = [e.time.era for e in events if e.time.era]
    if not eras:
        return None
    return Counter(eras).most_common(1)[0][0]


def _dominant_year(events: list[SpatiotemporalEvent]) -> int | None:
    years = [e.time.year_start for e in events if e.time.year_start is not None]
    if not years:
        return None
    return int(round(sum(years) / len(years)))


def backfill_temporal_grounding(events: list[SpatiotemporalEvent]) -> int:
    """Best-effort era/year grounding backfill.

    Strategy:
    1) per-book dominant era propagation,
    2) per-book dominant year fallback for events lacking any year interval.
    """
    by_book: dict[str, list[SpatiotemporalEvent]] = {}
    for ev in events:
        key = ev.source_book or "unknown"
        by_book.setdefault(key, []).append(ev)

    changes = 0
    for bucket in by_book.values():
        era = _dominant_era(bucket)
        yr = _dominant_year(bucket)

        for ev in bucket:
            if not ev.time.era and era:
                ev.time.era = era
                changes += 1
            if ev.time.year_start is None and ev.time.year_end is None and yr is not None:
                ev.time.year_start = yr
                ev.time.year_end = yr
                ev.time.confidence = min(0.6, ev.time.confidence)
                changes += 1
    return changes
