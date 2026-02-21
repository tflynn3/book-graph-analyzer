"""Detect timeline conflicts: overlapping events, infeasible travel."""

from __future__ import annotations

import uuid
from collections import defaultdict

from ..graph.temporal import era_to_order
from .models import (
    ConflictType, LocationEdge, LocationNode,
    NormalizedTime, SpatiotemporalEvent, TimelineConflict,
)

DEFAULT_TRAVEL_SPEED = 1.0


class ConflictDetector:
    """Detect spatiotemporal inconsistencies in a set of events."""

    def __init__(
        self,
        locations: dict[str, LocationNode] | None = None,
        edges: list[LocationEdge] | None = None,
        travel_speed: float = DEFAULT_TRAVEL_SPEED,
    ):
        self.locations = locations or {}
        self.edges = edges or []
        self.travel_speed = travel_speed
        self._travel_times: dict[tuple[str, str], float] = {}
        for edge in self.edges:
            self._travel_times[(edge.source_id, edge.target_id)] = edge.travel_days
            if edge.bidirectional:
                self._travel_times[(edge.target_id, edge.source_id)] = edge.travel_days

    def get_travel_days(self, loc_a_id: str, loc_b_id: str) -> float | None:
        if loc_a_id == loc_b_id:
            return 0.0
        if (loc_a_id, loc_b_id) in self._travel_times:
            return self._travel_times[(loc_a_id, loc_b_id)]
        loc_a = self.locations.get(loc_a_id)
        loc_b = self.locations.get(loc_b_id)
        if loc_a and loc_b:
            dist = loc_a.distance_to(loc_b)
            return dist / self.travel_speed if self.travel_speed > 0 else None
        return None

    def detect_conflicts(self, events: list[SpatiotemporalEvent]) -> list[TimelineConflict]:
        conflicts: list[TimelineConflict] = []
        conflicts.extend(self._detect_temporal_overlaps(events))
        conflicts.extend(self._detect_travel_infeasibility(events))
        conflicts.sort(key=lambda c: -c.confidence)
        return conflicts

    def _detect_temporal_overlaps(self, events: list[SpatiotemporalEvent]) -> list[TimelineConflict]:
        conflicts = []
        by_entity: dict[str, list[SpatiotemporalEvent]] = defaultdict(list)
        for event in events:
            by_entity[event.entity_id].append(event)

        for entity_id, entity_events in by_entity.items():
            for i in range(len(entity_events)):
                for j in range(i + 1, len(entity_events)):
                    a, b = entity_events[i], entity_events[j]
                    if a.location_id and b.location_id and a.location_id == b.location_id:
                        continue
                    if a.location_name and b.location_name and a.location_name == b.location_name:
                        continue
                    if a.time.overlaps(b.time):
                        loc_a = a.location_name or a.location_id or "unknown"
                        loc_b = b.location_name or b.location_id or "unknown"
                        if loc_a == "unknown" or loc_b == "unknown":
                            continue
                        confidence = min(a.time.confidence, b.time.confidence) * 0.8
                        conflicts.append(TimelineConflict(
                            id=f"overlap_{uuid.uuid4().hex[:8]}",
                            conflict_type=ConflictType.TEMPORAL_OVERLAP,
                            severity="error" if confidence > 0.6 else "warning",
                            description=(
                                f"{a.entity_name or entity_id} appears at both "
                                f"'{loc_a}' and '{loc_b}' during overlapping time "
                                f"({_time_desc(a.time)} vs {_time_desc(b.time)})"
                            ),
                            event_a_id=a.id, event_b_id=b.id, entity_id=entity_id,
                            suggestion=f"Adjust timeline bounds or add travel event between {loc_a} and {loc_b}",
                            confidence=confidence,
                        ))
        return conflicts

    def _detect_travel_infeasibility(self, events: list[SpatiotemporalEvent]) -> list[TimelineConflict]:
        conflicts = []
        by_entity: dict[str, list[SpatiotemporalEvent]] = defaultdict(list)
        for event in events:
            if event.time.midpoint is not None and event.location_id:
                by_entity[event.entity_id].append(event)

        for entity_id, entity_events in by_entity.items():
            sorted_events = sorted(
                entity_events,
                key=lambda e: (era_to_order(e.time.era), e.time.midpoint or 0),
            )
            for i in range(len(sorted_events) - 1):
                a, b = sorted_events[i], sorted_events[i + 1]
                if a.location_id == b.location_id:
                    continue
                if a.time.era == b.time.era and a.time.midpoint and b.time.midpoint:
                    available_days = abs(b.time.midpoint - a.time.midpoint) * 365
                else:
                    continue
                travel_days = self.get_travel_days(a.location_id, b.location_id)
                if travel_days is None:
                    continue
                if travel_days > available_days and available_days > 0:
                    confidence = min(a.time.confidence, b.time.confidence) * 0.7
                    conflicts.append(TimelineConflict(
                        id=f"travel_{uuid.uuid4().hex[:8]}",
                        conflict_type=ConflictType.TRAVEL_INFEASIBLE,
                        severity="warning",
                        description=(
                            f"{a.entity_name or entity_id} travels from "
                            f"'{a.location_name or a.location_id}' to "
                            f"'{b.location_name or b.location_id}' in "
                            f"~{available_days:.0f} days, but minimum "
                            f"travel time is ~{travel_days:.0f} days"
                        ),
                        event_a_id=a.id, event_b_id=b.id, entity_id=entity_id,
                        suggestion=(
                            f"Check timeline: need ≥{travel_days:.0f} days between "
                            f"events, only {available_days:.0f} available"
                        ),
                        confidence=confidence,
                    ))
        return conflicts


def _time_desc(t: NormalizedTime) -> str:
    parts = []
    if t.era:
        parts.append(t.era)
    if t.year_start is not None:
        if t.year_start == t.year_end:
            parts.append(str(t.year_start))
        elif t.year_end is not None:
            parts.append(f"{t.year_start}-{t.year_end}")
        else:
            parts.append(f"~{t.year_start}")
    return " ".join(parts) if parts else "unknown time"
