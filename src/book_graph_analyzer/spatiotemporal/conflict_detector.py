"""Detect timeline conflicts: overlapping events, infeasible travel."""

from __future__ import annotations

import uuid
from collections import defaultdict

from ..graph.temporal import era_to_order
from .models import (
    CausalLink, ConflictType, LocationEdge, LocationNode,
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
        causal_links: list[CausalLink] | None = None,
    ):
        self.locations = locations or {}
        self.edges = edges or []
        self.travel_speed = travel_speed
        self.causal_links = causal_links or []
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

    def detect_conflicts(
        self,
        events: list[SpatiotemporalEvent],
        *,
        check_era_mismatches: bool = True,
        check_causal_paradoxes: bool = True,
        causal_links: list[CausalLink] | None = None,
    ) -> list[TimelineConflict]:
        conflicts: list[TimelineConflict] = []
        conflicts.extend(self._detect_temporal_overlaps(events))
        conflicts.extend(self._detect_travel_infeasibility(events))
        if check_era_mismatches:
            conflicts.extend(self._detect_era_mismatches(events))
        if check_causal_paradoxes:
            links = causal_links if causal_links is not None else self.causal_links
            if links:
                conflicts.extend(self._detect_causal_paradoxes(events, links))
        conflicts.sort(key=lambda c: -c.confidence)
        return conflicts

    def _detect_era_mismatches(self, events: list[SpatiotemporalEvent]) -> list[TimelineConflict]:
        """Detect events where an entity's claimed era contradicts other events.

        If entity X has N events in era A and 1 event in era B, and those eras
        are non-adjacent, the outlier is flagged as a likely era mismatch.
        """
        conflicts: list[TimelineConflict] = []
        by_entity: dict[str, list[SpatiotemporalEvent]] = defaultdict(list)
        for event in events:
            if event.time.era:
                by_entity[event.entity_id].append(event)

        for entity_id, entity_events in by_entity.items():
            # Count events per era
            era_counts: dict[str, list[SpatiotemporalEvent]] = defaultdict(list)
            for ev in entity_events:
                if ev.time.era:
                    era_counts[ev.time.era].append(ev)

            if len(era_counts) < 2:
                continue

            # Find the dominant era (most events)
            dominant_era = max(era_counts, key=lambda e: len(era_counts[e]))
            dominant_order = era_to_order(dominant_era)

            for era, era_events in era_counts.items():
                if era == dominant_era:
                    continue
                other_order = era_to_order(era)
                # Non-adjacent eras (gap > 1 in ordering) are suspicious
                gap = abs(dominant_order - other_order)
                if gap <= 1:
                    continue

                for ev in era_events:
                    entity_name = ev.entity_name or entity_id
                    confidence = ev.time.confidence * 0.7
                    conflicts.append(TimelineConflict(
                        id=f"era_mismatch_{uuid.uuid4().hex[:8]}",
                        conflict_type=ConflictType.ERA_MISMATCH,
                        severity="error" if confidence > 0.5 else "warning",
                        description=(
                            f"{entity_name} has {len(era_counts[dominant_era])} event(s) "
                            f"in {dominant_era} but event '{ev.description or ev.id}' "
                            f"is placed in {era} (era gap: {gap})"
                        ),
                        event_a_id=era_counts[dominant_era][0].id,
                        event_b_id=ev.id,
                        entity_id=entity_id,
                        suggestion=(
                            f"Verify era for '{ev.description or ev.id}' — "
                            f"expected {dominant_era}, found {era}"
                        ),
                        confidence=confidence,
                    ))

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


    def _detect_causal_paradoxes(
        self,
        events: list[SpatiotemporalEvent],
        causal_links: list[CausalLink],
    ) -> list[TimelineConflict]:
        """Detect causal paradoxes: event A causes B but B occurs before A.

        A causal paradox exists when:
        1. A CausalLink says cause_event -> effect_event
        2. The effect_event's time is strictly *before* the cause_event's time

        Also detects cycles in causal chains (A causes B, B causes C, C causes A).
        """
        conflicts: list[TimelineConflict] = []
        events_by_id = {e.id: e for e in events}

        # --- Direct paradoxes: effect before cause ---
        for link in causal_links:
            cause = events_by_id.get(link.cause_event_id)
            effect = events_by_id.get(link.effect_event_id)
            if not cause or not effect:
                continue

            if _event_strictly_before(effect, cause):
                confidence = min(
                    cause.time.confidence, effect.time.confidence, link.confidence
                ) * 0.85
                conflicts.append(TimelineConflict(
                    id=f"causal_paradox_{uuid.uuid4().hex[:8]}",
                    conflict_type=ConflictType.CAUSAL_PARADOX,
                    severity="error" if confidence > 0.5 else "warning",
                    description=(
                        f"Causal paradox: '{cause.description or cause.id}' causes "
                        f"'{effect.description or effect.id}', but the effect "
                        f"occurs before the cause "
                        f"({_time_desc(effect.time)} < {_time_desc(cause.time)})"
                    ),
                    event_a_id=link.cause_event_id,
                    event_b_id=link.effect_event_id,
                    entity_id=cause.entity_id,
                    suggestion=(
                        f"Reverse causal direction or fix timeline for "
                        f"'{link.description or link.cause_event_id} -> {link.effect_event_id}'"
                    ),
                    confidence=confidence,
                ))

        # --- Cycle detection in causal graph ---
        # Build adjacency from causal links
        adj: dict[str, list[str]] = defaultdict(list)
        for link in causal_links:
            adj[link.cause_event_id].append(link.effect_event_id)

        visited: set[str] = set()
        in_stack: set[str] = set()
        cycle_nodes: list[list[str]] = []

        def _dfs(node: str, path: list[str]) -> None:
            if node in in_stack:
                # Found cycle — extract it
                idx = path.index(node)
                cycle_nodes.append(path[idx:])
                return
            if node in visited:
                return
            visited.add(node)
            in_stack.add(node)
            path.append(node)
            for nxt in adj.get(node, []):
                _dfs(nxt, path)
            path.pop()
            in_stack.discard(node)

        for start in adj:
            if start not in visited:
                _dfs(start, [])

        for cycle in cycle_nodes:
            cycle_descs = []
            for eid in cycle:
                ev = events_by_id.get(eid)
                cycle_descs.append(ev.description or eid if ev else eid)
            conflicts.append(TimelineConflict(
                id=f"causal_cycle_{uuid.uuid4().hex[:8]}",
                conflict_type=ConflictType.CAUSAL_PARADOX,
                severity="error",
                description=(
                    f"Causal cycle detected: {' -> '.join(cycle_descs)} -> {cycle_descs[0]}"
                ),
                event_a_id=cycle[0],
                event_b_id=cycle[-1] if len(cycle) > 1 else cycle[0],
                suggestion="Break the causal cycle — one link must be removed or reversed",
                confidence=0.9,
            ))

        return conflicts


def _event_strictly_before(a: SpatiotemporalEvent, b: SpatiotemporalEvent) -> bool:
    """Return True if event a's time is strictly before event b's time."""
    # Compare eras first
    a_order = era_to_order(a.time.era)
    b_order = era_to_order(b.time.era)
    if a_order < b_order:
        return True
    if a_order > b_order:
        return False
    # Same era — compare years
    if a.time.year_end is not None and b.time.year_start is not None:
        return a.time.year_end < b.time.year_start
    return False


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
