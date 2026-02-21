"""Extract CausalLink candidates from events.

Provides heuristic extraction (no LLM required) and an LLM-assisted path.
Integrated into the extraction bridge pipeline.

TODO(#48): LLM-assisted extraction for higher-quality causal links.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .models import CausalLink, SpatiotemporalEvent

# Causal signal words/phrases — used for heuristic extraction
_CAUSAL_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\b(?:caused|led to|resulted in|triggered|provoked)\b", re.I), 0.8),
    (re.compile(r"\b(?:because of|due to|as a result of|owing to)\b", re.I), 0.75),
    (re.compile(r"\b(?:therefore|consequently|thus|hence|so that)\b", re.I), 0.65),
    (re.compile(r"\b(?:in response to|after .+ then|following)\b", re.I), 0.6),
    (re.compile(r"\b(?:forced|compelled|drove .+ to|made .+ possible)\b", re.I), 0.7),
    (re.compile(r"\b(?:destroyed|killed|slew|broke|shattered).+\b(?:which|and so|causing)\b", re.I), 0.75),
]


def extract_causal_links_heuristic(
    events: list[SpatiotemporalEvent],
    *,
    min_confidence: float = 0.4,
) -> list[CausalLink]:
    """Extract causal link candidates using heuristic signals.

    Strategy:
    1. For events with causal language in descriptions, link to temporally
       adjacent events of the same entity.
    2. For events where one event's description references another event's
       key terms (agent/location), propose a causal link.

    Returns candidate CausalLinks sorted by confidence descending.
    """
    links: list[CausalLink] = []
    seen: set[tuple[str, str]] = set()

    # Group by entity for temporal adjacency
    by_entity: dict[str, list[SpatiotemporalEvent]] = defaultdict(list)
    for ev in events:
        by_entity[ev.entity_id].append(ev)

    # Sort each entity's events by time
    for entity_id, entity_events in by_entity.items():
        sorted_events = sorted(
            entity_events,
            key=lambda e: (
                e.time.era or "",
                e.time.year_start if e.time.year_start is not None else 0,
            ),
        )

        for i in range(len(sorted_events) - 1):
            ev_a = sorted_events[i]
            ev_b = sorted_events[i + 1]

            # Check if ev_b's description contains causal language
            # suggesting it was caused by something (which would be ev_a)
            for pattern, base_conf in _CAUSAL_PATTERNS:
                if pattern.search(ev_b.description):
                    conf = base_conf * min(ev_a.time.confidence, ev_b.time.confidence)
                    if conf >= min_confidence:
                        pair = (ev_a.id, ev_b.id)
                        if pair not in seen:
                            seen.add(pair)
                            links.append(CausalLink(
                                cause_event_id=ev_a.id,
                                effect_event_id=ev_b.id,
                                description=f"{ev_a.description} -> {ev_b.description}",
                                confidence=round(conf, 3),
                            ))
                    break  # Only one link per adjacent pair

            # Also check ev_a's description for forward-causal language
            for pattern, base_conf in _CAUSAL_PATTERNS:
                if pattern.search(ev_a.description):
                    conf = base_conf * min(ev_a.time.confidence, ev_b.time.confidence) * 0.9
                    if conf >= min_confidence:
                        pair = (ev_a.id, ev_b.id)
                        if pair not in seen:
                            seen.add(pair)
                            links.append(CausalLink(
                                cause_event_id=ev_a.id,
                                effect_event_id=ev_b.id,
                                description=f"{ev_a.description} -> {ev_b.description}",
                                confidence=round(conf, 3),
                            ))
                    break

    # Cross-entity causal links: if event A mentions event B's entity or location
    events_by_id = {e.id: e for e in events}
    all_events = list(events)

    for i, ev_a in enumerate(all_events):
        for ev_b in all_events[i + 1:]:
            if ev_a.entity_id == ev_b.entity_id:
                continue  # Already handled above
            pair = (ev_a.id, ev_b.id)
            if pair in seen:
                continue

            # Check if ev_a's description references ev_b's entity
            if ev_b.entity_name and ev_b.entity_name.lower() in ev_a.description.lower():
                for pattern, base_conf in _CAUSAL_PATTERNS:
                    if pattern.search(ev_a.description):
                        conf = base_conf * 0.6  # Lower confidence for cross-entity
                        if conf >= min_confidence:
                            seen.add(pair)
                            links.append(CausalLink(
                                cause_event_id=ev_a.id,
                                effect_event_id=ev_b.id,
                                description=f"Cross-entity: {ev_a.description} -> {ev_b.description}",
                                confidence=round(conf, 3),
                            ))
                        break

    links.sort(key=lambda l: -l.confidence)
    return links
