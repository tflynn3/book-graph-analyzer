"""LLM-assisted causal link extraction from events/passages.

Provides a higher-quality extraction path using LLM reasoning to identify
causal relationships between events. Falls back to heuristic extraction
when LLM is unavailable.

TODO(#48): Add batch extraction for large event sets.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Optional

from .causal_extraction import extract_causal_links_heuristic
from .models import CausalLink, SpatiotemporalEvent

logger = logging.getLogger(__name__)


class ExtractionMode(str, Enum):
    """How causal links were extracted."""
    HEURISTIC = "heuristic"
    LLM = "llm"
    LLM_FALLBACK_HEURISTIC = "llm_fallback_heuristic"


_LLM_CAUSAL_PROMPT = """\
You are a literary analyst. Given a list of events from a narrative, identify \
causal relationships between them.

For each causal link, provide:
- cause_event_id: the event that causes the other
- effect_event_id: the event that is caused
- description: brief explanation of the causal relationship
- confidence: 0.0-1.0 how confident you are this is a real causal link

Return a JSON array of objects with those fields. Only include genuine causal \
links where one event demonstrably leads to or causes another. Be conservative \
— do not invent links where temporal sequence alone is the only evidence.

Events:
{events_json}

Return ONLY the JSON array, no other text.
"""


def _format_events_for_prompt(events: list[SpatiotemporalEvent]) -> str:
    """Format events into a compact JSON for the LLM prompt."""
    items = []
    for ev in events:
        items.append({
            "id": ev.id,
            "entity": ev.entity_name or ev.entity_id,
            "time": f"{ev.time.era or '?'} {ev.time.year_start or '?'}",
            "description": ev.description,
        })
    return json.dumps(items, indent=2)


def _parse_llm_response(
    response: str,
    valid_ids: set[str],
) -> list[CausalLink]:
    """Parse LLM JSON response into CausalLink objects."""
    # Strip markdown fences if present
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("LLM causal extraction returned invalid JSON")
        return []

    if not isinstance(data, list):
        return []

    links: list[CausalLink] = []
    seen: set[tuple[str, str]] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        cause = item.get("cause_event_id", "")
        effect = item.get("effect_event_id", "")
        if cause not in valid_ids or effect not in valid_ids:
            continue
        if cause == effect:
            continue
        pair = (cause, effect)
        if pair in seen:
            continue
        seen.add(pair)

        conf = item.get("confidence", 0.7)
        if not isinstance(conf, (int, float)):
            conf = 0.7
        conf = max(0.0, min(1.0, float(conf)))

        links.append(CausalLink(
            cause_event_id=cause,
            effect_event_id=effect,
            description=item.get("description", ""),
            confidence=round(conf, 3),
        ))

    links.sort(key=lambda l: -l.confidence)
    return links


class CausalExtractionResult:
    """Result of causal extraction including mode metadata."""

    def __init__(
        self,
        links: list[CausalLink],
        mode: ExtractionMode,
        event_count: int,
    ):
        self.links = links
        self.mode = mode
        self.event_count = event_count

    def to_dict(self) -> dict:
        return {
            "extraction_mode": self.mode.value,
            "event_count": self.event_count,
            "link_count": len(self.links),
            "links": [
                {
                    "cause": l.cause_event_id,
                    "effect": l.effect_event_id,
                    "description": l.description,
                    "confidence": l.confidence,
                }
                for l in self.links
            ],
        }


def extract_causal_links(
    events: list[SpatiotemporalEvent],
    *,
    use_llm: bool = False,
    llm_client: object | None = None,
    min_confidence: float = 0.4,
) -> CausalExtractionResult:
    """Extract causal links with optional LLM assistance.

    Args:
        events: Events to analyze.
        use_llm: If True, attempt LLM extraction first.
        llm_client: An LLMClient instance (from book_graph_analyzer.llm).
            Must have a .generate(prompt) method returning a string.
        min_confidence: Minimum confidence threshold for heuristic path.

    Returns:
        CausalExtractionResult with links and extraction mode metadata.
    """
    if not events:
        return CausalExtractionResult([], ExtractionMode.HEURISTIC, 0)

    if use_llm and llm_client is not None:
        try:
            prompt = _LLM_CAUSAL_PROMPT.format(
                events_json=_format_events_for_prompt(events),
            )
            response = llm_client.generate(prompt)  # type: ignore[union-attr]
            valid_ids = {e.id for e in events}
            links = _parse_llm_response(response, valid_ids)
            if links:
                logger.info(
                    "LLM causal extraction: %d links from %d events",
                    len(links), len(events),
                )
                return CausalExtractionResult(links, ExtractionMode.LLM, len(events))
            else:
                logger.warning("LLM returned no valid links, falling back to heuristic")
        except Exception:
            logger.warning("LLM causal extraction failed, falling back to heuristic", exc_info=True)

        # Fallback
        links = extract_causal_links_heuristic(events, min_confidence=min_confidence)
        return CausalExtractionResult(links, ExtractionMode.LLM_FALLBACK_HEURISTIC, len(events))

    # Pure heuristic
    links = extract_causal_links_heuristic(events, min_confidence=min_confidence)
    return CausalExtractionResult(links, ExtractionMode.HEURISTIC, len(events))
