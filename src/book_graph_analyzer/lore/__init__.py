"""
Lore Checking Module

Validates claims and statements against extracted world knowledge.
Use for:
- Fact-checking statements ("Turin lived in the Second Age" -> False)
- Consistency validation in generated text
- Finding contradictions in drafts
- Temporal reasoning (was X alive during Y Age?)
- Event ordering (did X happen before Y?)
"""

from .checker import LoreChecker, ValidationResult, ValidationStatus
from .parser import ClaimParser, ParsedClaim, ClaimType
from .temporal import Timeline, TemporalEntity, TemporalExtractor, Era
from .events import Event, EventGraph, EventExtractor, EventRelation

__all__ = [
    "LoreChecker",
    "ValidationResult",
    "ValidationStatus",
    "ClaimType",
    "ClaimParser",
    "ParsedClaim",
    "Timeline",
    "TemporalEntity",
    "TemporalExtractor",
    "Era",
    "Event",
    "EventGraph",
    "EventExtractor",
    "EventRelation",
]
