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
from .rules import (
    LoreRuleRegistry,
    LoreRuleValidator,
    WorldBibleRuleMapper,
    LoreRuleNeo4jWriter,
    SceneContext,
    TOLKIEN_LORE_RULES,
)
from .conflicts import (
    ConflictRegistry,
    ConflictDetector,
    ConflictAwareValidator,
    LoreConflictNeo4jWriter,
    KNOWN_TOLKIEN_CONFLICTS,
)
from .sociolinguistic_registers import (
    SociolinguisticRegister,
    RegisterProfile,
    RegisterDrift,
    SociolinguisticRegisterClassifier,
    detect_register_drift,
)

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
    "LoreRuleRegistry",
    "LoreRuleValidator",
    "WorldBibleRuleMapper",
    "LoreRuleNeo4jWriter",
    "SceneContext",
    "TOLKIEN_LORE_RULES",
    "ConflictRegistry",
    "ConflictDetector",
    "ConflictAwareValidator",
    "LoreConflictNeo4jWriter",
    "KNOWN_TOLKIEN_CONFLICTS",
    "SociolinguisticRegister",
    "RegisterProfile",
    "RegisterDrift",
    "SociolinguisticRegisterClassifier",
    "detect_register_drift",
]
