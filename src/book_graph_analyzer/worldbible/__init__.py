"""
World Bible Extraction Module

Phase 7: Capture implicit rules and patterns of fictional worlds.
Synthesizes patterns across many passages to document:
- Magic/power systems and constraints
- Cultural profiles per race/people
- Geography and travel rules
- Cosmology and metaphysics
- Thematic patterns
"""

from .models import (
    WorldBible,
    WorldRule,
    WorldBibleCategory,
    CulturalProfile,
    MagicSystem,
    GeographyEntry,
)
from .extractor import WorldBibleExtractor, ExtractionConfig
from .patterns import PatternMatcher

__all__ = [
    "WorldBible",
    "WorldRule", 
    "WorldBibleCategory",
    "CulturalProfile",
    "MagicSystem",
    "GeographyEntry",
    "WorldBibleExtractor",
    "ExtractionConfig",
    "PatternMatcher",
]
