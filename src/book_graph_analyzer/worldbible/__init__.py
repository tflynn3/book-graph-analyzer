"""
World Bible Module

Phase 7: Extract and synthesize the implicit rules and patterns
of a fictional world from text passages.
"""

from .categories import WorldBibleCategory, CATEGORY_PROMPTS
from .extractor import WorldBibleExtractor, ExtractionConfig
from .models import WorldRule, WorldBible, CulturalProfile

__all__ = [
    "WorldBibleCategory",
    "CATEGORY_PROMPTS",
    "WorldBibleExtractor",
    "ExtractionConfig",
    "WorldRule",
    "WorldBible",
    "CulturalProfile",
]
