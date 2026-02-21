"""
Character Voice Profiles Module

Phase 5: Capture how each character speaks distinctly.
Extract dialogue, attribute to speakers, build per-character profiles.
"""

from .audience import (
    AudienceClassification,
    classify_audience_type,
    classify_context_type,
    classify_dialogue_line,
    AUDIENCE_TYPES,
    CONTEXT_TYPES,
)
from .dialogue import (
    DialogueExtraction,
    DialogueLine,
    extract_dialogue,
    extract_dialogue_from_passages,
)
from .profile import CharacterVoiceProfile
from .analyzer import VoiceAnalyzer

__all__ = [
    # Audience classification
    "AudienceClassification",
    "classify_audience_type",
    "classify_context_type",
    "classify_dialogue_line",
    "AUDIENCE_TYPES",
    "CONTEXT_TYPES",
    # Dialogue extraction
    "DialogueExtraction",
    "DialogueLine",
    "extract_dialogue",
    "extract_dialogue_from_passages",
    # Profile
    "CharacterVoiceProfile",
    # Analyzer
    "VoiceAnalyzer",
]
