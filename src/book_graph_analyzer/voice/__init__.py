"""
Character Voice Profiles Module

Phase 5: Capture how each character speaks distinctly.
Extract dialogue, attribute to speakers, build per-character profiles.
"""

from .dialogue import (
    DialogueExtraction,
    DialogueLine,
    extract_dialogue,
    extract_dialogue_from_passages,
)
from .profile import CharacterVoiceProfile
from .analyzer import VoiceAnalyzer

__all__ = [
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
