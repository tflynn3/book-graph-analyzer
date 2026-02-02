"""
Style Analysis Module

Phase 4: Extract quantifiable patterns in writing style to create
an author's stylistic fingerprint.
"""

from .metrics import (
    SentenceMetrics,
    VocabularyProfile,
    Distribution,
    calculate_sentence_metrics,
    calculate_vocabulary_profile,
)
from .classifier import PassageType, classify_passage
from .fingerprint import AuthorStyleFingerprint
from .analyzer import StyleAnalyzer

__all__ = [
    # Metrics
    "SentenceMetrics",
    "VocabularyProfile", 
    "Distribution",
    "calculate_sentence_metrics",
    "calculate_vocabulary_profile",
    # Classification
    "PassageType",
    "classify_passage",
    # Fingerprint
    "AuthorStyleFingerprint",
    # Analyzer
    "StyleAnalyzer",
]
