"""Spatiotemporal Engine: Cartography + Interlaced Timelines (Issue #48).

Provides:
- Normalized event time representation with uncertainty bounds
- Location graph metadata for travel feasibility checks
- Timeline conflict detection (overlapping events, impossible travel)
- CLI integration for reconciliation reports
"""

from .models import (
    NormalizedTime,
    SpatiotemporalEvent,
    LocationNode,
    LocationEdge,
    TimelineConflict,
    ConflictType,
    CausalLink,
)
from .normalizer import TimeNormalizer
from .conflict_detector import ConflictDetector
from .report import ReconciliationReport
from .extraction_bridge import ExtractionBridge, NormalizationResult, BridgeReport
from .causal_extraction import extract_causal_links_heuristic
from .corpus_reconciler import CorpusReconciler, CorpusReconciliationResult, BookEvents

__all__ = [
    "NormalizedTime",
    "SpatiotemporalEvent",
    "LocationNode",
    "LocationEdge",
    "TimelineConflict",
    "ConflictType",
    "CausalLink",
    "TimeNormalizer",
    "ConflictDetector",
    "ReconciliationReport",
    "ExtractionBridge",
    "NormalizationResult",
    "BridgeReport",
    "extract_causal_links_heuristic",
    "CorpusReconciler",
    "CorpusReconciliationResult",
    "BookEvents",
]
