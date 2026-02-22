"""Spatiotemporal Engine: Cartography + Interlaced Timelines (Issue #48).

Provides:
- Normalized event time representation with uncertainty bounds
- Location graph metadata for travel feasibility checks
- Timeline conflict detection (overlapping events, impossible travel)
- LLM-assisted and heuristic causal link extraction
- Confidence calibration with source authority weights
- Location graph seeding from canonical Middle-earth data
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
from .llm_causal_extraction import (
    extract_causal_links,
    CausalExtractionResult,
    ExtractionMode,
)
from .confidence import (
    SourceAuthorityRegistry,
    CalibrationResult,
    calibrate_event_confidence,
    calibrate_causal_link_confidence,
    calibrate_conflict_confidence,
)
from .location_seeds import (
    load_seed_locations,
    load_seed_edges,
    load_seed_location_graph,
)
from .corpus_reconciler import (
    CorpusReconciler, CorpusReconciliationResult, BookEvents, ContradictionCluster,
)
from .grounding import (
    METRICS_VERSION,
    TemporalGroundingGate,
    TemporalGroundingGateResult,
    TemporalGroundingMetrics,
    compute_temporal_grounding_metrics,
    backfill_temporal_grounding,
)

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
    "extract_causal_links",
    "CausalExtractionResult",
    "ExtractionMode",
    "SourceAuthorityRegistry",
    "CalibrationResult",
    "calibrate_event_confidence",
    "calibrate_causal_link_confidence",
    "calibrate_conflict_confidence",
    "load_seed_locations",
    "load_seed_edges",
    "load_seed_location_graph",
    "CorpusReconciler",
    "CorpusReconciliationResult",
    "BookEvents",
    "ContradictionCluster",
    "METRICS_VERSION",
    "TemporalGroundingGate",
    "TemporalGroundingGateResult",
    "TemporalGroundingMetrics",
    "compute_temporal_grounding_metrics",
    "backfill_temporal_grounding",
]
