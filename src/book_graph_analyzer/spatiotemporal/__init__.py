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
)
from .normalizer import TimeNormalizer
from .conflict_detector import ConflictDetector
from .report import ReconciliationReport

__all__ = [
    "NormalizedTime",
    "SpatiotemporalEvent",
    "LocationNode",
    "LocationEdge",
    "TimelineConflict",
    "ConflictType",
    "TimeNormalizer",
    "ConflictDetector",
    "ReconciliationReport",
]
