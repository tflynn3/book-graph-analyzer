"""Human-readable reconciliation report for timeline conflicts."""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections import Counter

from .models import ConflictType, SpatiotemporalEvent, TimelineConflict
from .grounding import compute_temporal_grounding_metrics

if TYPE_CHECKING:
    from .confidence import CalibrationResult
    from .extraction_bridge import BridgeReport
    from .llm_causal_extraction import CausalExtractionResult


class ReconciliationReport:
    """Generate human-readable inconsistency reports."""

    def __init__(
        self,
        conflicts: list[TimelineConflict],
        events: list[SpatiotemporalEvent] | None = None,
        bridge_report: BridgeReport | None = None,
        causal_result: CausalExtractionResult | None = None,
        calibration: CalibrationResult | None = None,
    ):
        self.conflicts = conflicts
        self.events = events or []
        self.bridge_report = bridge_report
        self.causal_result = causal_result
        self.calibration = calibration

    @property
    def error_count(self) -> int:
        return sum(1 for c in self.conflicts if c.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.conflicts if c.severity == "warning")

    def summary_line(self) -> str:
        total = len(self.conflicts)
        if total == 0:
            return "No timeline conflicts detected."
        return f"{total} conflict(s) found: {self.error_count} error(s), {self.warning_count} warning(s)"

    @property
    def era_mismatch_count(self) -> int:
        return sum(1 for c in self.conflicts if c.conflict_type == ConflictType.ERA_MISMATCH)

    @property
    def causal_paradox_count(self) -> int:
        return sum(1 for c in self.conflicts if c.conflict_type == ConflictType.CAUSAL_PARADOX)

    @property
    def source_counts(self) -> dict[str, int]:
        counter = Counter((e.source_book or "unknown") for e in self.events)
        return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))

    def to_text(self) -> str:
        lines = [
            "=" * 60, "  TIMELINE RECONCILIATION REPORT", "=" * 60, "",
            f"Events analyzed: {len(self.events)}",
            f"Conflicts found: {len(self.conflicts)}",
            f"  Errors:   {self.error_count}",
            f"  Warnings: {self.warning_count}",
        ]
        if self.era_mismatch_count:
            lines.append(f"  Era mismatches: {self.era_mismatch_count}")
        if self.causal_paradox_count:
            lines.append(f"  Causal paradoxes: {self.causal_paradox_count}")
        lines.append("")

        if self.events:
            metrics = compute_temporal_grounding_metrics(self.events)
            lines.extend([
                "--- TEMPORAL GROUNDING ---", "",
                f"  Grounded events: {metrics.grounded_events}/{metrics.total_events} ({metrics.grounded_ratio:.1%})",
                f"  Era coverage:    {metrics.era_grounded_events}/{metrics.total_events} ({metrics.era_ratio:.1%})",
                (
                    "  Year/interval coverage: "
                    f"{metrics.year_or_interval_grounded_events}/{metrics.total_events} "
                    f"({metrics.year_or_interval_ratio:.1%})"
                ),
                "",
            ])

            lines.extend(["--- SOURCE ATTRIBUTION ---", ""])
            for source, count in self.source_counts.items():
                lines.append(f"  {source}: {count} event(s)")
            lines.append("")

        # Confidence bridge summary
        if self.bridge_report and self.bridge_report.total > 0:
            br = self.bridge_report
            lines.extend([
                "--- EXTRACTION-VS-NORMALIZED CONFIDENCE ---", "",
                f"  Events bridged:            {br.total}",
                f"  Confidence aligned:        {br.aligned_count}",
                f"  Extraction overconfident:  {br.overconfident_count}",
                f"  Normalization boosted:     {br.boosted_count}",
                f"  Era changed during norm:   {br.era_changed_count}",
                f"  Avg confidence delta:      {br.avg_confidence_delta:+.3f}",
                "",
            ])

        # Causal extraction mode summary
        if self.causal_result:
            cr = self.causal_result
            lines.extend([
                "--- CAUSAL EXTRACTION ---", "",
                f"  Mode:          {cr.mode.value}",
                f"  Events input:  {cr.event_count}",
                f"  Links found:   {len(cr.links)}",
                "",
            ])

        # Confidence calibration summary
        if self.calibration:
            cal = self.calibration
            lines.extend([
                "--- CONFIDENCE CALIBRATION ---", "",
                f"  Events calibrated:   {cal.events_calibrated}",
                f"  Links calibrated:    {cal.links_calibrated}",
                f"  Conflicts calibrated:{cal.conflicts_calibrated}",
                f"  Avg authority weight: {cal.avg_authority_weight:.3f}",
                "",
            ])

        if not self.conflicts:
            lines.append("No inconsistencies detected.")
            return "\n".join(lines)

        by_type: dict[ConflictType, list[TimelineConflict]] = {}
        for c in self.conflicts:
            by_type.setdefault(c.conflict_type, []).append(c)

        for ctype, conflicts in by_type.items():
            lines.append(f"--- {ctype.value.replace('_', ' ').upper()} ({len(conflicts)}) ---")
            lines.append("")
            for c in conflicts:
                icon = "X" if c.severity == "error" else "~"
                lines.append(f"  {icon} [{c.severity.upper()}] {c.description}")
                if c.suggestion:
                    lines.append(f"    -> {c.suggestion}")
                lines.append(f"    Confidence: {c.confidence:.0%}")
                lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d: dict = {
            "events_analyzed": len(self.events),
            "temporal_grounding": compute_temporal_grounding_metrics(self.events).to_dict(),
            "total_conflicts": len(self.conflicts),
            "errors": self.error_count,
            "warnings": self.warning_count,
            "era_mismatches": self.era_mismatch_count,
            "causal_paradoxes": self.causal_paradox_count,
            "source_attribution": self.source_counts,
            "conflicts": [c.to_dict() for c in self.conflicts],
        }
        if self.bridge_report:
            d["bridge_report"] = self.bridge_report.to_dict()
        if self.causal_result:
            d["causal_extraction"] = self.causal_result.to_dict()
        if self.calibration:
            d["confidence_calibration"] = self.calibration.to_dict()
        return d
