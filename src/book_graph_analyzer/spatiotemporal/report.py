"""Human-readable reconciliation report for timeline conflicts."""

from __future__ import annotations

from .models import ConflictType, SpatiotemporalEvent, TimelineConflict


class ReconciliationReport:
    """Generate human-readable inconsistency reports."""

    def __init__(self, conflicts: list[TimelineConflict], events: list[SpatiotemporalEvent] | None = None):
        self.conflicts = conflicts
        self.events = events or []

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

    def to_text(self) -> str:
        lines = [
            "=" * 60, "  TIMELINE RECONCILIATION REPORT", "=" * 60, "",
            f"Events analyzed: {len(self.events)}",
            f"Conflicts found: {len(self.conflicts)}",
            f"  Errors:   {self.error_count}",
            f"  Warnings: {self.warning_count}", "",
        ]
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
        return {
            "events_analyzed": len(self.events),
            "total_conflicts": len(self.conflicts),
            "errors": self.error_count,
            "warnings": self.warning_count,
            "conflicts": [c.to_dict() for c in self.conflicts],
        }
