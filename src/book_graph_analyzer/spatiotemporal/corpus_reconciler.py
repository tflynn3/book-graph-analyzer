"""Cross-book timeline reconciliation.

Evaluates spatiotemporal conflicts across multiple books in a corpus,
producing a unified reconciliation report with per-book and cross-book
conflict summaries.

TODO(#48): LLM-assisted conflict resolution suggestions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .causal_extraction import extract_causal_links_heuristic
from .conflict_detector import ConflictDetector
from .models import (
    CausalLink, ConflictType, LocationEdge, LocationNode,
    SpatiotemporalEvent, TimelineConflict,
)
from .report import ReconciliationReport


@dataclass
class BookEvents:
    """Events from a single book."""
    book_id: str
    book_title: str
    events: list[SpatiotemporalEvent] = field(default_factory=list)
    causal_links: list[CausalLink] = field(default_factory=list)


@dataclass
class CorpusReconciliationResult:
    """Result of cross-book reconciliation."""
    books: list[BookEvents]
    per_book_conflicts: dict[str, list[TimelineConflict]] = field(default_factory=dict)
    cross_book_conflicts: list[TimelineConflict] = field(default_factory=list)
    all_causal_links: list[CausalLink] = field(default_factory=list)

    @property
    def total_events(self) -> int:
        return sum(len(b.events) for b in self.books)

    @property
    def total_conflicts(self) -> int:
        return sum(len(c) for c in self.per_book_conflicts.values()) + len(self.cross_book_conflicts)

    @property
    def total_errors(self) -> int:
        count = 0
        for conflicts in self.per_book_conflicts.values():
            count += sum(1 for c in conflicts if c.severity == "error")
        count += sum(1 for c in self.cross_book_conflicts if c.severity == "error")
        return count

    def summary_text(self) -> str:
        lines = [
            "=" * 60,
            "  CORPUS TIMELINE RECONCILIATION",
            "=" * 60,
            "",
            f"Books analyzed: {len(self.books)}",
            f"Total events: {self.total_events}",
            f"Total conflicts: {self.total_conflicts}",
            f"  Errors: {self.total_errors}",
            f"  Cross-book conflicts: {len(self.cross_book_conflicts)}",
            f"Causal links extracted: {len(self.all_causal_links)}",
            "",
        ]

        for book in self.books:
            book_conflicts = self.per_book_conflicts.get(book.book_id, [])
            errors = sum(1 for c in book_conflicts if c.severity == "error")
            warnings = len(book_conflicts) - errors
            lines.append(
                f"  {book.book_title}: {len(book.events)} events, "
                f"{len(book_conflicts)} conflicts ({errors} errors, {warnings} warnings)"
            )

        if self.cross_book_conflicts:
            lines.append("")
            lines.append(f"--- CROSS-BOOK CONFLICTS ({len(self.cross_book_conflicts)}) ---")
            lines.append("")
            for c in self.cross_book_conflicts:
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
            "books_analyzed": len(self.books),
            "total_events": self.total_events,
            "total_conflicts": self.total_conflicts,
            "total_errors": self.total_errors,
            "cross_book_conflict_count": len(self.cross_book_conflicts),
            "causal_links_extracted": len(self.all_causal_links),
            "per_book": {
                book.book_id: {
                    "title": book.book_title,
                    "event_count": len(book.events),
                    "conflicts": [c.to_dict() for c in self.per_book_conflicts.get(book.book_id, [])],
                    "causal_links": [
                        {"cause": l.cause_event_id, "effect": l.effect_event_id,
                         "description": l.description, "confidence": l.confidence}
                        for l in book.causal_links
                    ],
                }
                for book in self.books
            },
            "cross_book_conflicts": [c.to_dict() for c in self.cross_book_conflicts],
        }


class CorpusReconciler:
    """Reconcile timelines across multiple books.

    Usage:
        reconciler = CorpusReconciler()
        reconciler.add_book("hobbit", "The Hobbit", hobbit_events)
        reconciler.add_book("lotr", "Lord of the Rings", lotr_events)
        result = reconciler.reconcile()
        print(result.summary_text())
    """

    def __init__(
        self,
        locations: dict[str, LocationNode] | None = None,
        edges: list[LocationEdge] | None = None,
        extract_causal: bool = True,
    ):
        self.locations = locations or {}
        self.edges = edges or []
        self.books: list[BookEvents] = []
        self.extract_causal = extract_causal

    def add_book(
        self,
        book_id: str,
        book_title: str,
        events: list[SpatiotemporalEvent],
        causal_links: list[CausalLink] | None = None,
    ) -> None:
        """Add a book's events to the reconciler."""
        book = BookEvents(
            book_id=book_id,
            book_title=book_title,
            events=events,
            causal_links=causal_links or [],
        )
        self.books.append(book)

    def add_book_from_json(self, json_path: str | Path, book_id: str, book_title: str) -> int:
        """Load events from a JSON file and add to reconciler. Returns event count."""
        path = Path(json_path)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "events" in raw:
            raw_events = raw["events"]
        elif isinstance(raw, list):
            raw_events = raw
        else:
            raw_events = []

        events = [SpatiotemporalEvent(**e) for e in raw_events]
        self.add_book(book_id, book_title, events)
        return len(events)

    def reconcile(
        self,
        *,
        check_causal: bool = True,
        check_era_mismatches: bool = True,
        causal_min_confidence: float = 0.4,
    ) -> CorpusReconciliationResult:
        """Run full reconciliation across all books.

        Steps:
        1. Per-book: detect conflicts within each book
        2. Per-book: extract causal links (if enabled) and check for paradoxes
        3. Cross-book: merge all events and detect cross-book conflicts
        """
        detector = ConflictDetector(
            locations=self.locations, edges=self.edges,
        )

        per_book_conflicts: dict[str, list[TimelineConflict]] = {}
        all_causal_links: list[CausalLink] = []

        # Per-book analysis
        for book in self.books:
            # Extract causal links if enabled and none provided
            if self.extract_causal and not book.causal_links:
                book.causal_links = extract_causal_links_heuristic(
                    book.events, min_confidence=causal_min_confidence,
                )

            all_causal_links.extend(book.causal_links)

            conflicts = detector.detect_conflicts(
                book.events,
                check_era_mismatches=check_era_mismatches,
                check_causal_paradoxes=check_causal and bool(book.causal_links),
                causal_links=book.causal_links,
            )
            per_book_conflicts[book.book_id] = conflicts

        # Cross-book analysis: merge all events
        all_events = []
        for book in self.books:
            all_events.extend(book.events)

        cross_conflicts = detector.detect_conflicts(
            all_events,
            check_era_mismatches=check_era_mismatches,
            check_causal_paradoxes=check_causal and bool(all_causal_links),
            causal_links=all_causal_links,
        )

        # Filter to only truly cross-book conflicts (not duplicates of per-book)
        per_book_event_ids: dict[str, set[str]] = {}
        for book in self.books:
            per_book_event_ids[book.book_id] = {e.id for e in book.events}

        def is_cross_book(conflict: TimelineConflict) -> bool:
            if not conflict.event_a_id or not conflict.event_b_id:
                return False
            a_book = None
            b_book = None
            for book_id, eids in per_book_event_ids.items():
                if conflict.event_a_id in eids:
                    a_book = book_id
                if conflict.event_b_id in eids:
                    b_book = book_id
            return a_book is not None and b_book is not None and a_book != b_book

        cross_book_only = [c for c in cross_conflicts if is_cross_book(c)]

        return CorpusReconciliationResult(
            books=self.books,
            per_book_conflicts=per_book_conflicts,
            cross_book_conflicts=cross_book_only,
            all_causal_links=all_causal_links,
        )
