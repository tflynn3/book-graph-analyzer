from __future__ import annotations

from dataclasses import dataclass


BOOK_GENEALOGY_THRESHOLDS: dict[str, int] = {
    "the hobbit": 2,
    "the fellowship of the ring": 6,
    "the two towers": 4,
    "the return of the king": 4,
    "the silmarillion": 20,
}


@dataclass
class GenealogyThresholdResult:
    book: str
    observed: int
    threshold: int

    @property
    def passed(self) -> bool:
        return self.observed >= self.threshold

    def to_dict(self) -> dict[str, object]:
        return {
            "book": self.book,
            "observed": self.observed,
            "threshold": self.threshold,
            "passed": self.passed,
        }


def threshold_for_book(book: str, default_threshold: int = 1) -> int:
    return BOOK_GENEALOGY_THRESHOLDS.get((book or "").strip().lower(), default_threshold)


def evaluate_genealogy_threshold(book: str, observed: int, default_threshold: int = 1) -> GenealogyThresholdResult:
    threshold = threshold_for_book(book, default_threshold=default_threshold)
    return GenealogyThresholdResult(book=book, observed=observed, threshold=threshold)
