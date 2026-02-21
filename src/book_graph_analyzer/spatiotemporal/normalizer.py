"""Time normalization: parse temporal expressions into NormalizedTime."""

from __future__ import annotations

import re

from ..graph.temporal import ERA_ORDER, canonicalize_era
from .models import NormalizedTime


_YEAR_PATTERNS = [
    (r"(?:T\.?A\.?\s*|Third\s+Age\s+)(\d{1,5})", "Third Age"),
    (r"(?:S\.?A\.?\s*|Second\s+Age\s+)(\d{1,5})", "Second Age"),
    (r"(?:F\.?A\.?\s*|First\s+Age\s+)(\d{1,5})", "First Age"),
    (r"[Yy]ear\s+(\d{1,5})\s+of\s+the\s+(First|Second|Third|Fourth)\s+Age", None),
]

_FUZZY_ERA_PATTERNS = [
    (r"(?:in|during|of)\s+the\s+(First|Second|Third|Fourth)\s+Age", None),
    (r"(?:before|ere)\s+the\s+(First|Second|Third|Fourth)\s+Age", "before"),
    (r"(?:after)\s+the\s+(First|Second|Third|Fourth)\s+Age", "after"),
    (r"(?:Years?\s+of\s+the\s+)(Trees|Lamps)", None),
]


class TimeNormalizer:
    """Normalize temporal expressions into NormalizedTime objects."""

    def normalize(self, text: str, context_era: str | None = None) -> NormalizedTime:
        text = text.strip()

        for pattern, era in _YEAR_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                if era is None:
                    year = int(m.group(1))
                    era_name = f"{m.group(2)} Age"
                else:
                    year = int(m.group(1))
                    era_name = era
                return NormalizedTime(
                    era=canonicalize_era(era_name) or era_name,
                    year_start=year, year_end=year,
                    confidence=0.9, raw_text=text,
                )

        for pattern, modifier in _FUZZY_ERA_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                groups = m.groups()
                if "Trees" in groups[0] or "Lamps" in groups[0]:
                    era_name = f"Years of the {groups[0]}"
                else:
                    era_name = f"{groups[0]} Age"
                era_name = canonicalize_era(era_name) or era_name

                if modifier == "before":
                    order = ERA_ORDER.get(era_name, 99)
                    prev_eras = [e for e, o in ERA_ORDER.items() if o == order - 1]
                    if prev_eras:
                        era_name = prev_eras[0]

                return NormalizedTime(era=era_name, confidence=0.4, raw_text=text)

        m = re.search(r"(\d+)\s+years?\s+(before|after)", text, re.IGNORECASE)
        if m:
            return NormalizedTime(era=context_era, confidence=0.3, raw_text=text)

        return NormalizedTime(era=context_era, confidence=0.1, raw_text=text)

    def normalize_event_time(
        self, raw_text: str | None, era: str | None = None,
        year: int | None = None, year_end: int | None = None,
        confidence: float | None = None,
    ) -> NormalizedTime:
        if raw_text and not era and not year:
            return self.normalize(raw_text)

        era_canonical = canonicalize_era(era) if era else None
        conf = confidence if confidence is not None else (
            0.9 if year is not None else (0.4 if era else 0.1)
        )
        return NormalizedTime(
            era=era_canonical, year_start=year, year_end=year_end or year,
            confidence=conf, raw_text=raw_text,
        )
