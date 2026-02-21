# -*- coding: utf-8 -*-
"""
Text Normalizer -- P0 Fix for Entity Resolution v2 (Issue #12)

Fixes encoding artifacts and normalizes text before NER/extraction.

The core problem: text files encoded as UTF-8 get mis-read as Windows cp1252,
producing systematic mangling artifacts.

Example:
  U+201C (left double quote) in UTF-8 = bytes e2 80 9c
  Those 3 bytes decoded as cp1252: e2->a-circumflex, 80->Euro, 9c->oe-ligature
  Result: the sequence U+00E2 U+20AC U+0153 appears as "ae-oe" but displays as
  the well-known garbage string that the issue spec calls out.
"""

from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# cp1252 artifact replacements
#
# When UTF-8 bytes are decoded as cp1252, each 3-byte UTF-8 character
# becomes 3 cp1252 characters. These are the most common artifacts.
# Unicode codepoints are given explicitly to avoid source file encoding issues.
#
# How they were computed:
#   U+201C (left ")  -> UTF-8 e2 80 9c -> cp1252: U+00E2 U+20AC U+0153
#   U+201D (right ") -> UTF-8 e2 80 9d -> cp1252: U+00E2 U+20AC U+009D (undef)
#   U+2019 (apos)    -> UTF-8 e2 80 99 -> cp1252: U+00E2 U+20AC U+2122
#   U+2018 (l-sing)  -> UTF-8 e2 80 98 -> cp1252: U+00E2 U+20AC U+02DC
#   U+2014 (em dash) -> UTF-8 e2 80 94 -> cp1252: U+00E2 U+20AC U+201D
#   U+2013 (en dash) -> UTF-8 e2 80 93 -> cp1252: U+00E2 U+20AC U+201C
#   U+2026 (ellipsis)-> UTF-8 e2 80 a6 -> cp1252: U+00E2 U+20AC U+00A6
# ---------------------------------------------------------------------------
_ARTIFACT_TO_CLEAN: list[tuple[str, str]] = [
    # Must check longer/more specific patterns first
    ("\u00e2\u20ac\u0153", '"'),     # left double quote (U+201C)
    ("\u00e2\u20ac\u009d", '"'),     # right double quote (U+201D) -- undefined cp1252
    ("\u00e2\u20ac\u2122", "'"),     # right single / apostrophe (U+2019)
    ("\u00e2\u20ac\u02dc", "'"),     # left single quote (U+2018)
    ("\u00e2\u20ac\u201d", " - "),   # em dash (U+2014) -- NB: \x94 = U+201D in cp1252
    ("\u00e2\u20ac\u201c", " - "),   # en dash (U+2013) -- NB: \x93 = U+201C in cp1252
    ("\u00e2\u20ac\u00a6", "..."),   # ellipsis (U+2026)
    # Generic fallback: strip lone a-circumflex + euro sequences
    ("\u00e2\u20ac", ""),
]

# Unicode curly quotes -> straight ASCII
_UNICODE_QUOTES: list[tuple[str, str]] = [
    ("\u201c", '"'),   # LEFT DOUBLE QUOTATION MARK
    ("\u201d", '"'),   # RIGHT DOUBLE QUOTATION MARK
    ("\u2018", "'"),   # LEFT SINGLE QUOTATION MARK
    ("\u2019", "'"),   # RIGHT SINGLE QUOTATION MARK / apostrophe
    ("\u201a", "'"),   # SINGLE LOW-9 QUOTATION MARK
    ("\u201e", '"'),   # DOUBLE LOW-9 QUOTATION MARK
    ("\u2039", "<"),
    ("\u203a", ">"),
    ("\u00ab", '"'),
    ("\u00bb", '"'),
]

# Zero-width and invisible characters
_INVISIBLE_CHARS = re.compile(
    "[\u200b\u200c\u200d\ufeff\u00ad]+"
)

# All artifact start sequences (for fast detection)
_ARTIFACT_STARTS = tuple(a[0][:1] for a, _ in _ARTIFACT_TO_CLEAN)


class TextNormalizer:
    """
    Normalizes raw text for safe NER + entity extraction.

    Steps:
    1. NFC Unicode normalization
    2. Strip cp1252->UTF-8 mangling artifacts
    3. Normalize Unicode curly quotes to ASCII
    4. Remove zero-width characters
    """

    def __init__(
        self,
        fix_encoding: bool = True,
        normalize_quotes: bool = True,
        remove_zero_width: bool = True,
        nfc_normalize: bool = True,
    ) -> None:
        self.fix_encoding = fix_encoding
        self.normalize_quotes = normalize_quotes
        self.remove_zero_width = remove_zero_width
        self.nfc_normalize = nfc_normalize

    def normalize(self, text: str) -> str:
        """Full normalization pipeline."""
        if self.nfc_normalize:
            text = unicodedata.normalize("NFC", text)

        if self.fix_encoding:
            text = self.strip_encoding_artifacts(text)

        if self.normalize_quotes:
            text = self._normalize_quotes(text)

        if self.remove_zero_width:
            text = _INVISIBLE_CHARS.sub("", text)

        return text

    def strip_encoding_artifacts(self, text: str) -> str:
        """
        Remove cp1252->UTF-8 mangling artifacts (the P0 one-liner).

        Replaces the systematic 3-char sequences that appear when UTF-8 text
        is mis-decoded as cp1252.
        """
        for bad, good in _ARTIFACT_TO_CLEAN:
            if bad in text:
                text = text.replace(bad, good)
        return text

    def _normalize_quotes(self, text: str) -> str:
        """Replace Unicode curly quotes with ASCII equivalents."""
        for uchar, ascii_equiv in _UNICODE_QUOTES:
            text = text.replace(uchar, ascii_equiv)
        return text

    def is_clean(self, text: str) -> bool:
        """Return True if text has no known encoding artifacts."""
        for bad, _ in _ARTIFACT_TO_CLEAN:
            if bad in text:
                return False
        return True

    def find_artifacts(self, text: str) -> list[str]:
        """Return list of artifact strings found in text."""
        return [bad for bad, _ in _ARTIFACT_TO_CLEAN if bad in text]


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------
_default_normalizer = TextNormalizer()


def normalize_text(text: str) -> str:
    """Normalize text using default settings."""
    return _default_normalizer.normalize(text)


def strip_artifacts(text: str) -> str:
    """Strip encoding artifacts only."""
    return _default_normalizer.strip_encoding_artifacts(text)


def has_artifacts(text: str) -> bool:
    """Return True if text contains known encoding artifacts."""
    return not _default_normalizer.is_clean(text)
