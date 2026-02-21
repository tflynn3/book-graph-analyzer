"""Raw-text linguistic lineage extraction.

Extracts language/etymology statements from prose using regex patterns
and optional LLM fallback.  Outputs the same ``LinguisticLineage``
structures consumed by ``worldbible.lineage`` and ``graph.writer``.

See Issue #46.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from book_graph_analyzer.models.worldbuilding import (
    DerivationType,
    LanguageDerivation,
    LanguageForm,
    LinguisticLineage,
    TolkienLanguage,
)
from book_graph_analyzer.worldbible.lineage import _resolve_language


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    # "X, called Y in Sindarin"
    (
        "called_in",
        re.compile(
            r"(?P<form1>[A-ZÀ-Ž][\w''\-]+)"
            r",?\s+(?:called|known as|named)\s+"
            r"(?P<form2>[A-ZÀ-Ž][\w''\-]+)"
            r"\s+in\s+(?:the\s+)?(?P<lang2>[A-ZÀ-Ž][\w\s]+?)(?=[,;.\)]|$)",
            re.MULTILINE,
        ),
        "translation",
    ),
    # "the Sindarin name for X is Y"
    (
        "name_for",
        re.compile(
            r"the\s+(?P<lang>[A-ZÀ-Ž][\w\s]+?)\s+(?:name|word)\s+for\s+"
            r"(?P<form_common>[A-ZÀ-Ž][\w''\-]+)\s+(?:is|was)\s+"
            r"(?P<form_lang>[A-ZÀ-Ž][\w''\-]+)",
            re.IGNORECASE | re.MULTILINE,
        ),
        "translation",
    ),
    # "Y (Sindarin: 'gloss')"
    (
        "parenthetical",
        re.compile(
            r"(?P<form>[A-ZÀ-Ž][\w''\-]+)"
            r"\s*\(\s*(?P<lang>[A-ZÀ-Ž][\w\s]+?)\s*[:;]?\s*['\"]?"
            r"(?P<gloss>[^)]+?)['\"]?\s*\)",
            re.MULTILINE,
        ),
        "translation",
    ),
    # "X means 'Y' in Sindarin"
    (
        "means_in",
        re.compile(
            r"(?P<form>[A-ZÀ-Ž][\w''\-]+)\s+means?\s+['\"](?P<gloss>[^'\"]+)['\"]\s+"
            r"in\s+(?:the\s+)?(?P<lang>[A-ZÀ-Ž][\w\s]+?)(?=[,;.\)]|$)",
            re.IGNORECASE | re.MULTILINE,
        ),
        "translation",
    ),
    # "from the Sindarin X meaning Y"
    (
        "from_the",
        re.compile(
            r"from\s+(?:the\s+)?(?P<lang>[A-ZÀ-Ž][\w\s]+?)\s+"
            r"(?P<form>[A-ZÀ-Ž][\w''\-]+)"
            r"(?:\s+meaning\s+['\"]?(?P<gloss>[^'\",.\)]+)['\"]?)?",
            re.IGNORECASE | re.MULTILINE,
        ),
        "adaptation",
    ),
    # "X is a Sindarin word"
    (
        "is_a_word",
        re.compile(
            r"(?P<form>[A-ZÀ-Ž][\w''\-]+)\s+is\s+(?:a|an)\s+"
            r"(?P<lang>[A-ZÀ-Ž][\w\s]+?)\s+(?:word|name|term)",
            re.IGNORECASE | re.MULTILINE,
        ),
        "translation",
    ),
]


@dataclass
class _RawHit:
    form: str
    language: str
    gloss: str | None = None
    related_form: str | None = None
    related_language: str | None = None
    derivation_type: str = "translation"
    pattern_name: str = ""


@dataclass
class ExtractionResult:
    lineages: list[LinguisticLineage]
    extraction_mode: str = "regex"
    hit_count: int = 0


def extract_lineages_from_text(
    text: str,
    *,
    use_llm_fallback: bool = False,
    min_hits: int = 0,
) -> ExtractionResult:
    """Extract linguistic lineages from raw prose."""
    hits = _extract_regex(text)
    mode = "regex"

    if use_llm_fallback and len(hits) < min_hits:
        llm_lineages = _extract_llm(text)
        if llm_lineages:
            return ExtractionResult(lineages=llm_lineages, extraction_mode="llm", hit_count=len(llm_lineages))

    lineages = _group_hits(hits)
    return ExtractionResult(lineages=lineages, extraction_mode=mode, hit_count=len(hits))


def _extract_regex(text: str) -> list[_RawHit]:
    hits: list[_RawHit] = []
    for pat_name, pattern, deriv_type in _PATTERNS:
        for m in pattern.finditer(text):
            hit = _raw_hit_from_groups(m.groupdict(), pat_name, deriv_type)
            if hit:
                hits.append(hit)
    return hits


def _raw_hit_from_groups(g: dict[str, str | None], pat_name: str, deriv_type: str) -> _RawHit | None:
    form = (g.get("form") or g.get("form1") or g.get("form_lang") or "").strip()
    if not form:
        return None
    lang = (g.get("lang") or g.get("lang2") or "").strip()
    gloss = (g.get("gloss") or "").strip() or None
    related_form = (g.get("form2") or g.get("form_common") or "").strip() or None
    related_lang = (g.get("lang1") or "").strip() or None
    return _RawHit(form=form, language=lang, gloss=gloss, related_form=related_form,
                   related_language=related_lang, derivation_type=deriv_type, pattern_name=pat_name)


def _uid() -> str:
    return f"lf_{uuid.uuid4().hex[:8]}"


def _group_hits(hits: list[_RawHit]) -> list[LinguisticLineage]:
    clusters: dict[str, list[_RawHit]] = {}
    for hit in hits:
        key = hit.form.lower()
        if hit.related_form:
            alt = hit.related_form.lower()
            key = min(key, alt)
        clusters.setdefault(key, []).append(hit)

    lineages: list[LinguisticLineage] = []
    for cluster_key, cluster_hits in clusters.items():
        forms_map: dict[str, LanguageForm] = {}
        derivations: list[LanguageDerivation] = []
        entity_id = f"entity_{cluster_key.replace(' ', '_')}"

        for hit in cluster_hits:
            fid = _ensure_form(forms_map, hit.form, hit.language, hit.gloss, entity_id)
            if hit.related_form:
                rid = _ensure_form(forms_map, hit.related_form, hit.related_language or "Common Speech", None, entity_id)
                derivations.append(LanguageDerivation(
                    source_form_id=rid, target_form_id=fid,
                    derivation_type=_resolve_deriv(hit.derivation_type),
                ))

        lineages.append(LinguisticLineage(entity_id=entity_id, forms=list(forms_map.values()), derivations=derivations))
    return lineages


def _ensure_form(forms_map: dict[str, LanguageForm], form_text: str, lang_raw: str,
                 gloss: str | None, entity_id: str) -> str:
    key = form_text.lower()
    if key not in forms_map:
        forms_map[key] = LanguageForm(id=_uid(), form=form_text, language=_resolve_language(lang_raw),
                                       entity_id=entity_id, gloss=gloss)
    elif gloss and not forms_map[key].gloss:
        forms_map[key].gloss = gloss
    return forms_map[key].id


def _resolve_deriv(raw: str) -> DerivationType:
    for dt in DerivationType:
        if dt.value == raw:
            return dt
    return DerivationType.TRANSLATION


def _extract_llm(text: str) -> list[LinguisticLineage] | None:
    try:
        from book_graph_analyzer.llm import LLMClient
        from book_graph_analyzer.worldbible.lineage import parse_lineage
        llm = LLMClient()
        prompt = (
            "Extract all linguistic/etymology information from the following text.\n"
            "Output JSON: {\"lineages\": [{\"entity_id\": \"...\", \"forms\": [{\"id\": \"lf_...\", "
            "\"form\": \"...\", \"language\": \"...\", \"gloss\": \"...\"}], \"derivations\": [...]}]}\n\n"
            f"Text:\n{text[:4000]}\n\nJSON:"
        )
        response = llm.generate(prompt, temperature=0.2, timeout=60.0)
        if not response:
            return None
        data = llm.extract_json(response)
        if not data or not isinstance(data, dict):
            return None
        return [parse_lineage(item) for item in data.get("lineages", [])]
    except Exception:
        return None


def lineage_alias_hints(lineages: list[LinguisticLineage]) -> dict[str, list[str]]:
    """Derive alias hints from lineages for the entity resolver."""
    hints: dict[str, list[str]] = {}
    for lin in lineages:
        names = [f.form for f in lin.forms]
        if names:
            hints[lin.entity_id] = names
    return hints
