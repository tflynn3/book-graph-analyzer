"""Linguistic lineage parsing and loading utilities.

Parses structured JSON files containing linguistic lineage data
into LinguisticLineage model objects for graph persistence.

Example JSON format::

    {
      "lineages": [
        {
          "entity_id": "place_rivendell",
          "forms": [
            {"id": "lf_imladris", "form": "Imladris", "language": "Sindarin",
             "gloss": "Deep dale of the cleft"},
            {"id": "lf_rivendell", "form": "Rivendell", "language": "Common Speech",
             "gloss": "Cloven valley"},
            {"id": "lf_karningul", "form": "Karningul", "language": "Westron"}
          ],
          "derivations": [
            {"source_form_id": "lf_rivendell", "target_form_id": "lf_imladris",
             "derivation_type": "translation"},
            {"source_form_id": "lf_karningul", "target_form_id": "lf_imladris",
             "derivation_type": "adaptation"}
          ]
        }
      ]
    }

See Issue #46.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from book_graph_analyzer.models.worldbuilding import (
    DerivationType,
    LanguageDerivation,
    LanguageForm,
    LinguisticLineage,
    TolkienLanguage,
)


def _resolve_language(raw: str) -> TolkienLanguage:
    """Resolve a language string to TolkienLanguage enum, falling back to OTHER."""
    # Try exact value match
    for lang in TolkienLanguage:
        if lang.value.lower() == raw.lower():
            return lang
    # Try member name match
    try:
        return TolkienLanguage[raw.upper().replace(" ", "_")]
    except KeyError:
        return TolkienLanguage.OTHER


def _resolve_derivation_type(raw: str) -> DerivationType:
    """Resolve a derivation type string."""
    for dt in DerivationType:
        if dt.value.lower() == raw.lower():
            return dt
    return DerivationType.TRANSLATION  # safe default


def parse_language_form(data: dict[str, Any]) -> LanguageForm:
    """Parse a single LanguageForm from a dict."""
    return LanguageForm(
        id=data["id"],
        form=data["form"],
        language=_resolve_language(data["language"]),
        entity_id=data.get("entity_id"),
        gloss=data.get("gloss"),
        phonetic=data.get("phonetic"),
        source_passage_id=data.get("source_passage_id"),
    )


def parse_derivation(data: dict[str, Any]) -> LanguageDerivation:
    """Parse a single LanguageDerivation from a dict."""
    return LanguageDerivation(
        source_form_id=data["source_form_id"],
        target_form_id=data["target_form_id"],
        derivation_type=_resolve_derivation_type(data.get("derivation_type", "translation")),
        notes=data.get("notes"),
    )


def parse_lineage(data: dict[str, Any]) -> LinguisticLineage:
    """Parse a single LinguisticLineage from a dict.

    The dict must have ``entity_id`` and ``forms``.
    ``derivations`` is optional.
    """
    forms = [parse_language_form(f) for f in data.get("forms", [])]

    # Auto-populate entity_id on forms if not set
    entity_id = data["entity_id"]
    for form in forms:
        if form.entity_id is None:
            form.entity_id = entity_id

    derivations = [parse_derivation(d) for d in data.get("derivations", [])]

    return LinguisticLineage(
        entity_id=entity_id,
        forms=forms,
        derivations=derivations,
    )


def load_lineages_from_file(path: str | Path) -> list[LinguisticLineage]:
    """Load linguistic lineages from a JSON file.

    Expects ``{"lineages": [...]}`` at the top level.

    Args:
        path: Path to JSON file

    Returns:
        List of parsed LinguisticLineage objects
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_lineages = data.get("lineages", [])
    return [parse_lineage(item) for item in raw_lineages]


def lineages_to_json(lineages: list[LinguisticLineage]) -> dict[str, Any]:
    """Serialize lineages to a JSON-compatible dict."""
    return {
        "lineages": [
            {
                "entity_id": lin.entity_id,
                "forms": [
                    {
                        "id": f.id,
                        "form": f.form,
                        "language": f.language.value if hasattr(f.language, "value") else str(f.language),
                        "entity_id": f.entity_id,
                        "gloss": f.gloss,
                        "phonetic": f.phonetic,
                        "source_passage_id": f.source_passage_id,
                    }
                    for f in lin.forms
                ],
                "derivations": [
                    {
                        "source_form_id": d.source_form_id,
                        "target_form_id": d.target_form_id,
                        "derivation_type": d.derivation_type.value if hasattr(d.derivation_type, "value") else str(d.derivation_type),
                        "notes": d.notes,
                    }
                    for d in lin.derivations
                ],
            }
            for lin in lineages
        ]
    }
