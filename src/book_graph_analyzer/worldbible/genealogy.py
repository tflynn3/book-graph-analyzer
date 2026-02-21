"""Genealogy extraction + normalization utilities.

Issue #49 Slice 1 MVP:
- Relationship models normalization
- Regex/rule extraction for family relations
- Optional LLM fallback (best-effort, never required)
- JSON load/save helpers
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from book_graph_analyzer.models.worldbuilding import (
    GenealogyRelation,
    GenealogyRelationType,
)


_RELATION_NORMALIZATION: dict[str, GenealogyRelationType] = {
    "father": GenealogyRelationType.PARENT_OF,
    "mother": GenealogyRelationType.PARENT_OF,
    "parent": GenealogyRelationType.PARENT_OF,
    "son": GenealogyRelationType.CHILD_OF,
    "daughter": GenealogyRelationType.CHILD_OF,
    "child": GenealogyRelationType.CHILD_OF,
    "brother": GenealogyRelationType.SIBLING_OF,
    "sister": GenealogyRelationType.SIBLING_OF,
    "sibling": GenealogyRelationType.SIBLING_OF,
    "spouse": GenealogyRelationType.SPOUSE_OF,
    "wife": GenealogyRelationType.SPOUSE_OF,
    "husband": GenealogyRelationType.SPOUSE_OF,
    "ancestor": GenealogyRelationType.ANCESTOR_OF,
    "descendant": GenealogyRelationType.DESCENDANT_OF,
    "foster father": GenealogyRelationType.FOSTER_PARENT_OF,
    "foster mother": GenealogyRelationType.FOSTER_PARENT_OF,
    "foster parent": GenealogyRelationType.FOSTER_PARENT_OF,
    "foster child": GenealogyRelationType.FOSTER_CHILD_OF,
}

_INVERSE_RELATION: dict[GenealogyRelationType, GenealogyRelationType] = {
    GenealogyRelationType.PARENT_OF: GenealogyRelationType.CHILD_OF,
    GenealogyRelationType.CHILD_OF: GenealogyRelationType.PARENT_OF,
    GenealogyRelationType.SIBLING_OF: GenealogyRelationType.SIBLING_OF,
    GenealogyRelationType.SPOUSE_OF: GenealogyRelationType.SPOUSE_OF,
    GenealogyRelationType.GRANDPARENT_OF: GenealogyRelationType.GRANDCHILD_OF,
    GenealogyRelationType.GRANDCHILD_OF: GenealogyRelationType.GRANDPARENT_OF,
    GenealogyRelationType.ANCESTOR_OF: GenealogyRelationType.DESCENDANT_OF,
    GenealogyRelationType.DESCENDANT_OF: GenealogyRelationType.ANCESTOR_OF,
    GenealogyRelationType.FOSTER_PARENT_OF: GenealogyRelationType.FOSTER_CHILD_OF,
    GenealogyRelationType.FOSTER_CHILD_OF: GenealogyRelationType.FOSTER_PARENT_OF,
    GenealogyRelationType.HALF_SIBLING_OF: GenealogyRelationType.HALF_SIBLING_OF,
}


def normalize_relation_type(raw: str | GenealogyRelationType) -> GenealogyRelationType:
    if isinstance(raw, GenealogyRelationType):
        return raw

    token = (raw or "").strip().lower().replace("_", " ")
    if not token:
        return GenealogyRelationType.PARENT_OF

    if token in _RELATION_NORMALIZATION:
        return _RELATION_NORMALIZATION[token]

    try:
        return GenealogyRelationType[token.upper()]
    except KeyError:
        return GenealogyRelationType.PARENT_OF


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return s or "unknown"


def _canon_id(name: str) -> str:
    return f"char_{_slug(name)}"


def _make_relation(
    source_name: str,
    target_name: str,
    relation_type: GenealogyRelationType,
    house: str | None,
    passage_id: str | None,
    confidence: float,
) -> GenealogyRelation:
    return GenealogyRelation(
        source_id=_canon_id(source_name),
        source_name=source_name,
        target_id=_canon_id(target_name),
        target_name=target_name,
        relation_type=relation_type,
        house=house,
        passage_ids=[passage_id] if passage_id else [],
        confidence=confidence,
    )


def _add_with_inverse(relations: list[GenealogyRelation], relation: GenealogyRelation) -> None:
    relations.append(relation)
    relations.append(
        GenealogyRelation(
            source_id=relation.target_id,
            source_name=relation.target_name,
            target_id=relation.source_id,
            target_name=relation.source_name,
            relation_type=_INVERSE_RELATION[relation.relation_type],
            generation_depth=relation.generation_depth,
            house=relation.house,
            inheritance_traits=list(relation.inheritance_traits),
            era=relation.era,
            passage_ids=list(relation.passage_ids),
            confidence=relation.confidence,
        )
    )


_RULES: list[tuple[re.Pattern[str], GenealogyRelationType, float]] = [
    (re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+son of\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b"), GenealogyRelationType.CHILD_OF, 0.9),
    (re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+daughter of\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b"), GenealogyRelationType.CHILD_OF, 0.9),
    (re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+child of\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b"), GenealogyRelationType.CHILD_OF, 0.85),
    (re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+father of\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b"), GenealogyRelationType.PARENT_OF, 0.9),
    (re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+mother of\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b"), GenealogyRelationType.PARENT_OF, 0.9),
    (re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+brother of\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b"), GenealogyRelationType.SIBLING_OF, 0.8),
    (re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+sister of\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b"), GenealogyRelationType.SIBLING_OF, 0.8),
    (re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+wed\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b"), GenealogyRelationType.SPOUSE_OF, 0.75),
    (re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+married\s+([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b"), GenealogyRelationType.SPOUSE_OF, 0.8),
]


def extract_genealogy_from_text(
    text: str,
    passage_id: str | None = None,
    house: str | None = None,
    llm_client: Any | None = None,
) -> list[GenealogyRelation]:
    """Extract family relations from free text.

    Uses deterministic regex rules first. If no relations are found and
    ``llm_client`` is provided, tries a best-effort JSON fallback.
    """
    relations: list[GenealogyRelation] = []

    for pattern, relation_type, confidence in _RULES:
        for match in pattern.finditer(text):
            source_name, target_name = match.group(1).strip(), match.group(2).strip()
            if source_name == target_name:
                continue
            rel = _make_relation(source_name, target_name, relation_type, house, passage_id, confidence)
            _add_with_inverse(relations, rel)

    if relations or llm_client is None:
        return _dedupe_relations(relations)

    # Optional fallback: extremely defensive JSON contract.
    prompt = (
        "Extract genealogy relations from this passage as JSON array with objects "
        "{source_name,target_name,relation_type}. relation_type must be one of: "
        "PARENT_OF,CHILD_OF,SIBLING_OF,SPOUSE_OF,ANCESTOR_OF,DESCENDANT_OF.\n\n"
        f"Passage:\n{text}"
    )
    try:
        raw = llm_client.generate(prompt)
        payload = json.loads(raw)
        if isinstance(payload, dict):
            payload = payload.get("relations", [])
        for item in payload if isinstance(payload, list) else []:
            rel_type = normalize_relation_type(str(item.get("relation_type", "")))
            source_name = str(item.get("source_name", "")).strip()
            target_name = str(item.get("target_name", "")).strip()
            if not source_name or not target_name or source_name == target_name:
                continue
            rel = _make_relation(source_name, target_name, rel_type, house, passage_id, 0.6)
            _add_with_inverse(relations, rel)
    except Exception:
        pass

    return _dedupe_relations(relations)


def _dedupe_relations(relations: list[GenealogyRelation]) -> list[GenealogyRelation]:
    seen: set[tuple[str, str, str]] = set()
    out: list[GenealogyRelation] = []
    for rel in relations:
        key = (rel.source_id, rel.target_id, rel.relation_type.value)
        if key in seen:
            continue
        seen.add(key)
        out.append(rel)
    return out


def parse_genealogy_relation(data: dict[str, Any]) -> GenealogyRelation:
    return GenealogyRelation(
        source_id=data.get("source_id") or _canon_id(data.get("source_name", "unknown")),
        source_name=data.get("source_name"),
        target_id=data.get("target_id") or _canon_id(data.get("target_name", "unknown")),
        target_name=data.get("target_name"),
        relation_type=normalize_relation_type(data.get("relation_type", "PARENT_OF")),
        generation_depth=data.get("generation_depth"),
        house=data.get("house"),
        inheritance_traits=list(data.get("inheritance_traits") or []),
        era=data.get("era"),
        passage_ids=list(data.get("passage_ids") or []),
        confidence=float(data.get("confidence", 1.0) or 1.0),
    )


def load_genealogy_from_file(path: str | Path) -> list[GenealogyRelation]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("relations", data if isinstance(data, list) else [])
    return [parse_genealogy_relation(item) for item in raw]


def genealogy_to_json(relations: list[GenealogyRelation]) -> dict[str, Any]:
    return {
        "relations": [
            {
                "source_id": r.source_id,
                "source_name": r.source_name,
                "target_id": r.target_id,
                "target_name": r.target_name,
                "relation_type": r.relation_type.value,
                "generation_depth": r.generation_depth,
                "house": r.house,
                "inheritance_traits": r.inheritance_traits,
                "era": r.era,
                "passage_ids": r.passage_ids,
                "confidence": r.confidence,
            }
            for r in relations
        ]
    }


def build_ancestor_chain(relations: list[GenealogyRelation], character_id: str, depth: int = 3) -> list[GenealogyRelation]:
    return [
        r for r in relations
        if r.source_id == character_id and r.relation_type in (GenealogyRelationType.CHILD_OF, GenealogyRelationType.DESCENDANT_OF)
    ][:depth]


def build_descendant_tree(relations: list[GenealogyRelation], character_id: str, depth: int = 3) -> list[GenealogyRelation]:
    return [
        r for r in relations
        if r.source_id == character_id and r.relation_type in (GenealogyRelationType.PARENT_OF, GenealogyRelationType.ANCESTOR_OF)
    ][:depth]
