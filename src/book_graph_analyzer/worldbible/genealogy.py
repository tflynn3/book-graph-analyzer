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
from collections import defaultdict, deque
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


_HONORIFICS = {
    "king", "queen", "lord", "lady", "prince", "princess", "sir", "captain",
    "master", "steward", "high", "saint",
}


def _normalize_person_name(raw: str) -> str:
    """Best-effort cleanup for extracted person names.

    Removes leading honorifics and trailing appositive clauses to improve
    canonical-id stability for genealogy extraction.
    """
    name = re.sub(r"\s+", " ", (raw or "").strip(" ,.;:"))
    if not name:
        return name
    # Drop trailing appositives: "Aragorn, son of Arathorn" -> "Aragorn"
    name = re.split(r",\s*(?:son|daughter|child|father|mother|brother|sister|heir)\b", name, maxsplit=1, flags=re.I)[0]
    parts = name.split()
    while parts and parts[0].lower() in _HONORIFICS:
        parts = parts[1:]
    # Trim trailing epithet: "the Tall", "the Younger"
    if len(parts) >= 3 and parts[-2].lower() == "the":
        parts = parts[:-2]
    return " ".join(parts) if parts else name


def _resolve_name(raw: str, seen: dict[str, str]) -> str:
    """Resolve a surface form against names already seen in this passage."""
    cleaned = _normalize_person_name(raw)
    key = cleaned.lower()
    if key in seen:
        return seen[key]
    # Backoff: single-token mention maps to unique prior full-name tail token.
    if " " not in cleaned:
        token = cleaned.lower()
        matches = [v for k, v in seen.items() if k.endswith(f" {token}") or k == token]
        if len(set(matches)) == 1:
            return matches[0]
    seen[key] = cleaned
    return cleaned


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


_HOUSE_PATTERNS = [
    re.compile(r"\b(House of [A-Z][\w'’\-]+(?: [A-Z][\w'’\-]+)*)\b"),
    re.compile(r"\b(?:of|from) (?:the )?(House [A-Z][\w'’\-]+(?: [A-Z][\w'’\-]+)*)\b"),
    re.compile(r"\b(?:of|from) (?:the )?(clan [A-Z][\w'’\-]+(?: [A-Z][\w'’\-]+)*)\b", re.I),
]


def infer_house_from_context(text: str, start: int, end: int, explicit_house: str | None = None) -> str | None:
    """Infer house/clan near a matched genealogy statement."""
    if explicit_house:
        return explicit_house
    window_start = max(0, start - 120)
    window_end = min(len(text), end + 120)
    window = text[window_start:window_end]
    for pat in _HOUSE_PATTERNS:
        m = pat.search(window)
        if m:
            h = m.group(1).strip()
            if h.lower().startswith("clan "):
                return f"Clan {h[6:]}"
            return h
    return None


def infer_generation_depths(relations: list[GenealogyRelation]) -> list[GenealogyRelation]:
    """Infer missing generation_depth via relationship type + graph traversal."""
    parent_to_child: dict[str, set[str]] = defaultdict(set)
    child_to_parent: dict[str, set[str]] = defaultdict(set)

    for r in relations:
        if r.relation_type == GenealogyRelationType.PARENT_OF:
            parent_to_child[r.source_id].add(r.target_id)
            child_to_parent[r.target_id].add(r.source_id)
        elif r.relation_type == GenealogyRelationType.CHILD_OF:
            parent_to_child[r.target_id].add(r.source_id)
            child_to_parent[r.source_id].add(r.target_id)

    def shortest_up(src: str, tgt: str) -> int | None:
        q = deque([(src, 0)])
        seen = {src}
        while q:
            cur, d = q.popleft()
            if cur == tgt:
                return d
            for nxt in child_to_parent.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, d + 1))
        return None

    for r in relations:
        if r.generation_depth is not None:
            continue
        if r.relation_type in (GenealogyRelationType.PARENT_OF, GenealogyRelationType.CHILD_OF):
            r.generation_depth = 1
        elif r.relation_type in (GenealogyRelationType.GRANDPARENT_OF, GenealogyRelationType.GRANDCHILD_OF):
            r.generation_depth = 2
        elif r.relation_type in (GenealogyRelationType.SIBLING_OF, GenealogyRelationType.HALF_SIBLING_OF, GenealogyRelationType.SPOUSE_OF):
            r.generation_depth = 0
        elif r.relation_type == GenealogyRelationType.ANCESTOR_OF:
            d = shortest_up(r.target_id, r.source_id)
            r.generation_depth = d if d is not None else None
        elif r.relation_type == GenealogyRelationType.DESCENDANT_OF:
            d = shortest_up(r.source_id, r.target_id)
            r.generation_depth = d if d is not None else None

    return relations


_NAME = r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)*(?: [A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)*)*"


_RULES: list[tuple[re.Pattern[str], GenealogyRelationType, float]] = [
    (re.compile(rf"\b({_NAME})\s+son of\s+({_NAME})\b"), GenealogyRelationType.CHILD_OF, 0.9),
    (re.compile(rf"\b({_NAME})\s+daughter of\s+({_NAME})\b"), GenealogyRelationType.CHILD_OF, 0.9),
    (re.compile(rf"\b({_NAME})\s+child of\s+({_NAME})\b"), GenealogyRelationType.CHILD_OF, 0.85),
    (re.compile(rf"\b({_NAME}),\s*(?:the\s+)?son of\s+({_NAME})\b"), GenealogyRelationType.CHILD_OF, 0.9),
    (re.compile(rf"\b({_NAME}),\s*(?:the\s+)?daughter of\s+({_NAME})\b"), GenealogyRelationType.CHILD_OF, 0.9),
    (re.compile(rf"\b({_NAME})'s\s+father\s+(?:was\s+)?({_NAME})\b"), GenealogyRelationType.CHILD_OF, 0.8),
    (re.compile(rf"\b({_NAME})'s\s+mother\s+(?:was\s+)?({_NAME})\b"), GenealogyRelationType.CHILD_OF, 0.8),
    (re.compile(rf"\b({_NAME})\s+father of\s+({_NAME})\b"), GenealogyRelationType.PARENT_OF, 0.9),
    (re.compile(rf"\b({_NAME})\s+mother of\s+({_NAME})\b"), GenealogyRelationType.PARENT_OF, 0.9),
    (re.compile(rf"\b({_NAME})\s+brother of\s+({_NAME})\b"), GenealogyRelationType.SIBLING_OF, 0.8),
    (re.compile(rf"\b({_NAME})\s+sister of\s+({_NAME})\b"), GenealogyRelationType.SIBLING_OF, 0.8),
    (re.compile(rf"\b({_NAME})\s+wed\s+({_NAME})\b"), GenealogyRelationType.SPOUSE_OF, 0.75),
    (re.compile(rf"\b({_NAME})\s+married\s+({_NAME})\b"), GenealogyRelationType.SPOUSE_OF, 0.8),
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
    seen_names: dict[str, str] = {}

    for pattern, relation_type, confidence in _RULES:
        for match in pattern.finditer(text):
            source_name = _resolve_name(match.group(1).strip(), seen_names)
            target_name = _resolve_name(match.group(2).strip(), seen_names)
            if source_name == target_name:
                continue
            inferred_house = infer_house_from_context(text, match.start(), match.end(), explicit_house=house)
            rel = _make_relation(source_name, target_name, relation_type, inferred_house, passage_id, confidence)
            _add_with_inverse(relations, rel)

    if relations or llm_client is None:
        return infer_generation_depths(_dedupe_relations(relations))

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

    return infer_generation_depths(_dedupe_relations(relations))


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
    q = deque([(character_id, 0)])
    seen = {character_id}
    out: list[GenealogyRelation] = []
    by_source: dict[str, list[GenealogyRelation]] = defaultdict(list)
    for r in relations:
        by_source[r.source_id].append(r)

    while q:
        node, d = q.popleft()
        if d >= depth:
            continue
        for r in by_source.get(node, []):
            if r.relation_type not in (GenealogyRelationType.CHILD_OF, GenealogyRelationType.DESCENDANT_OF):
                continue
            out.append(r)
            if r.target_id not in seen:
                seen.add(r.target_id)
                q.append((r.target_id, d + 1))
    return out


def build_descendant_tree(relations: list[GenealogyRelation], character_id: str, depth: int = 3) -> list[GenealogyRelation]:
    q = deque([(character_id, 0)])
    seen = {character_id}
    out: list[GenealogyRelation] = []
    by_source: dict[str, list[GenealogyRelation]] = defaultdict(list)
    for r in relations:
        by_source[r.source_id].append(r)

    while q:
        node, d = q.popleft()
        if d >= depth:
            continue
        for r in by_source.get(node, []):
            if r.relation_type not in (GenealogyRelationType.PARENT_OF, GenealogyRelationType.ANCESTOR_OF):
                continue
            out.append(r)
            if r.target_id not in seen:
                seen.add(r.target_id)
                q.append((r.target_id, d + 1))
    return out
