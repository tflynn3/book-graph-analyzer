"""Genealogy extraction + normalization utilities.

Enhancements:
- Deterministic extraction with local coreference/context carry-over
- Confidence and evidence-span metadata on each extracted relation
- Optional LLM proposal stage with deterministic validation + reason codes
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
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

_TITLE_WORDS = {
    "king", "queen", "lord", "lady", "prince", "princess", "sir", "captain", "master", "steward", "thane", "duke",
}

_PRONOUNS = {"he", "she", "his", "her", "him", "hers"}

LLM_REJECT_SCHEMA = "schema_invalid"
LLM_REJECT_ENTITY = "entity_unresolvable"
LLM_REJECT_EVIDENCE = "evidence_misaligned"
LLM_REJECT_LOW_CONF = "low_confidence"
LLM_REJECT_RELATION = "unsupported_relation"


@dataclass
class _Sentence:
    text: str
    start: int
    end: int


@dataclass
class _ContextState:
    # Recent explicit names (most recent first)
    recent_names: list[str] = field(default_factory=list)
    # title phrase -> canonical name
    title_refs: dict[str, str] = field(default_factory=dict)
    last_subject: str | None = None

    def add_name(self, name: str) -> None:
        n = _normalize_person_name(name)
        if not n:
            return
        self.recent_names = [x for x in self.recent_names if x.lower() != n.lower()]
        self.recent_names.insert(0, n)
        self.recent_names = self.recent_names[:6]

    def bind_title(self, title_phrase: str, name: str) -> None:
        t = re.sub(r"\s+", " ", title_phrase.strip().lower())
        if t:
            self.title_refs[t] = name


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
    name = re.sub(r"\s+", " ", (raw or "").strip(" ,.;:"))
    if not name:
        return name
    name = re.split(r",\s*(?:son|daughter|child|father|mother|brother|sister|heir)\b", name, maxsplit=1, flags=re.I)[0]
    parts = name.split()
    while parts and parts[0].lower() in _HONORIFICS:
        parts = parts[1:]
    if len(parts) >= 3 and parts[-2].lower() == "the":
        parts = parts[:-2]
    return " ".join(parts) if parts else name


def _resolve_name(raw: str, seen: dict[str, str]) -> str:
    cleaned = _normalize_person_name(raw)
    key = cleaned.lower()
    if key in seen:
        return seen[key]
    if " " not in cleaned:
        token = cleaned.lower()
        matches = [v for k, v in seen.items() if k.endswith(f" {token}") or k == token]
        if len(set(matches)) == 1:
            return matches[0]
    seen[key] = cleaned
    return cleaned


def _split_sentences(text: str) -> list[_Sentence]:
    out: list[_Sentence] = []
    for m in re.finditer(r"[^.!?]+[.!?]?", text):
        frag = m.group(0)
        if frag.strip():
            out.append(_Sentence(text=frag, start=m.start(), end=m.end()))
    return out


def _extract_title_bindings(sentence: str, ctx: _ContextState, seen_names: dict[str, str]) -> None:
    # "King Aragorn" => bind "king" and "the king" to Aragorn
    for m in re.finditer(r"\b(?P<title>King|Queen|Lord|Lady|Prince|Princess|Captain|Steward)\s+(?P<name>[A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)*)", sentence):
        nm = _resolve_name(m.group("name"), seen_names)
        tl = m.group("title").lower()
        ctx.bind_title(tl, nm)
        ctx.bind_title(f"the {tl}", nm)
        ctx.add_name(nm)


def _resolve_mention(raw: str, ctx: _ContextState, seen_names: dict[str, str]) -> tuple[str | None, float]:
    token = re.sub(r"\s+", " ", (raw or "").strip(" ,.;:")).lower()
    if not token:
        return None, 0.0

    if token in _PRONOUNS:
        if ctx.last_subject:
            return ctx.last_subject, 0.76
        if ctx.recent_names:
            return ctx.recent_names[0], 0.72
        return None, 0.0

    if token in ctx.title_refs:
        return ctx.title_refs[token], 0.86

    if token.startswith("the ") and token in ctx.title_refs:
        return ctx.title_refs[token], 0.84

    # explicit name path
    if re.match(r"[A-Z]", raw.strip()[:1]):
        nm = _resolve_name(raw, seen_names)
        ctx.add_name(nm)
        return nm, 0.95

    return None, 0.0


def _make_relation(
    source_name: str,
    target_name: str,
    relation_type: GenealogyRelationType,
    house: str | None,
    passage_id: str | None,
    confidence: float,
    evidence_text: str | None = None,
    evidence_start: int | None = None,
    evidence_end: int | None = None,
    resolution_confidence: float | None = None,
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
        evidence_text=evidence_text,
        evidence_start=evidence_start,
        evidence_end=evidence_end,
        resolution_confidence=resolution_confidence,
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
            evidence_text=relation.evidence_text,
            evidence_start=relation.evidence_start,
            evidence_end=relation.evidence_end,
            resolution_confidence=relation.resolution_confidence,
        )
    )


_HOUSE_PATTERNS = [
    re.compile(r"\b(House of [A-Z][\w'’\-]+(?: [A-Z][\w'’\-]+)*)\b"),
    re.compile(r"\b(?:of|from) (?:the )?(House [A-Z][\w'’\-]+(?: [A-Z][\w'’\-]+)*)\b"),
    re.compile(r"\b(?:of|from) (?:the )?(clan [A-Z][\w'’\-]+(?: [A-Z][\w'’\-]+)*)\b", re.I),
]


def infer_house_from_context(text: str, start: int, end: int, explicit_house: str | None = None) -> str | None:
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


_NAME = r"([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)*)"
_SUBJ = rf"(?P<source>{_NAME}|[Hh]e|[Ss]he|[Hh]is|[Hh]er|(?:[Tt]he\s+)?(?:king|queen|lord|lady|prince|princess|captain|steward))"
_OBJ = rf"(?P<target>{_NAME}|(?:[Tt]he\s+)?(?:king|queen|lord|lady|prince|princess|captain|steward))"
_RULES: list[tuple[re.Pattern[str], GenealogyRelationType, float]] = [
    (re.compile(rf"\b{_SUBJ}\s+son of\s+{_OBJ}\b"), GenealogyRelationType.CHILD_OF, 0.9),
    (re.compile(rf"\b{_SUBJ}\s+daughter of\s+{_OBJ}\b"), GenealogyRelationType.CHILD_OF, 0.9),
    (re.compile(rf"\b{_SUBJ}\s+child of\s+{_OBJ}\b"), GenealogyRelationType.CHILD_OF, 0.85),
    (re.compile(rf"\b{_SUBJ}\s+(?:was\s+)?father of\s+{_OBJ}\b"), GenealogyRelationType.PARENT_OF, 0.9),
    (re.compile(rf"\b{_SUBJ}\s+(?:was\s+)?mother of\s+{_OBJ}\b"), GenealogyRelationType.PARENT_OF, 0.9),
    (re.compile(rf"\b{_SUBJ}\s+brother of\s+{_OBJ}\b"), GenealogyRelationType.SIBLING_OF, 0.8),
    (re.compile(rf"\b{_SUBJ}\s+sister of\s+{_OBJ}\b"), GenealogyRelationType.SIBLING_OF, 0.8),
    (re.compile(rf"\b{_SUBJ}\s+wed\s+{_OBJ}\b"), GenealogyRelationType.SPOUSE_OF, 0.75),
    (re.compile(rf"\b{_SUBJ}\s+married\s+{_OBJ}\b"), GenealogyRelationType.SPOUSE_OF, 0.8),
    (re.compile(rf"\b{_NAME}\s*,\s*son of\s+{_OBJ}\b"), GenealogyRelationType.CHILD_OF, 0.92),
    (re.compile(rf"\b{_NAME}\s*,\s*daughter of\s+{_OBJ}\b"), GenealogyRelationType.CHILD_OF, 0.92),
    (re.compile(rf"\b{_NAME}\s*,\s*the son of\s+{_NAME}\b"), GenealogyRelationType.CHILD_OF, 0.92),
    (re.compile(rf"\b{_NAME}\s*,\s*the daughter of\s+{_NAME}\b"), GenealogyRelationType.CHILD_OF, 0.92),
    (re.compile(rf"\b{_NAME}\s+was\s+the son of\s+{_NAME}\b"), GenealogyRelationType.CHILD_OF, 0.9),
    (re.compile(rf"\b{_NAME}\s+was\s+the daughter of\s+{_NAME}\b"), GenealogyRelationType.CHILD_OF, 0.9),
    (re.compile(rf"\b{_NAME}\s+grandson of\s+{_NAME}\b"), GenealogyRelationType.DESCENDANT_OF, 0.86),
    (re.compile(rf"\b{_NAME}\s+granddaughter of\s+{_NAME}\b"), GenealogyRelationType.DESCENDANT_OF, 0.86),
    (re.compile(rf"\b{_NAME}\s+heir of\s+{_NAME}\b"), GenealogyRelationType.DESCENDANT_OF, 0.78),
    # TT/ROTK high-value forms
    (re.compile(rf"\b{_SUBJ}\s+is\s+the\s+son of\s+{_OBJ}\b", re.I), GenealogyRelationType.CHILD_OF, 0.9),
    (re.compile(rf"\b{_SUBJ}\s+is\s+the\s+daughter of\s+{_OBJ}\b", re.I), GenealogyRelationType.CHILD_OF, 0.9),
    (re.compile(rf"\b{_SUBJ}\s+is\s+heir to\s+{_OBJ}\b", re.I), GenealogyRelationType.DESCENDANT_OF, 0.8),
    (re.compile(rf"\b{_SUBJ}\s+descended from\s+{_OBJ}\b", re.I), GenealogyRelationType.DESCENDANT_OF, 0.84),
    (re.compile(rf"\b{_SUBJ}\s+of the line of\s+{_OBJ}\b", re.I), GenealogyRelationType.DESCENDANT_OF, 0.82),
    (re.compile(rf"\b{_SUBJ}\s+brother to\s+{_OBJ}\b", re.I), GenealogyRelationType.SIBLING_OF, 0.8),
    (re.compile(rf"\b{_SUBJ}\s+sister to\s+{_OBJ}\b", re.I), GenealogyRelationType.SIBLING_OF, 0.8),
]


_KINSHIP_CUES = re.compile(
    r"\b(son of|daughter of|child of|father of|mother of|brother of|sister of|"
    r"married|wed|wedded|grandson of|granddaughter of|heir of|line of|descended from)\b",
    re.I,
)


def validate_llm_genealogy_proposals(
    text: str,
    proposals: list[dict[str, Any]],
    known_entities: set[str] | None = None,
    min_confidence: float = 0.65,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    entities = {e.lower() for e in (known_entities or set())}

    for item in proposals:
        reason = None
        source = str(item.get("source_name", "")).strip()
        target = str(item.get("target_name", "")).strip()
        rel_raw = str(item.get("relation_type", "")).strip()
        evidence_text = str(item.get("evidence_text", "")).strip()
        ev_start_raw = item.get("evidence_start")
        ev_end_raw = item.get("evidence_end")
        try:
            ev_start = int(ev_start_raw)
            ev_end = int(ev_end_raw)
        except Exception:
            ev_start = ev_end = -1
        conf = float(item.get("confidence", 0.7) or 0.7)

        if not source or not target or not rel_raw:
            reason = LLM_REJECT_SCHEMA
        elif source == target:
            reason = LLM_REJECT_ENTITY
        elif conf < min_confidence:
            reason = LLM_REJECT_LOW_CONF

        try:
            rel_type = normalize_relation_type(rel_raw)
        except Exception:
            rel_type = None
        if reason is None and rel_type is None:
            reason = LLM_REJECT_RELATION

        # entity resolvability guardrail
        if reason is None and entities:
            if source.lower() not in entities and target.lower() not in entities:
                reason = LLM_REJECT_ENTITY

        # Backward-compat fallback for legacy LLM schema without evidence span fields.
        if reason is None and (ev_start < 0 or ev_end < 0):
            src_i = text.lower().find(source.lower())
            tgt_i = text.lower().find(target.lower())
            if src_i >= 0 and tgt_i >= 0:
                ev_start = min(src_i, tgt_i)
                ev_end = max(src_i + len(source), tgt_i + len(target))
                evidence_text = text[ev_start:ev_end]

        # evidence span alignment guardrail
        if reason is None:
            if not (0 <= ev_start < ev_end <= len(text)):
                reason = LLM_REJECT_EVIDENCE
            else:
                span = text[ev_start:ev_end]
                if evidence_text and evidence_text not in span and span not in evidence_text:
                    reason = LLM_REJECT_EVIDENCE
                elif source.lower() not in span.lower() and target.lower() not in span.lower():
                    reason = LLM_REJECT_EVIDENCE

        if reason is None:
            accepted.append({
                "source_name": source,
                "target_name": target,
                "relation_type": rel_type,
                "evidence_text": evidence_text or text[ev_start:ev_end],
                "evidence_start": ev_start,
                "evidence_end": ev_end,
                "confidence": conf,
            })
        else:
            item = dict(item)
            item["reason_code"] = reason
            rejected.append(item)
    return accepted, rejected


def extract_genealogy_from_text(
    text: str,
    passage_id: str | None = None,
    house: str | None = None,
    llm_client: Any | None = None,
    min_relations_for_fallback: int = 2,
) -> list[GenealogyRelation]:
    relations: list[GenealogyRelation] = []
    seen_names: dict[str, str] = {}
    ctx = _ContextState()

    for sentence in _split_sentences(text):
        _extract_title_bindings(sentence.text, ctx, seen_names)
        for pattern, relation_type, base_confidence in _RULES:
            for match in pattern.finditer(sentence.text):
                source_raw = match.group("source") if "source" in match.groupdict() else match.group(1)
                target_raw = match.group("target") if "target" in match.groupdict() else match.group(2)

                source_name, source_rc = _resolve_mention(source_raw, ctx, seen_names)
                target_name, target_rc = _resolve_mention(target_raw, ctx, seen_names)
                if not source_name or not target_name or source_name == target_name:
                    continue

                local_start = sentence.start + match.start()
                local_end = sentence.start + match.end()
                inferred_house = infer_house_from_context(text, local_start, local_end, explicit_house=house)
                resolution_conf = min(source_rc, target_rc)
                conf = round(max(0.0, min(1.0, base_confidence * (0.85 + 0.2 * resolution_conf))), 3)
                rel = _make_relation(
                    source_name,
                    target_name,
                    relation_type,
                    inferred_house,
                    passage_id,
                    conf,
                    evidence_text=text[local_start:local_end],
                    evidence_start=local_start,
                    evidence_end=local_end,
                    resolution_confidence=resolution_conf,
                )
                _add_with_inverse(relations, rel)
                ctx.last_subject = source_name
                ctx.add_name(source_name)
                ctx.add_name(target_name)

    # LLM proposal stage (optional, strictly validated, only when useful)
    deduped = _dedupe_relations(relations)
    if llm_client is not None and _KINSHIP_CUES.search(text) and len(deduped) < min_relations_for_fallback:
        entities = set(seen_names.values())
        prompt = (
            "Extract genealogy relations from this passage as JSON array. Each object MUST include "
            "source_name,target_name,relation_type,evidence_text,evidence_start,evidence_end,confidence. "
            "relation_type one of PARENT_OF,CHILD_OF,SIBLING_OF,SPOUSE_OF,ANCESTOR_OF,DESCENDANT_OF,"
            "FOSTER_PARENT_OF,FOSTER_CHILD_OF,HALF_SIBLING_OF. "
            "Only include relations explicitly supported by the span.\n\n"
            f"Passage:\n{text}"
        )
        try:
            raw = llm_client.generate(prompt)
            payload = json.loads(raw)
            if isinstance(payload, dict):
                proposals = payload.get("relations", [])
            else:
                proposals = payload if isinstance(payload, list) else []
            accepted, _rejected = validate_llm_genealogy_proposals(text, proposals, known_entities=entities or None)
            for item in accepted:
                inferred_house = infer_house_from_context(
                    text,
                    int(item["evidence_start"]),
                    int(item["evidence_end"]),
                    explicit_house=house,
                )
                rel = _make_relation(
                    source_name=item["source_name"],
                    target_name=item["target_name"],
                    relation_type=item["relation_type"],
                    house=inferred_house,
                    passage_id=passage_id,
                    confidence=float(item["confidence"]),
                    evidence_text=str(item["evidence_text"]),
                    evidence_start=int(item["evidence_start"]),
                    evidence_end=int(item["evidence_end"]),
                    resolution_confidence=0.7,
                )
                _add_with_inverse(deduped, rel)
        except Exception:
            pass
    return infer_generation_depths(_dedupe_relations(deduped))


def _dedupe_relations(relations: list[GenealogyRelation]) -> list[GenealogyRelation]:
    seen: set[tuple[str, str, str, int | None, int | None]] = set()
    out: list[GenealogyRelation] = []
    for rel in relations:
        # keep at most one relation per identity + evidence span; this prevents
        # duplicate writes when multiple patterns match the same text span while
        # preserving distinct evidentiary occurrences.
        key = (
            rel.source_id,
            rel.target_id,
            rel.relation_type.value,
            rel.evidence_start,
            rel.evidence_end,
        )
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
        evidence_text=data.get("evidence_text"),
        evidence_start=data.get("evidence_start"),
        evidence_end=data.get("evidence_end"),
        resolution_confidence=data.get("resolution_confidence"),
    )


def load_genealogy_from_file(path: str | Path) -> list[GenealogyRelation]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("relations", data if isinstance(data, list) else [])
    return [parse_genealogy_relation(item) for item in raw]


def genealogy_to_json(relations: list[GenealogyRelation]) -> dict[str, Any]:
    unique_character_ids = {r.source_id for r in relations} | {r.target_id for r in relations}
    distinct_houses = sorted({r.house for r in relations if r.house})
    return {
        "metrics": {
            "relation_count": len(relations),
            "unique_character_count": len(unique_character_ids),
            "distinct_house_count": len(distinct_houses),
            "houses": distinct_houses,
        },
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
                "evidence_text": r.evidence_text,
                "evidence_start": r.evidence_start,
                "evidence_end": r.evidence_end,
                "resolution_confidence": r.resolution_confidence,
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
