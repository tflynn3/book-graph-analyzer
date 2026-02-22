"""Lore-depth extraction helpers for artifacts and unresolved references."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from ..extract.disambiguation import DisambiguationDict
from ..models.lore_depth import (
    BrokenReference,
    LoreArtifact,
    LoreArtifactType,
    LoreDepthExtractionResult,
    ReferenceCandidate,
)


_ARTIFACT_RE = re.compile(
    r"\b(?P<kind>song|poem|lay|chant|verse|artifact|relic|amulet|ring|sword|crown)\s+(?:of\s+)?(?P<name>[A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*){0,4})"
)

_BROKEN_REF_BRACKETS_RE = re.compile(r"\[\[([^\]]+)\]\]")
_BROKEN_REF_QUAL_RE = re.compile(
    r"\b(?P<text>(unknown|unnamed|forgotten)\s+(artifact|relic|song|poem|name|heir))\b",
    re.IGNORECASE,
)

_EXPECTED_TYPE_RE = re.compile(
    r"\b(?P<t>artifact|relic|song|poem|blade|sword|ring|crown|heir|lord|lady|king|queen|city|fortress)\b",
    re.IGNORECASE,
)


@dataclass
class _Ctx:
    before: str
    match: str
    after: str


def _build_context(text: str, start: int, end: int, context_window: int) -> _Ctx:
    left = max(0, start - context_window)
    right = min(len(text), end + context_window)
    before = text[left:start].strip()
    match = text[start:end]
    after = text[end:right].strip()
    return _Ctx(before=before, match=match, after=after)


def _infer_expected_type(mention: str, context: str) -> str:
    joined = f"{mention} {context}"
    m = _EXPECTED_TYPE_RE.search(joined)
    if not m:
        return "unknown"
    token = m.group("t").lower()
    if token in {"song", "poem"}:
        return token
    if token in {"city", "fortress"}:
        return "place"
    if token in {"lord", "lady", "king", "queen"}:
        return "character"
    return "artifact"


def _fallback_llm_broken_refs(text: str, llm_client: Any) -> list[tuple[str, float, str]]:
    """Best-effort JSON contract: [{mention_text, confidence, reason}]"""
    prompt = (
        "Extract unresolved or broken references from this passage. "
        "Return JSON array only, with objects containing mention_text, confidence (0-1), reason.\n\n"
        f"PASSAGE:\n{text[:3000]}"
    )
    try:
        raw = llm_client.generate(prompt)
        if hasattr(llm_client, "extract_json"):
            data = llm_client.extract_json(raw)
        else:
            data = json.loads(raw)
        if not isinstance(data, list):
            return []
        out = []
        for item in data:
            mention = str(item.get("mention_text", "")).strip()
            if not mention:
                continue
            conf = float(item.get("confidence", 0.55) or 0.55)
            reason = str(item.get("reason", "llm_fallback")).strip() or "llm_fallback"
            out.append((mention, max(0.0, min(1.0, conf)), reason))
        return out
    except Exception:
        return []


def extract_lore_depth(
    text: str,
    source_book: str | None = None,
    passage_id: str | None = None,
    *,
    context_window: int = 80,
    llm_client: Any | None = None,
) -> LoreDepthExtractionResult:
    """Extract lore artifacts and unresolved references from raw text."""
    artifacts: list[LoreArtifact] = []
    broken: list[BrokenReference] = []

    for i, m in enumerate(_ARTIFACT_RE.finditer(text)):
        kind = m.group("kind").lower()
        if kind in {"song", "lay", "chant", "verse"}:
            artifact_type = LoreArtifactType.SONG
        elif kind == "poem":
            artifact_type = LoreArtifactType.POEM
        else:
            artifact_type = LoreArtifactType.ARTIFACT

        artifacts.append(
            LoreArtifact(
                id=f"artifact_{passage_id or 'inline'}_{i}",
                name=m.group("name").strip(),
                artifact_type=artifact_type,
                description=m.group(0),
                source_book=source_book,
                passage_id=passage_id,
            )
        )

    for i, m in enumerate(_BROKEN_REF_BRACKETS_RE.finditer(text)):
        mention = m.group(1).strip()
        ctx = _build_context(text, m.start(), m.end(), context_window)
        expected = _infer_expected_type(mention, f"{ctx.before} {ctx.after}")
        broken.append(
            BrokenReference(
                id=f"broken_{passage_id or 'inline'}_b{i}",
                mention_text=mention,
                context_text=ctx.match,
                context_before=ctx.before,
                context_after=ctx.after,
                expected_type=expected,
                source_book=source_book,
                passage_id=passage_id,
                confidence=0.85,
                provenance_notes=["pattern:brackets"],
                conflict_weight=0.35,
            )
        )

    for i, m in enumerate(_BROKEN_REF_QUAL_RE.finditer(text)):
        ctx = _build_context(text, m.start(), m.end(), context_window)
        broken.append(
            BrokenReference(
                id=f"broken_{passage_id or 'inline'}_q{i}",
                mention_text=m.group("text"),
                context_text=ctx.match,
                context_before=ctx.before,
                context_after=ctx.after,
                expected_type="unresolved_qualifier",
                source_book=source_book,
                passage_id=passage_id,
                confidence=0.65,
                provenance_notes=["pattern:qualifier"],
                conflict_weight=0.20,
            )
        )

    if llm_client is not None:
        seeded = {b.mention_text.lower() for b in broken}
        llm_refs = _fallback_llm_broken_refs(text, llm_client)
        for i, (mention, conf, reason) in enumerate(llm_refs):
            if mention.lower() in seeded:
                continue
            ix = text.lower().find(mention.lower())
            if ix >= 0:
                ctx = _build_context(text, ix, ix + len(mention), context_window)
                match_ctx = mention
            else:
                ctx = _Ctx(before="", match=mention, after="")
                match_ctx = mention
            broken.append(
                BrokenReference(
                    id=f"broken_{passage_id or 'inline'}_m{i}",
                    mention_text=mention,
                    context_text=match_ctx,
                    context_before=ctx.before,
                    context_after=ctx.after,
                    expected_type=_infer_expected_type(mention, f"{ctx.before} {ctx.after}"),
                    source_book=source_book,
                    passage_id=passage_id,
                    confidence=max(0.55, conf),
                    provenance_notes=[f"llm:{reason}"],
                    conflict_weight=0.30,
                )
            )

    return LoreDepthExtractionResult(artifacts=artifacts, broken_references=broken)


def link_broken_reference_candidates(
    broken_references: list[BrokenReference],
    *,
    disambiguation: DisambiguationDict | None = None,
    era: str | None = None,
    book: str | None = None,
    max_candidates: int = 3,
) -> list[BrokenReference]:
    """Attach resolver-backed candidate links to unresolved references."""
    d = disambiguation or DisambiguationDict(load_builtins=True)

    for ref in broken_references:
        mention = ref.mention_text.strip()
        if not mention:
            continue

        candidates: list[ReferenceCandidate] = []

        resolved_id, conf = d.resolve(mention, era=era, book=book)
        if resolved_id:
            candidates.append(
                ReferenceCandidate(
                    canonical_id=resolved_id,
                    surface=mention,
                    confidence=conf,
                    source="resolver:exact",
                )
            )

        mention_l = mention.lower()
        fuzzy_hits: list[tuple[str, str, float]] = []
        for surface, entry in d._entries.items():  # internal access for scoring candidates
            score = fuzz.ratio(mention_l, surface)
            if score < 75:
                continue
            fuzzy_hits.append((surface, str(entry.get("default", "")), score / 100.0))

        fuzzy_hits.sort(key=lambda x: -x[2])
        for surface, cid, score in fuzzy_hits:
            if not cid or any(c.canonical_id == cid for c in candidates):
                continue
            candidates.append(
                ReferenceCandidate(
                    canonical_id=cid,
                    surface=surface,
                    confidence=score,
                    source="resolver:fuzzy",
                )
            )
            if len(candidates) >= max_candidates:
                break

        ref.candidates = candidates
        if not ref.resolved_entity_id and candidates:
            ref.provenance_notes.append("candidate_linked")
            ref.conflict_weight = max(ref.conflict_weight, min(0.6, candidates[0].confidence * 0.5))

    return broken_references


def build_candidate_linking_audit_buckets(
    broken_references: list[BrokenReference],
    *,
    weak_candidate_threshold: float = 0.70,
    ambiguous_delta: float = 0.08,
) -> dict[str, list[str]]:
    """Bucket unresolved references for candidate-linking review workflows."""
    buckets: dict[str, list[str]] = {
        "no_candidates": [],
        "weak_top_candidate": [],
        "ambiguous_top_candidates": [],
        "high_confidence_top_candidate": [],
    }

    for ref in broken_references:
        if ref.resolved_entity_id:
            continue
        if not ref.candidates:
            buckets["no_candidates"].append(ref.id)
            continue

        top = ref.candidates[0]
        if top.confidence < weak_candidate_threshold:
            buckets["weak_top_candidate"].append(ref.id)
            continue

        if len(ref.candidates) > 1:
            delta = top.confidence - ref.candidates[1].confidence
            if delta <= ambiguous_delta:
                buckets["ambiguous_top_candidates"].append(ref.id)
                continue

        buckets["high_confidence_top_candidate"].append(ref.id)

    return buckets


def evaluate_unresolved_quality_gates(
    broken_references: list[BrokenReference],
    *,
    min_context_chars: int = 8,
    min_context_coverage: float = 0.85,
    min_candidate_coverage: float = 0.60,
) -> dict[str, Any]:
    """Evaluate precision/quality gates for unresolved references."""
    unresolved = [r for r in broken_references if not r.resolved_entity_id]
    total = len(unresolved)

    context_ok = 0
    candidate_ok = 0
    low_context_ids: list[str] = []
    no_candidate_ids: list[str] = []

    for ref in unresolved:
        before = (ref.context_before or "").strip()
        after = (ref.context_after or "").strip()
        if len(before) >= min_context_chars or len(after) >= min_context_chars:
            context_ok += 1
        else:
            low_context_ids.append(ref.id)

        if ref.candidates:
            candidate_ok += 1
        else:
            no_candidate_ids.append(ref.id)

    context_coverage = (context_ok / total) if total else 1.0
    candidate_coverage = (candidate_ok / total) if total else 1.0

    candidate_audit_buckets = build_candidate_linking_audit_buckets(unresolved)

    return {
        "passed": (context_coverage >= min_context_coverage and candidate_coverage >= min_candidate_coverage),
        "summary": {
            "total_unresolved": total,
            "context_coverage": round(context_coverage, 4),
            "candidate_coverage": round(candidate_coverage, 4),
            "min_context_coverage": min_context_coverage,
            "min_candidate_coverage": min_candidate_coverage,
        },
        "failures": {
            "low_context": low_context_ids,
            "no_candidates": no_candidate_ids,
        },
        "candidate_linking_audit_buckets": candidate_audit_buckets,
    }
