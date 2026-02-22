"""Editorial-layer helpers: passage tagging and basic divergence detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from statistics import mean

from ..models.passage import Passage


@dataclass
class EditorialDivergence:
    """A potential contradiction between passages from different strata/sources."""

    kind: str  # factual | style
    signal: str
    key: str
    involved_passage_ids: list[str] = field(default_factory=list)
    involved_sources: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ProvenanceValidationResult:
    """Validation summary for editorial provenance completeness."""

    checked_count: int = 0
    missing_count: int = 0
    invalid_authority_count: int = 0
    max_missing_ratio: float = 0.05
    min_authority_weight: float = 0.0

    @property
    def missing_ratio(self) -> float:
        if self.checked_count == 0:
            return 0.0
        return self.missing_count / self.checked_count

    @property
    def is_valid(self) -> bool:
        return (
            self.missing_ratio <= self.max_missing_ratio
            and self.invalid_authority_count == 0
        )


def validate_editorial_provenance(
    passages: list[Passage],
    *,
    max_missing_ratio: float = 0.05,
    min_authority_weight: float = 0.0,
) -> ProvenanceValidationResult:
    """Validate provenance invariants for passages carrying factual claims.

    Required fields for claim-bearing passages:
    - source_id
    - source_stratum
    - source_authority_weight (0..1 and >= min_authority_weight)
    - provenance_tags (non-empty)
    """

    checked = 0
    missing = 0
    invalid_authority = 0

    for p in passages:
        if not (p.factual_claims or {}):
            continue
        checked += 1
        has_required = bool(
            p.source_id
            and p.source_stratum
            and p.provenance_tags
            and p.source_authority_weight is not None
        )
        if not has_required:
            missing += 1
            continue

        w = float(p.source_authority_weight)
        if w < min_authority_weight or w < 0.0 or w > 1.0:
            invalid_authority += 1

    return ProvenanceValidationResult(
        checked_count=checked,
        missing_count=missing,
        invalid_authority_count=invalid_authority,
        max_missing_ratio=max_missing_ratio,
        min_authority_weight=min_authority_weight,
    )


def detect_editorial_divergences(passages: list[Passage]) -> list[EditorialDivergence]:
    """Detect lightweight cross-strata divergence signals.

    MVP heuristics:
    - factual: same claim key appears with different values across passages
    - style: same book/source family but large shifts in sentence length/passive/dialogue
    """

    divergences: list[EditorialDivergence] = []

    # --- factual contradictions ---
    by_claim: dict[str, dict[str, list[Passage]]] = defaultdict(lambda: defaultdict(list))
    for p in passages:
        for key, value in (p.factual_claims or {}).items():
            by_claim[key][str(value).strip().lower()].append(p)

    for key, value_groups in by_claim.items():
        if len(value_groups) <= 1:
            continue
        ids: list[str] = []
        sources: set[str] = set()
        for plist in value_groups.values():
            for p in plist:
                ids.append(p.id)
                sources.add(p.source_id or p.source_title or p.book)
        divergences.append(
            EditorialDivergence(
                kind="factual",
                signal=f"Conflicting values for '{key}': {', '.join(sorted(value_groups.keys()))}",
                key=key,
                involved_passage_ids=sorted(set(ids)),
                involved_sources=sorted(sources),
                confidence=min(1.0, 0.55 + 0.1 * (len(value_groups) - 1)),
            )
        )

    # --- style drift across strata ---
    by_source: dict[str, list[Passage]] = defaultdict(list)
    for p in passages:
        source_key = p.source_id or p.source_title or p.book
        if source_key:
            by_source[source_key].append(p)

    for source_key, group in by_source.items():
        by_stratum: dict[str, list[Passage]] = defaultdict(list)
        for p in group:
            by_stratum[p.source_stratum or "core_text"].append(p)
        if len(by_stratum) < 2:
            continue

        strata_stats = {}
        for stratum, plist in by_stratum.items():
            strata_stats[stratum] = {
                "avg_sentence_length": mean([(pp.avg_sentence_length or 0.0) for pp in plist]),
                "passive_ratio": mean([(pp.passive_ratio or 0.0) for pp in plist]),
                "dialogue_density": mean([(pp.dialogue_density or 0.0) for pp in plist]),
                "ids": [pp.id for pp in plist],
            }

        ordered = sorted(strata_stats.items(), key=lambda x: x[0])
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a_name, a = ordered[i]
                b_name, b = ordered[j]
                if abs(a["avg_sentence_length"] - b["avg_sentence_length"]) >= 12:
                    divergences.append(
                        EditorialDivergence(
                            kind="style",
                            signal=(
                                f"Sentence-length drift between strata {a_name} and {b_name} "
                                f"for {source_key}"
                            ),
                            key="avg_sentence_length",
                            involved_passage_ids=sorted(set(a["ids"] + b["ids"])),
                            involved_sources=[source_key],
                            confidence=0.6,
                        )
                    )
                if abs(a["passive_ratio"] - b["passive_ratio"]) >= 0.25:
                    divergences.append(
                        EditorialDivergence(
                            kind="style",
                            signal=f"Passive-voice drift between strata {a_name} and {b_name} for {source_key}",
                            key="passive_ratio",
                            involved_passage_ids=sorted(set(a["ids"] + b["ids"])),
                            involved_sources=[source_key],
                            confidence=0.5,
                        )
                    )

    return divergences
