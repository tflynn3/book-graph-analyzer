"""Generic entity bootstrapper — infer canonical entities from raw text without seed files.

Multi-pass pipeline:
    1. Extract candidates  — all capitalized name-like tokens with frequency + context
    2. Cluster aliases     — string similarity + transitivity grouping
    3. Canonicalize        — LLM confirms cluster, elects canonical name, infers type
    4. Confidence gate     — auto-accept / flag-for-review / skip

Works on ANY corpus. Seed files accelerate but are not required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

from ..llm import get_llm_client

# ---------------------------------------------------------------------------
# Stop-words: common capitalized words that are NOT proper nouns
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "The", "And", "But", "For", "Not", "He", "She", "It", "They",
    "Was", "Had", "Did", "His", "Her", "Its", "Our", "You", "Who",
    "Then", "When", "Where", "What", "How", "Yes", "No", "All",
    "One", "Two", "Three", "Now", "So", "Well", "Still", "Yet",
    "There", "Here", "That", "This", "Those", "These", "Very",
    "Great", "Old", "New", "Long", "Dark", "High", "First", "Last",
    "Men", "Man", "Men", "Good", "Evil", "Ring", "Fire", "Water",
}

# Epithet / title prefixes that signal a proper noun follows
_EPITHET_PREFIX = re.compile(
    r"\b(?:the\s+)?(?:King|Queen|Lord|Lady|Prince|Princess|Captain|"
    r"Wizard|Grey|White|Black|Dark|High|Great|Elder|Master|"
    r"Sir|Dame|General|Steward)\s+(?:of\s+)?([A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+)?)",
    re.IGNORECASE,
)

# Core proper-noun pattern: 1-4 consecutive Capitalised words
_PROPER_NOUN = re.compile(r"\b([A-Z][a-zA-Z'\-]{2,}(?:\s+[A-Z][a-zA-Z'\-]{2,}){0,3})\b")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EntityCandidate:
    """A raw name-like token found in text, before alias resolution."""
    text: str
    frequency: int = 0
    contexts: list[str] = field(default_factory=list)
    source: str = "pattern"   # 'ner' | 'pattern' | 'caps'


@dataclass
class EntityCluster:
    """A group of name variants believed to refer to the same entity."""
    variants: list[str]
    canonical_name: str = ""
    entity_type: str = "unknown"      # 'character' | 'place' | 'object' | 'concept'
    frequency: int = 0
    contexts: list[str] = field(default_factory=list)
    cluster_confidence: float = 0.0
    inferred_attributes: dict = field(default_factory=dict)
    needs_review: bool = False
    source: str = "inferred"          # 'inferred' | 'seed' | 'human_verified'


@dataclass
class BootstrapResult:
    """Output of the full bootstrap pipeline."""
    entities: list[EntityCluster]         # confidence >= ACCEPT_THRESHOLD
    flagged: list[EntityCluster]          # REVIEW_THRESHOLD <= confidence < ACCEPT_THRESHOLD
    stats: dict = field(default_factory=dict)

    def all_entities(self) -> list[EntityCluster]:
        return self.entities + self.flagged

    def to_dict_list(self) -> list[dict]:
        """Serialise accepted + flagged entities to a list of dicts."""
        out = []
        for e in self.all_entities():
            out.append({
                "canonical_name": e.canonical_name,
                "variants": e.variants,
                "entity_type": e.entity_type,
                "frequency": e.frequency,
                "cluster_confidence": round(e.cluster_confidence, 3),
                "needs_review": e.needs_review,
                "inferred_attributes": e.inferred_attributes,
                "source": e.source,
                "sample_context": e.contexts[0] if e.contexts else "",
            })
        return out


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EntityBootstrapper:
    """Bootstrap canonical entities from raw text without pre-existing seed files."""

    # Tuning knobs
    STRING_SIM_THRESHOLD: int = 80          # rapidfuzz ratio 0-100
    MIN_FREQUENCY: int = 2                  # ignore hapax legomena
    ACCEPT_THRESHOLD: float = 0.80          # auto-accept above this confidence
    REVIEW_THRESHOLD: float = 0.55          # flag-for-review between this and ACCEPT
    CONTEXT_WINDOW: int = 150               # chars either side of a mention
    MAX_CONTEXTS: int = 5                   # max context windows stored per entity

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._llm = get_llm_client() if use_llm else None

    # ------------------------------------------------------------------
    # Pass 1 — candidate extraction
    # ------------------------------------------------------------------

    def extract_candidates(self, text: str) -> list[EntityCandidate]:
        """Extract all capitalised name-like tokens with frequency and context windows."""
        candidates: dict[str, EntityCandidate] = {}

        for match in _PROPER_NOUN.finditer(text):
            name = match.group(1).strip()

            # Filter stop-words and too-short tokens
            if name in _STOPWORDS or len(name) < 3:
                continue
            # Filter if it looks like a sentence-start capitalisation (preceded by . or ?)
            preceding = text[max(0, match.start() - 2): match.start()].strip()
            if preceding in (".", "?", "!", ":") and " " not in name:
                continue

            start = max(0, match.start() - self.CONTEXT_WINDOW)
            end = min(len(text), match.end() + self.CONTEXT_WINDOW)
            context = text[start:end].replace("\n", " ").strip()

            if name not in candidates:
                candidates[name] = EntityCandidate(text=name, frequency=0, source="pattern")
            candidates[name].frequency += 1
            if len(candidates[name].contexts) < self.MAX_CONTEXTS:
                candidates[name].contexts.append(context)

        return [c for c in candidates.values() if c.frequency >= self.MIN_FREQUENCY]

    # ------------------------------------------------------------------
    # Pass 2 — alias clustering via transitivity
    # ------------------------------------------------------------------

    def cluster_aliases(self, candidates: list[EntityCandidate]) -> list[EntityCluster]:
        """Group candidates that are likely aliases using string similarity + transitivity.

        Algorithm (based on spaCy community research):
        1. Build similarity pairs: (A, B) where string ratio > threshold
           OR A is a substring of B (or vice versa)
        2. Apply transitivity: if A~B and B~C then {A, B, C} form one cluster
        3. Merge candidate metadata within each cluster
        """
        cand_map = {c.text: c for c in candidates}
        names = list(cand_map.keys())

        # Build similarity pairs
        pairs: set[tuple[str, str]] = set()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                a_lo, b_lo = a.lower(), b.lower()
                is_sub = a_lo in b_lo or b_lo in a_lo
                str_sim = fuzz.ratio(a, b)
                tok_sim = fuzz.token_set_ratio(a, b)

                if is_sub or str_sim >= self.STRING_SIM_THRESHOLD or tok_sim >= self.STRING_SIM_THRESHOLD:
                    pairs.add((a, b))

        # Transitivity expansion
        clusters: list[EntityCluster] = []
        remaining = set(names)

        while remaining:
            seed = next(iter(remaining))
            members: set[str] = {seed}
            remaining.discard(seed)

            changed = True
            while changed:
                changed = False
                for a, b in list(pairs):
                    if a in members and b in remaining:
                        members.add(b)
                        remaining.discard(b)
                        changed = True
                    elif b in members and a in remaining:
                        members.add(a)
                        remaining.discard(a)
                        changed = True

            member_list = list(members)
            total_freq = sum(cand_map[m].frequency for m in member_list if m in cand_map)
            all_ctx: list[str] = []
            for m in member_list:
                if m in cand_map:
                    all_ctx.extend(cand_map[m].contexts)

            # Single-member clusters have lower confidence (no alias evidence)
            conf = 0.85 if len(member_list) > 1 else 0.70

            clusters.append(EntityCluster(
                variants=member_list,
                frequency=total_freq,
                contexts=all_ctx[: self.MAX_CONTEXTS],
                cluster_confidence=conf,
                needs_review=(conf < self.ACCEPT_THRESHOLD),
            ))

        return sorted(clusters, key=lambda c: c.frequency, reverse=True)

    # ------------------------------------------------------------------
    # Pass 3 — LLM canonicalisation
    # ------------------------------------------------------------------

    def canonicalize_clusters(self, clusters: list[EntityCluster]) -> list[EntityCluster]:
        """Use LLM to confirm each cluster, elect a canonical name, and infer entity type."""
        if not self.use_llm or not self._llm:
            for c in clusters:
                c.canonical_name = max(c.variants, key=len)
                c.entity_type = "unknown"
            return clusters

        results: list[EntityCluster] = []
        for cluster in clusters:
            if cluster.frequency < self.MIN_FREQUENCY:
                continue

            variants_str = ", ".join(f'"{v}"' for v in cluster.variants[:10])
            ctx_str = "\n".join(
                f"  - ...{ctx[:200]}..." for ctx in cluster.contexts[:3]
            )

            prompt = (
                "Analyze these name variants found in a literary text and answer in JSON only.\n\n"
                f"Variants: {variants_str}\n"
                f"Total mentions: {cluster.frequency}\n\n"
                f"Context examples:\n{ctx_str}\n\n"
                "JSON response:\n"
                '{\n'
                '  "same_entity": true or false,\n'
                '  "canonical_name": "best name for this entity",\n'
                '  "entity_type": "character" or "place" or "object" or "concept" or "unknown",\n'
                '  "confidence": 0.0 to 1.0,\n'
                '  "inferred_attributes": {"notes": "any inferred race, role, era, etc."}\n'
                '}'
            )

            try:
                response = self._llm.generate(prompt, temperature=0.1, max_tokens=300)
                data = self._llm.extract_json(response)

                if data and isinstance(data, dict):
                    if not data.get("same_entity", True):
                        # LLM says cluster should be split — flag for human review
                        cluster.needs_review = True
                        cluster.canonical_name = max(cluster.variants, key=len)
                        cluster.cluster_confidence = 0.4
                    else:
                        cluster.canonical_name = data.get("canonical_name") or max(cluster.variants, key=len)
                        cluster.entity_type = data.get("entity_type", "unknown")
                        cluster.cluster_confidence = float(data.get("confidence", 0.7))
                        cluster.inferred_attributes = data.get("inferred_attributes", {})
                        cluster.needs_review = cluster.cluster_confidence < self.ACCEPT_THRESHOLD
                else:
                    cluster.canonical_name = max(cluster.variants, key=len)
                    cluster.needs_review = True

            except Exception:  # noqa: BLE001
                cluster.canonical_name = max(cluster.variants, key=len)
                cluster.needs_review = True

            results.append(cluster)

        return results

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def bootstrap(self, text: str, verbose: bool = True) -> BootstrapResult:
        """Run the full pipeline on raw text.

        Returns a BootstrapResult with accepted entities (high confidence)
        and flagged entities (medium confidence, need human review).
        """
        if verbose:
            print(f"Bootstrapping entities from {len(text):,} chars of text...")

        candidates = self.extract_candidates(text)
        if verbose:
            print(f"  Pass 1 — candidates: {len(candidates)}")

        clusters = self.cluster_aliases(candidates)
        if verbose:
            print(f"  Pass 2 — clusters:   {len(clusters)}")

        clusters = self.canonicalize_clusters(clusters)
        if verbose:
            print(f"  Pass 3 — canonicalized: {len(clusters)}")

        accepted = [c for c in clusters if c.cluster_confidence >= self.ACCEPT_THRESHOLD and not c.needs_review]
        flagged = [c for c in clusters if self.REVIEW_THRESHOLD <= c.cluster_confidence < self.ACCEPT_THRESHOLD or c.needs_review]
        skipped = [c for c in clusters if c.cluster_confidence < self.REVIEW_THRESHOLD and not c.needs_review]

        if verbose:
            print(f"  Result  — accepted: {len(accepted)}, flagged: {len(flagged)}, skipped: {len(skipped)}")

        return BootstrapResult(
            entities=accepted,
            flagged=flagged,
            stats={
                "candidates": len(candidates),
                "clusters": len(clusters),
                "accepted": len(accepted),
                "flagged": len(flagged),
                "skipped": len(skipped),
            },
        )
