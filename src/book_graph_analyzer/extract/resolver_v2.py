"""
Entity Resolution v2

Combines the four improvements from Issue #12:
1. TextNormalizer    — fix cp1252/encoding artifacts before NER
2. PronounResolver   — lightweight spaCy coreference within passages
3. DisambiguationDict — context-aware alias → canonical ID mapping
4. Embedding-based clustering — semantic similarity for alias merging
5. Confidence-gated output — threshold logic with needs_review flagging

Drop-in improvement over EntityBootstrapper + EntityResolver.

Usage:
    resolver = EntityResolverV2()
    result = resolver.resolve_text(raw_text, era="Third Age", book="The Two Towers")

    # Access results
    for entity in result.accepted:
        print(entity.canonical_name, entity.confidence)
    for entity in result.flagged:
        print(entity.canonical_name, "NEEDS REVIEW")

Confidence thresholds (from issue spec):
  >= 0.85 → write directly (accepted)
  0.60 <= conf < 0.85 → write with needs_review=True (flagged)
  < 0.60  → do not write, add to review queue (rejected)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

from .bootstrap import EntityBootstrapper, EntityCluster, BootstrapResult
from .normalizer import TextNormalizer
from .coref import PronounResolver, detect_explicit_aliases
from .disambiguation import DisambiguationDict

logger = logging.getLogger(__name__)

# Confidence thresholds (issue #12 spec)
ACCEPT_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.60


@dataclass
class ResolvedEntityV2:
    """
    Final resolved entity with all v2 metadata.
    Extends EntityCluster with disambiguation + coref info.
    """
    canonical_name: str
    canonical_id: Optional[str]          # From disambiguation dict (if known)
    entity_type: str                     # character / place / object / concept / unknown
    variants: list[str]                  # All surface forms seen
    frequency: int
    confidence: float
    needs_review: bool
    source: str                          # 'inferred' | 'disambiguation' | 'embedding' | 'seed'

    # Audit trail
    coref_resolved: list[str] = field(default_factory=list)  # Pronouns resolved here
    explicit_aliases: list[tuple[str, str]] = field(default_factory=list)
    sample_context: str = ""
    inferred_attributes: dict = field(default_factory=dict)

    @property
    def is_accepted(self) -> bool:
        return self.confidence >= ACCEPT_THRESHOLD and not self.needs_review

    @property
    def is_flagged(self) -> bool:
        return REVIEW_THRESHOLD <= self.confidence < ACCEPT_THRESHOLD or self.needs_review

    @property
    def is_rejected(self) -> bool:
        return self.confidence < REVIEW_THRESHOLD and not self.needs_review

    def to_dict(self) -> dict:
        return {
            "canonical_name": self.canonical_name,
            "canonical_id": self.canonical_id,
            "entity_type": self.entity_type,
            "variants": self.variants,
            "frequency": self.frequency,
            "confidence": round(self.confidence, 3),
            "needs_review": self.needs_review,
            "source": self.source,
            "coref_resolved": self.coref_resolved,
            "sample_context": self.sample_context[:200],
            "inferred_attributes": self.inferred_attributes,
        }


@dataclass
class ResolutionResultV2:
    """Output of EntityResolverV2.resolve_text()"""
    accepted: list[ResolvedEntityV2] = field(default_factory=list)   # conf >= 0.85
    flagged: list[ResolvedEntityV2] = field(default_factory=list)    # 0.60 <= conf < 0.85
    rejected: list[ResolvedEntityV2] = field(default_factory=list)   # conf < 0.60
    stats: dict = field(default_factory=dict)

    def all_entities(self) -> list[ResolvedEntityV2]:
        return self.accepted + self.flagged

    def to_dict_list(self, include_rejected: bool = False) -> list[dict]:
        entities = self.all_entities()
        if include_rejected:
            entities += self.rejected
        return [e.to_dict() for e in entities]

    def get_by_id(self, canonical_id: str) -> Optional[ResolvedEntityV2]:
        for e in self.all_entities():
            if e.canonical_id == canonical_id:
                return e
        return None

    def get_by_name(self, name: str) -> Optional[ResolvedEntityV2]:
        name_lower = name.lower()
        for e in self.all_entities():
            if e.canonical_name.lower() == name_lower:
                return e
            if name_lower in [v.lower() for v in e.variants]:
                return e
        return None


class EntityResolverV2:
    """
    Entity Resolution v2 pipeline.

    Improvements over v1:
    - Encoding normalization (no more â€œ artifacts)
    - Pronoun coreference within + across passages
    - Disambiguation dict: surface form → canonical ID with era overrides
    - Embedding-based alias clustering (optional, uses issue #11 VectorStore)
    - Formal confidence gating with three tiers

    Args:
        use_llm: Whether to use LLM for cluster canonicalization
        use_embeddings: Whether to use embedding similarity for clustering
        disambiguation_path: Optional path to a custom disambiguation JSON file
        accept_threshold: Confidence threshold to accept without review (default: 0.85)
        review_threshold: Confidence threshold for flagging (default: 0.60)
    """

    def __init__(
        self,
        use_llm: bool = False,
        use_embeddings: bool = False,
        disambiguation_path: Optional[Path] = None,
        accept_threshold: float = ACCEPT_THRESHOLD,
        review_threshold: float = REVIEW_THRESHOLD,
    ) -> None:
        self.accept_threshold = accept_threshold
        self.review_threshold = review_threshold

        # Sub-components
        self.normalizer = TextNormalizer()
        self.coref = PronounResolver(window_size=3)
        self.disambiguation = DisambiguationDict(load_builtins=True)
        if disambiguation_path:
            self.disambiguation.load(disambiguation_path)

        # Bootstrap pipeline (string-similarity clustering + optional LLM)
        self.bootstrapper = EntityBootstrapper(use_llm=use_llm)
        # Adjust bootstrapper thresholds to match our v2 thresholds
        self.bootstrapper.ACCEPT_THRESHOLD = accept_threshold
        self.bootstrapper.REVIEW_THRESHOLD = review_threshold

        # Optional embedding-based clustering
        self.use_embeddings = use_embeddings
        self._embedder = None  # Lazy-loaded

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def resolve_text(
        self,
        text: str,
        era: Optional[str] = None,
        book: Optional[str] = None,
        passages: Optional[list[str]] = None,
    ) -> ResolutionResultV2:
        """
        Full resolution pipeline on raw text.

        Steps:
        1. Normalize encoding
        2. Detect explicit aliases (textual statements)
        3. Run bootstrapper (candidate extraction + string-similarity clustering)
        4. Apply disambiguation dict to assign canonical IDs
        5. Optionally apply embedding-based alias clustering
        6. Apply confidence gating

        Args:
            text: Raw text to process
            era: Story era for context-dependent disambiguation (e.g., "Third Age")
            book: Book title for context-dependent disambiguation
            passages: Pre-split passages (for coreference; if None, splits by sentence)
        """
        # --- Step 1: Normalize encoding ---
        normalized = self.normalizer.normalize(text)

        # --- Step 2: Detect explicit aliases from text ---
        explicit_aliases = detect_explicit_aliases(normalized)
        logger.info("Explicit aliases found: %d", len(explicit_aliases))

        # --- Step 3: Bootstrap (candidate extraction + string clustering) ---
        bootstrap_result: BootstrapResult = self.bootstrapper.bootstrap(
            normalized, verbose=False
        )
        all_clusters = bootstrap_result.entities + bootstrap_result.flagged

        # --- Step 3b: Supplementary disambiguation pass ---
        # For names that appear only once (below bootstrapper MIN_FREQUENCY),
        # check the disambiguation dict directly. Known aliases always get resolved.
        all_clusters = self._supplement_with_disambiguation(
            all_clusters, normalized, era, book
        )

        # --- Step 4: Apply embedding clustering (optional) ---
        if self.use_embeddings and all_clusters:
            all_clusters = self._apply_embedding_clustering(all_clusters)

        # --- Step 5: Apply disambiguation + confidence gating ---
        accepted: list[ResolvedEntityV2] = []
        flagged: list[ResolvedEntityV2] = []
        rejected: list[ResolvedEntityV2] = []

        # Merge explicit alias pairs into clusters
        alias_map = self._build_alias_map(explicit_aliases)

        for cluster in all_clusters:
            entity = self._resolve_cluster(cluster, era, book, alias_map)

            if entity.confidence >= self.accept_threshold and not entity.needs_review:
                accepted.append(entity)
            elif entity.confidence >= self.review_threshold or entity.needs_review:
                entity.needs_review = True
                flagged.append(entity)
            else:
                rejected.append(entity)

        # Sort by frequency desc
        accepted.sort(key=lambda e: -e.frequency)
        flagged.sort(key=lambda e: -e.frequency)

        stats = {
            "input_chars": len(text),
            "normalized_chars": len(normalized),
            "explicit_aliases": len(explicit_aliases),
            "candidates": bootstrap_result.stats.get("candidates", 0),
            "clusters": bootstrap_result.stats.get("clusters", 0),
            "accepted": len(accepted),
            "flagged": len(flagged),
            "rejected": len(rejected),
            "disambiguation_hits": sum(1 for e in accepted + flagged if e.canonical_id),
        }

        return ResolutionResultV2(
            accepted=accepted,
            flagged=flagged,
            rejected=rejected,
            stats=stats,
        )

    def resolve_passages(
        self,
        passages: list[str],
        era: Optional[str] = None,
        book: Optional[str] = None,
    ) -> ResolutionResultV2:
        """
        Resolve entities across multiple passages with coreference.

        The passages are joined for bootstrapping, but coreference runs
        over the passage sequence for pronoun resolution.
        """
        # Run coreference over passage sequence
        coref_chains = self.coref.get_pronoun_chain(passages)
        logger.info("Coreference chains found: %d", len(coref_chains))

        # Join passages for entity extraction
        combined = "\n\n".join(passages)

        result = self.resolve_text(combined, era=era, book=book, passages=passages)

        # Annotate accepted/flagged entities with coreference info
        for entity in result.accepted + result.flagged:
            name_lower = entity.canonical_name.lower()
            if entity.canonical_name in coref_chains:
                chain = coref_chains[entity.canonical_name]
                entity.coref_resolved = [m.text for m in chain.mentions[:5]]

        return result

    # ------------------------------------------------------------------
    # Supplementary disambiguation pass
    # ------------------------------------------------------------------

    def _supplement_with_disambiguation(
        self,
        clusters: list[EntityCluster],
        text: str,
        era: Optional[str],
        book: Optional[str],
    ) -> list[EntityCluster]:
        """
        Scan text for known disambiguation entries that weren't picked up
        by the bootstrapper (because they appeared only once).

        Creates minimal EntityCluster entries for each match not already
        covered by existing clusters.
        """
        import re

        # Build set of canonical names already found
        already_found: set[str] = set()
        for c in clusters:
            already_found.add(c.canonical_name.lower() if c.canonical_name else "")
            for v in c.variants:
                already_found.add(v.lower())

        new_clusters: list[EntityCluster] = []
        text_lower = text.lower()

        # Check each disambiguation entry against the text
        for surface, entry in self.disambiguation._entries.items():
            if surface in text_lower and surface not in already_found:
                canonical_id, conf = self.disambiguation.resolve(
                    surface, era=era, book=book
                )
                if canonical_id and conf >= 0.5:
                    # Find the canonical name for display
                    # Use a proper-cased version of the surface form from text
                    import re
                    pattern = re.escape(surface)
                    match = re.search(pattern, text, re.IGNORECASE)
                    display_name = match.group(0) if match else surface.title()

                    cluster = EntityCluster(
                        canonical_name=display_name,
                        entity_type=_infer_type_from_id(canonical_id),
                        variants=[display_name],
                        frequency=1,
                        cluster_confidence=conf,
                        needs_review=conf < self.accept_threshold,
                        contexts=[],
                    )
                    new_clusters.append(cluster)
                    already_found.add(surface)

        return clusters + new_clusters

    # ------------------------------------------------------------------
    # Disambiguation resolution
    # ------------------------------------------------------------------

    def _resolve_cluster(
        self,
        cluster: EntityCluster,
        era: Optional[str],
        book: Optional[str],
        alias_map: dict[str, str],
    ) -> ResolvedEntityV2:
        """Convert a bootstrap EntityCluster to a ResolvedEntityV2."""
        canonical_name = cluster.canonical_name or max(cluster.variants, key=len)

        # Check disambiguation dict for all variants
        canonical_id = None
        best_conf = cluster.cluster_confidence
        source = "inferred"

        for variant in [canonical_name] + cluster.variants:
            if self.disambiguation.has_entry(variant):
                cid, conf = self.disambiguation.resolve(variant, era=era, book=book)
                if cid and conf > 0.5:
                    canonical_id = cid
                    # Boost confidence when we have a disambiguation entry
                    best_conf = max(best_conf, conf * 0.9)
                    source = "disambiguation"
                    break

        # Check alias_map (from explicit alias detection)
        for variant in [canonical_name] + cluster.variants:
            if variant in alias_map:
                alias_target = alias_map[variant]
                # This entity is known to be an alias of another — merge
                if not canonical_id:
                    canonical_id = alias_target.lower().replace(" ", "_")
                    source = "explicit_alias"
                    best_conf = max(best_conf, 0.88)

        return ResolvedEntityV2(
            canonical_name=canonical_name,
            canonical_id=canonical_id,
            entity_type=cluster.entity_type,
            variants=list(cluster.variants),
            frequency=cluster.frequency,
            confidence=min(best_conf, 1.0),
            needs_review=cluster.needs_review,
            source=source,
            inferred_attributes=cluster.inferred_attributes,
            sample_context=cluster.contexts[0] if cluster.contexts else "",
        )

    def _build_alias_map(
        self, explicit_aliases: list[tuple[str, str]]
    ) -> dict[str, str]:
        """
        Build a bidirectional alias map from explicit alias pairs.

        The "longer" / "more specific" form is usually canonical.
        """
        alias_map: dict[str, str] = {}
        for a, b in explicit_aliases:
            # Longer name is usually canonical
            canonical, alias = (a, b) if len(a) >= len(b) else (b, a)
            alias_map[alias] = canonical
            alias_map[a] = canonical
            alias_map[b] = canonical
        return alias_map

    # ------------------------------------------------------------------
    # Embedding-based clustering
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_embedder(self):
        """Lazy-load the sentence-transformers embedder."""
        if self._embedder is None:
            try:
                from ..embed import Embedder
                self._embedder = Embedder()
            except ImportError:
                logger.warning("sentence-transformers not available; skipping embedding clustering")
                self._embedder = False
        return self._embedder if self._embedder is not False else None

    def _apply_embedding_clustering(
        self,
        clusters: list[EntityCluster],
        similarity_threshold: float = 0.88,
    ) -> list[EntityCluster]:
        """
        Apply embedding-based alias merging on top of string-similarity clusters.

        Clusters with high embedding similarity are merged.
        This catches cases like "Mithrandir" / "Gandalf" which have low
        string similarity but high semantic similarity (both wizard names).

        Returns updated cluster list with additional merges applied.
        """
        embedder = self._get_embedder()
        if not embedder or len(clusters) < 2:
            return clusters

        import numpy as np

        # Build text representations for each cluster
        texts = []
        for c in clusters:
            ctx = c.contexts[0][:100] if c.contexts else ""
            texts.append(f"{c.canonical_name or c.variants[0]}: {ctx}")

        try:
            embeddings = embedder.embed(texts)
        except Exception as exc:
            logger.warning("Embedding clustering failed: %s", exc)
            return clusters

        emb_array = np.array(embeddings)

        # Normalize
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        emb_norm = emb_array / norms

        # Compute cosine similarity matrix
        sim_matrix = emb_norm @ emb_norm.T

        # Find pairs above threshold (same entity type, different string clusters)
        merged: set[int] = set()
        merge_map: dict[int, int] = {}  # index → merge_target_index

        for i in range(len(clusters)):
            if i in merged:
                continue
            for j in range(i + 1, len(clusters)):
                if j in merged:
                    continue
                if sim_matrix[i, j] >= similarity_threshold:
                    # Check entity type compatibility
                    ci, cj = clusters[i], clusters[j]
                    if (ci.entity_type == cj.entity_type or
                            "unknown" in (ci.entity_type, cj.entity_type)):
                        # Merge j into i
                        logger.info(
                            "Embedding merge: '%s' + '%s' (sim=%.3f)",
                            ci.canonical_name or ci.variants[0],
                            cj.canonical_name or cj.variants[0],
                            sim_matrix[i, j],
                        )
                        merged.add(j)
                        merge_map[j] = i

        # Apply merges
        result_clusters = []
        for i, cluster in enumerate(clusters):
            if i in merged:
                # Merge this cluster into its target
                target = clusters[merge_map[i]]
                target.variants.extend(cluster.variants)
                target.contexts.extend(cluster.contexts[:2])
                target.frequency += cluster.frequency
                if not target.canonical_name:
                    target.canonical_name = cluster.canonical_name
                # Boost confidence slightly when embedding confirms the merge
                target.cluster_confidence = min(
                    target.cluster_confidence * 1.05, 1.0
                )
            else:
                result_clusters.append(cluster)

        logger.info("Embedding clustering: %d merges, %d → %d clusters",
                    len(merge_map), len(clusters), len(result_clusters))
        return result_clusters

    # ------------------------------------------------------------------
    # Utility: check resolution rate
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def resolution_rate(
        self, result: ResolutionResultV2, min_frequency: int = 5
    ) -> float:
        """
        Compute resolution rate for frequent entities.

        Issue spec: >= 90% for entities appearing 5+ times.
        """
        frequent = [e for e in result.all_entities() if e.frequency >= min_frequency]
        if not frequent:
            return 1.0
        resolved = [e for e in frequent if e.canonical_id is not None]
        return len(resolved) / len(frequent)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _infer_type_from_id(canonical_id: str) -> str:
    """Infer entity type from canonical ID prefix."""
    if canonical_id.startswith("char_"):
        return "character"
    if canonical_id.startswith("place_"):
        return "place"
    if canonical_id.startswith("obj_"):
        return "object"
    return "unknown"
