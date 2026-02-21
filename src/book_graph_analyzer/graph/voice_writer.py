"""Neo4j VoiceProfile writer and speaker-identification helper.

Follows the same patterns as passage_writer.py:
- get_driver() from .connection
- Session context managers
- MERGE queries for idempotency
"""

from __future__ import annotations

import json
import math
import re
import statistics
from typing import Optional

from .connection import get_driver
from ..voice.profile import (
    CharacterVoiceProfile,
    MODERN_ANACHRONISMS,
    _is_imperative,
    _is_rhetorical,
    _compute_formality_score,
)


class VoiceProfileWriter:
    """Write and query VoiceProfile data in Neo4j."""

    def __init__(self, driver=None):
        self._driver = driver

    @property
    def driver(self):
        if self._driver is None:
            self._driver = get_driver()
            if self._driver is None:
                raise ConnectionError("Cannot connect to Neo4j")
        return self._driver

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def upsert_voice_profile(self, profile: CharacterVoiceProfile) -> None:
        """Create or update a VoiceProfile node in Neo4j.

        The node is keyed on character_id (falls back to character_name slug).
        Calling this multiple times is idempotent.
        """
        profile_id = _make_profile_id(profile)

        props: dict = {
            "id": profile_id,
            "character_id": profile.character_id or _slugify(profile.character_name),
            "character_name": profile.character_name,
            # Corpus stats
            "total_utterances": profile.total_lines,
            "total_words": profile.total_words,
            "vocabulary_size": profile.unique_words,
            "avg_utterance_length": profile.avg_utterance_length,
            # Speech pattern ratios
            "question_ratio": profile.question_ratio,
            "exclamation_ratio": profile.exclamation_ratio,
            "statement_ratio": profile.statement_ratio,
            # Formality metrics
            "formality_score": profile.formality_score,
            "archaism_rate": profile.archaism_rate,
            "contraction_usage": profile.contraction_ratio,
            "first_person_ratio": profile.first_person_ratio,
            "imperative_ratio": profile.imperative_ratio,
            "rhetorical_density": profile.rhetorical_density,
            # Audience variants (stored as JSON strings)
            "formality_by_audience": json.dumps(profile.formality_by_audience),
            "length_by_audience": json.dumps(profile.length_by_audience),
            "register_by_audience": json.dumps(profile.register_by_audience),
            # Fingerprint lists (stored as Neo4j lists)
            "distinctive_words": profile.distinctive_words,
            "signature_phrases": profile.signature_phrases,
            "never_says": profile.never_says,
            # Topic distribution (JSON string)
            "topic_distribution": json.dumps(profile.topic_distribution),
        }

        with self.driver.session() as session:
            session.run(
                "MERGE (vp:VoiceProfile {id: $id}) SET vp += $props",
                id=profile_id,
                props=props,
            )

    def link_passage_to_profile(
        self,
        passage_id: str,
        profile_id: str,
        speaker_id: str,
        audience_type: str,
        context_type: str,
    ) -> None:
        """Create (Passage)-[:VOICED_IN {…}]->(VoiceProfile) edge."""
        edge_props = {
            "speaker_id": speaker_id,
            "audience_type": audience_type,
            "context_type": context_type,
        }

        with self.driver.session() as session:
            session.run(
                """
                MATCH (p:Passage {id: $passage_id})
                MATCH (vp:VoiceProfile {id: $profile_id})
                MERGE (p)-[r:VOICED_IN {speaker_id: $speaker_id}]->(vp)
                SET r += $props
                """,
                passage_id=passage_id,
                profile_id=profile_id,
                speaker_id=speaker_id,
                props=edge_props,
            )

    def get_voice_profile(self, character_id: str) -> dict | None:
        """Fetch a VoiceProfile node from Neo4j by character_id.

        Returns the node properties as a dict, or None if not found.
        """
        with self.driver.session() as session:
            result = session.run(
                "MATCH (vp:VoiceProfile {character_id: $cid}) RETURN vp",
                cid=character_id,
            )
            row = result.single()
            if not row:
                return None
            return dict(row["vp"])

    def identify_speaker(self, text: str, top_n: int = 3) -> list[tuple[str, float]]:
        """Given raw text, return top N (character_id, confidence) matches.

        Extracts surface metrics from *text* and compares them against all
        stored VoiceProfile nodes.  Returns a sorted list (highest confidence
        first), with confidence values in [0.0, 1.0].
        """
        # -- Extract metrics from the query text --
        query_metrics = _extract_text_metrics(text)

        # -- Fetch all profiles --
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (vp:VoiceProfile)
                RETURN vp.character_id AS cid,
                       vp.formality_score AS formality_score,
                       vp.archaism_rate AS archaism_rate,
                       vp.question_ratio AS question_ratio,
                       vp.exclamation_ratio AS exclamation_ratio,
                       vp.imperative_ratio AS imperative_ratio,
                       vp.rhetorical_density AS rhetorical_density,
                       vp.contraction_usage AS contraction_usage,
                       vp.avg_utterance_length AS avg_utterance_length
                """
            )
            profiles = [dict(row) for row in result]

        if not profiles:
            return []

        # -- Score each profile --
        scored: list[tuple[str, float]] = []
        for prof in profiles:
            cid = prof.get("cid", "unknown")
            confidence = _compute_similarity(query_metrics, prof)
            scored.append((cid, round(confidence, 4)))

        # Sort descending by confidence
        scored.sort(key=lambda x: -x[1])
        return scored[:top_n]


# ---------------------------------------------------------------------------
# Pure-python text metric extraction (no Neo4j, used by identify_speaker)
# ---------------------------------------------------------------------------

def _extract_text_metrics(text: str) -> dict:
    """Extract the same metrics we store on VoiceProfile, from raw text."""
    archaisms_set = {
        "thee", "thou", "thy", "thine", "ye", "hath", "doth", "art", "wast",
        "wherefore", "hither", "thither", "whither", "hence", "thence",
        "ere", "nay", "aye", "yea", "behold", "lo", "alas", "forsooth",
        "methinks", "mayhap", "perchance", "betwixt", "amongst", "whilst",
        "verily", "hark", "hearken", "tarry", "prithee",
    }
    contraction_patterns = ["n't", "'s", "'re", "'ve", "'ll", "'d", "'m"]
    fp_words = {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'll", "i'd"}

    # Split into sentences/utterances (simple split on . ! ?)
    utterances = [u.strip() for u in re.split(r'[.!?]+', text) if u.strip()]
    if not utterances:
        utterances = [text]

    total_utter = len(utterances)
    questions = sum(1 for u in utterances if text.find(u) >= 0 and u.endswith("?"))
    exclamations = sum(1 for u in utterances if u.endswith("!"))
    # Recount from original text
    questions = text.count("?")
    exclamations = text.count("!")
    question_ratio = questions / total_utter if total_utter else 0.0
    exclamation_ratio = exclamations / total_utter if total_utter else 0.0

    words = text.lower().split()
    total_words = len(words) or 1

    arch_count = sum(1 for w in words if w.strip('.,!?"\'') in archaisms_set)
    contr_count = sum(1 for w in words if any(p in w for p in contraction_patterns))
    fp_count = sum(1 for w in words if w.strip('.,!?"\'') in fp_words)
    word_lengths = [len(w.strip('.,!?"\'')) for w in words if w.strip('.,!?"\'')]
    avg_wl = statistics.mean(word_lengths) if word_lengths else 4.0

    archaism_rate = arch_count / total_words
    contraction_usage = contr_count / total_words
    first_person_ratio = fp_count / total_words

    # Imperative / rhetorical
    imperative_count = sum(1 for u in re.split(r'[.!?]+', text) if _is_imperative(u.strip()))
    rhetorical_count = sum(1 for u in re.split(r'[.!?]+', text) if _is_rhetorical(u.strip() + "?"))
    imperative_ratio = imperative_count / total_utter if total_utter else 0.0
    rhetorical_density = rhetorical_count / total_utter if total_utter else 0.0

    formality_score = _compute_formality_score(
        archaism_rate, contraction_usage, avg_wl, first_person_ratio
    )

    avg_utterance_length = total_words / total_utter if total_utter else total_words

    return {
        "formality_score": formality_score,
        "archaism_rate": archaism_rate,
        "question_ratio": question_ratio,
        "exclamation_ratio": exclamation_ratio,
        "imperative_ratio": imperative_ratio,
        "rhetorical_density": rhetorical_density,
        "contraction_usage": contraction_usage,
        "avg_utterance_length": avg_utterance_length,
    }


def _compute_similarity(query: dict, profile: dict) -> float:
    """Cosine-distance-inspired similarity in [0, 1] between query and profile metrics."""
    # Metric weights (must sum to 1.0)
    weights = {
        "formality_score": 0.30,
        "archaism_rate": 0.20,
        "question_ratio": 0.10,
        "exclamation_ratio": 0.05,
        "imperative_ratio": 0.15,
        "rhetorical_density": 0.10,
        "contraction_usage": 0.05,
        "avg_utterance_length": 0.05,
    }

    total_weight = 0.0
    weighted_sim = 0.0

    for metric, weight in weights.items():
        q_val = float(query.get(metric) or 0.0)
        p_val = float(profile.get(metric) or 0.0)

        if metric == "avg_utterance_length":
            # Normalise: typical range 0–30 words
            q_val = min(1.0, q_val / 30.0)
            p_val = min(1.0, p_val / 30.0)

        # Similarity = 1 - absolute difference (both values in [0,1])
        diff = abs(q_val - p_val)
        sim = max(0.0, 1.0 - diff)
        weighted_sim += weight * sim
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_sim / total_weight


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_")


def _make_profile_id(profile: CharacterVoiceProfile) -> str:
    if profile.character_id:
        return f"voice_{profile.character_id}"
    return f"voice_{_slugify(profile.character_name)}"
