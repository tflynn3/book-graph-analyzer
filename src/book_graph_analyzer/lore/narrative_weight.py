"""NarrativeWeightComputer — computes the NarrativeWeight metric for Passages.

Computation strategy:
  - Rule-based components: computed from passage text, era-reference counts,
    entity counts, and style metrics (no LLM required).
  - LLM-based components: optional, defaults to 0.0 when LLM unavailable.

All scores are normalised to [0.0, 1.0] before being passed to NarrativeWeight.
"""

from __future__ import annotations

import re
from typing import Optional


from ..models.narrative_weight import (
    NarrativeWeight,
    ThemeNode,
    TOLKIEN_THEMES,
    THEME_BY_ID,
    COMPONENT_WEIGHTS,
)
from ..models.passage import Passage


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


def _norm(value: float, max_val: float) -> float:
    """Normalise value to [0, 1] given max_val. Returns 0 if max_val is 0."""
    if max_val <= 0:
        return 0.0
    return _clamp(value / max_val)


# ---------------------------------------------------------------------------
# Text analysis helpers
# ---------------------------------------------------------------------------

# Common dialogue markers — used to detect is_dialogue in raw text
_DIALOGUE_PATTERN = re.compile(r'["\u201c\u201d\u2018\u2019]')
_QUOTE_OPEN = re.compile(r'["\u201c]')

# Proper noun heuristic: capitalized word not at sentence start, not a common word
_COMMON_WORDS = frozenset({
    "The", "A", "An", "And", "But", "Or", "Nor", "For", "Yet", "So",
    "He", "She", "It", "They", "We", "I", "You", "His", "Her", "Its",
    "Their", "Our", "My", "Your", "This", "That", "These", "Those",
    "At", "By", "In", "Of", "On", "To", "Up", "As", "If", "Is", "Are",
    "Was", "Were", "Be", "Been", "Have", "Has", "Had", "Do", "Did",
    "Not", "No", "All", "One", "Two", "Three", "Then", "Than",
    "When", "Where", "Who", "What", "How", "Why", "Which",
})

_SENTENCE_SPLIT = re.compile(r'[.!?]+\s+')

def _count_proper_nouns(text: str) -> int:
    """Heuristic count of proper nouns in text (capitalized words, not common)."""
    words = text.split()
    count = 0
    sentence_starts: set[int] = set()
    # Mark approximate sentence-start positions
    for match in _SENTENCE_SPLIT.finditer(text):
        # Next word after punctuation is at a sentence start
        sentence_starts.add(match.end())
    for i, word in enumerate(words):
        # Skip sentence-start words (they're capitalised for grammar, not proper nouns)
        clean = re.sub(r"[^a-zA-Z']", "", word)
        if (
            clean
            and clean[0].isupper()
            and clean not in _COMMON_WORDS
            and len(clean) > 2
        ):
            count += 1
    return count


def _count_sentences(text: str) -> int:
    """Rough sentence count."""
    parts = _SENTENCE_SPLIT.split(text.strip())
    return max(1, len([p for p in parts if p.strip()]))


def _count_words(text: str) -> int:
    return len(text.split())


def _dialogue_density(text: str) -> float:
    """Approximate fraction of text in dialogue (inside quotes)."""
    # Count characters inside quotation marks
    inside = 0
    total = len(text)
    if total == 0:
        return 0.0
    in_quote = False
    for ch in text:
        if ch in ('"', '\u201c', '\u2018'):
            in_quote = True
        elif ch in ('"', '\u201d', '\u2019'):
            in_quote = False
        elif in_quote:
            inside += 1
    return _clamp(inside / total)


def _passive_ratio(text: str) -> float:
    """Heuristic passive voice ratio using 'was/were/been + past participle' pattern."""
    passive_pat = re.compile(
        r'\b(was|were|been|is|are|be)\s+\w+ed\b', re.IGNORECASE
    )
    sentences = _count_sentences(text)
    matches = len(passive_pat.findall(text))
    return _clamp(matches / max(1, sentences))


def _detect_themes(text: str) -> list[str]:
    """Return list of theme IDs whose keywords appear in the text."""
    text_lower = text.lower()
    detected = []
    for theme in TOLKIEN_THEMES:
        for kw in theme.detection_keywords:
            if kw.lower() in text_lower:
                detected.append(theme.id)
                break
    return detected


def _theme_coherence(theme_ids: list[str]) -> float:
    """Estimate theme coherence: themes that are thematically related score higher.

    Uses a simple adjacency table — themes that commonly co-occur in Tolkien are
    considered coherent.
    """
    if len(theme_ids) <= 1:
        return 1.0 if theme_ids else 0.0

    # Coherence pairs — these themes naturally reinforce each other
    COHERENT_PAIRS: set[frozenset] = {
        frozenset({"eucatastrophe", "hope_vs_despair"}),
        frozenset({"eucatastrophe", "mercy"}),
        frozenset({"the_long_defeat", "hope_vs_despair"}),
        frozenset({"the_long_defeat", "diminishment"}),
        frozenset({"the_past_pressing_on_present", "diminishment"}),
        frozenset({"the_past_pressing_on_present", "mortality"}),
        frozenset({"power_corrupts", "stewardship"}),
        frozenset({"loyalty", "mercy"}),
        frozenset({"loyalty", "stewardship"}),
        frozenset({"mortality", "hope_vs_despair"}),
        frozenset({"mortality", "the_long_defeat"}),
    }

    ids = set(theme_ids)
    pairs_possible = len(ids) * (len(ids) - 1) / 2
    if pairs_possible == 0:
        return 0.0

    coherent_count = 0
    for a in ids:
        for b in ids:
            if a < b and frozenset({a, b}) in COHERENT_PAIRS:
                coherent_count += 1

    return _clamp(coherent_count / pairs_possible)


def _emotional_keywords(text: str) -> int:
    """Count distinct emotional register markers."""
    emotional_words = {
        # sorrow / grief
        "grief", "wept", "tears", "sorrow", "mourned", "lament",
        # joy / wonder
        "joy", "wonder", "marvelled", "beautiful", "glory", "radiant",
        # fear / dread
        "fear", "dread", "terror", "shadow", "darkness", "horror",
        # courage / resolve
        "courage", "resolve", "stood fast", "endure", "defiance",
        # awe / reverence
        "awe", "reverence", "ancient", "sacred", "holy", "mighty",
    }
    text_lower = text.lower()
    return sum(1 for w in emotional_words if w in text_lower)


def _emotional_contrast(text: str) -> float:
    """Detect light/dark contrast in the same passage."""
    light_words = {"light", "hope", "joy", "beauty", "radiant", "dawn", "bright", "golden"}
    dark_words = {"shadow", "dark", "doom", "despair", "dread", "night", "gloom", "fear"}
    text_lower = text.lower()
    has_light = any(w in text_lower for w in light_words)
    has_dark = any(w in text_lower for w in dark_words)
    return 1.0 if (has_light and has_dark) else 0.0


def _foreshadowing_score(text: str) -> float:
    """Detect foreshadowing language heuristically."""
    foreshadowing_pats = [
        r'\bone day\b', r'\bperhaps\b', r'\bin time\b', r'\bwhen the time comes\b',
        r'\byou will (need|see|understand)\b', r'\bremember (this|that)\b',
        r'\bthe time (will come|is coming)\b', r'\bit may be\b', r'\bif ever\b',
    ]
    count = sum(
        1 for pat in foreshadowing_pats
        if re.search(pat, text, re.IGNORECASE)
    )
    return _clamp(count / 3.0)  # 3+ foreshadowing hints = score 1.0


def _revelation_score(text: str) -> float:
    """Detect revelation language heuristically."""
    revelation_pats = [
        r'\bI (never|did not) knew?\b', r'\bnow I understand\b', r'\bthe truth\b',
        r'\bhe (was|is) (?!not\b)', r'\bshe (was|is) (?!not\b)',
        r'\bsecret\b', r'\brevealed\b', r'\buntold\b', r'\bhidden\b',
        r'\ball along\b', r'\bnow you know\b',
    ]
    count = sum(
        1 for pat in revelation_pats
        if re.search(pat, text, re.IGNORECASE)
    )
    return _clamp(count / 3.0)


def _callback_density(text: str) -> float:
    """Detect callbacks/references to past events heuristically."""
    callback_pats = [
        r'\byou (may )?remember\b', r'\bas I (said|told)\b', r'\bonce (before|again)\b',
        r'\blong ago\b', r'\bwhen (we|you|he|she|they) (first|last)\b',
        r'\bthe last time\b', r'\bstill (remember|recall)\b',
    ]
    count = sum(
        1 for pat in callback_pats
        if re.search(pat, text, re.IGNORECASE)
    )
    return _clamp(count / 2.0)  # 2+ callbacks = score 1.0


def _voice_distinctiveness(text: str, is_dialogue: bool) -> float:
    """Heuristic distinctiveness: dialogue with archaic language = more distinctive."""
    if not is_dialogue:
        return 0.2  # Narration is less voice-distinctive by default
    # Archaic markers boost score
    archaic = {"thee", "thou", "thy", "dost", "hast", "art", "wilt", "canst", "verily"}
    words = set(text.lower().split())
    archaic_count = len(words & archaic)
    return _clamp(0.3 + archaic_count * 0.15)


# ---------------------------------------------------------------------------
# NarrativeWeightComputer
# ---------------------------------------------------------------------------

class NarrativeWeightComputer:
    """Computes NarrativeWeight for a Passage using rule-based heuristics.

    All computation is pure-Python — no Neo4j or LLM required.
    For LLM-enhanced scoring, subclass and override `_score_llm_components`.
    """

    # Maximum corpus temporal depth (years) — used to normalise temporal_depth score.
    # 20,000 years = Before Time / Ainulindalë, the maximum possible depth.
    MAX_TEMPORAL_DEPTH_YEARS: float = 20_000.0

    # Cap for normalisation of raw counts
    MAX_ENTITY_COUNT: int = 15
    MAX_ERA_REF_COUNT: int = 7
    MAX_EMOTIONAL_KEYWORDS: int = 8

    def compute_from_text(
        self,
        text: str,
        era_ref_count: int = 0,
        temporal_depth_years: Optional[float] = None,
        entity_count: int = 0,
        story_era: Optional[str] = None,
        is_dialogue: bool = False,
    ) -> NarrativeWeight:
        """Compute NarrativeWeight from raw text and optional metadata.

        This is the primary computation path — all rule-based, no external deps.

        Args:
            text: The passage text.
            era_ref_count: Number of distinct era references (from REFERENCES_ERA edges).
            temporal_depth_years: Years before story-time of the oldest era reference.
            entity_count: Number of distinct named entities in the passage.
            story_era: Era in which the scene is set (string, e.g. 'Third Age').
            is_dialogue: Whether the passage is primarily dialogue.

        Returns:
            NarrativeWeight with overall recomputed.
        """
        word_count = _count_words(text)

        # --- Temporal complexity ---
        temporal_depth = _norm(
            temporal_depth_years or 0.0,
            self.MAX_TEMPORAL_DEPTH_YEARS,
        )
        era_reference_count_score = _norm(
            era_ref_count,
            self.MAX_ERA_REF_COUNT,
        )

        # --- Lore density ---
        proper_nouns = _count_proper_nouns(text)
        lore_density = _norm(
            proper_nouns / max(1, word_count) * 100,
            25.0,  # 25 proper nouns per 100 words = score 1.0
        )
        entity_reference_count_score = _norm(entity_count, self.MAX_ENTITY_COUNT)

        # --- Thematic resonance ---
        detected_theme_ids = _detect_themes(text)
        thematic_threads = _norm(len(detected_theme_ids), len(TOLKIEN_THEMES))
        theme_coherence = _theme_coherence(detected_theme_ids)

        # --- Narrative structure ---
        revelation_count = _revelation_score(text)
        callback_density = _callback_density(text)
        foreshadowing_count = _foreshadowing_score(text)
        dramatic_irony = 0.0  # Cannot be computed without reader/character context

        # --- Character depth ---
        character_revelation = revelation_count * 0.7  # proxy
        voice_distinctiveness = _voice_distinctiveness(text, is_dialogue)

        # --- Emotional complexity ---
        emotional_kw_count = _emotional_keywords(text)
        emotional_register_count = _norm(emotional_kw_count, self.MAX_EMOTIONAL_KEYWORDS)
        emotional_contrast = _emotional_contrast(text)

        weight = NarrativeWeight(
            temporal_depth=round(temporal_depth, 4),
            era_reference_count=round(era_reference_count_score, 4),
            lore_density=round(lore_density, 4),
            entity_reference_count=round(entity_reference_count_score, 4),
            thematic_threads=round(thematic_threads, 4),
            theme_coherence=round(theme_coherence, 4),
            revelation_count=round(revelation_count, 4),
            callback_density=round(callback_density, 4),
            foreshadowing_count=round(foreshadowing_count, 4),
            dramatic_irony=round(dramatic_irony, 4),
            character_revelation=round(character_revelation, 4),
            voice_distinctiveness=round(voice_distinctiveness, 4),
            emotional_register_count=round(emotional_register_count, 4),
            emotional_contrast=round(emotional_contrast, 4),
            overall=0.0,
        )
        return weight.compute_overall()

    def compute_from_passage(self, passage: Passage) -> NarrativeWeight:
        """Compute NarrativeWeight from a Passage model object.

        Uses temporal fields, speaker_ids, and is_dialogue from the passage.
        """
        return self.compute_from_text(
            text=passage.text,
            era_ref_count=passage.era_reference_count,
            temporal_depth_years=passage.temporal_depth_years_back,
            entity_count=len(passage.speaker_ids),  # use speaker count as proxy
            story_era=passage.story_era,
            is_dialogue=passage.is_dialogue,
        )

    def detect_themes(self, text: str) -> list[ThemeNode]:
        """Return ThemeNode objects for all themes detected in the text."""
        ids = _detect_themes(text)
        return [THEME_BY_ID[i] for i in ids if i in THEME_BY_ID]

    def improvement_suggestions(self, weight: NarrativeWeight, n: int = 3) -> list[str]:
        """Return n improvement suggestions for the weakest components."""
        return weight.improvement_suggestions(n)

    def compute_corpus_stats(
        self, passages: list[Passage]
    ) -> dict:
        """Compute aggregate NarrativeWeight stats across a corpus of passages.

        Returns a dict with mean/max overall and a breakdown per component.
        """
        if not passages:
            return {"count": 0, "mean_overall": 0.0, "max_overall": 0.0}

        weights = [self.compute_from_passage(p) for p in passages]
        overalls = [w.overall for w in weights]
        mean_overall = sum(overalls) / len(overalls)
        max_overall = max(overalls)

        component_means = {}
        for comp in COMPONENT_WEIGHTS:
            values = [getattr(w, comp) for w in weights]
            component_means[comp] = round(sum(values) / len(values), 4)

        return {
            "count": len(passages),
            "mean_overall": round(mean_overall, 4),
            "max_overall": round(max_overall, 4),
            "component_means": component_means,
        }


# ---------------------------------------------------------------------------
# Neo4j write helpers (optional — only used when Neo4j is available)
# ---------------------------------------------------------------------------

class NarrativeWeightNeo4jWriter:
    """Write NarrativeWeight scores and ThemeNodes to Neo4j."""

    def __init__(self, driver=None):
        self._driver = driver

    @property
    def driver(self):
        if self._driver is None:
            from ..graph.connection import get_driver
            self._driver = get_driver()
        return self._driver

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def ensure_theme_nodes(self) -> None:
        """Create/update all ThemeNodes in the graph. Idempotent."""
        with self.driver.session() as session:
            for theme in TOLKIEN_THEMES:
                session.run(
                    "MERGE (t:Theme {id: $id}) SET t += $props",
                    id=theme.id,
                    props=theme.to_neo4j_props(),
                )

    def write_passage_weight(
        self,
        passage_id: str,
        weight: NarrativeWeight,
        theme_ids: Optional[list[str]] = None,
    ) -> None:
        """Write NarrativeWeight properties to a Passage node and create EXPRESSES edges."""
        props = weight.to_dict()
        with self.driver.session() as session:
            # Update passage node
            session.run(
                "MATCH (p:Passage {id: $id}) SET p += $props",
                id=passage_id,
                props=props,
            )
            # Create EXPRESSES edges to Theme nodes
            if theme_ids:
                for theme_id in theme_ids:
                    if theme_id not in THEME_BY_ID:
                        continue
                    theme = THEME_BY_ID[theme_id]
                    session.run(
                        """
                        MATCH (p:Passage {id: $pid})
                        MERGE (t:Theme {id: $tid})
                        SET t.name = $name, t.description = $desc, t.tolkien_specific = $ts
                        MERGE (p)-[r:EXPRESSES {passage_id: $pid}]->(t)
                        SET r.weight = $weight
                        """,
                        pid=passage_id,
                        tid=theme_id,
                        name=theme.name,
                        desc=theme.description,
                        ts=theme.tolkien_specific,
                        weight=weight.thematic_threads,
                    )

    def query_top_passages(
        self, limit: int = 20, min_overall: float = 0.0
    ) -> list[dict]:
        """Return top passages by NarrativeWeight overall score."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Passage)
                WHERE p.nw_overall IS NOT NULL AND p.nw_overall >= $min
                RETURN p.id AS id, p.text AS text,
                       p.nw_overall AS overall,
                       p.story_era AS story_era,
                       p.temporal_depth_era AS depth_era
                ORDER BY p.nw_overall DESC
                LIMIT $limit
                """,
                min=min_overall,
                limit=limit,
            )
            return [dict(r) for r in result]
