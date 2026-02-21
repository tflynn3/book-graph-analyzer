"""Dynamic style injection for scene generation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Optional


@dataclass
class StyleConstraints:
    """Aggregated style constraints for a scene type."""

    scene_type: str
    sample_size: int
    avg_sentence_length_words: float
    dialogue_ratio: float
    passive_ratio: float
    archaic_word_density: float
    characteristic_vocab: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StyleInjector:
    """Build prompt constraints from passage-level style data in Neo4j."""

    SCENE_TYPES = [
        "battle",
        "council",
        "journey",
        "myth_cosmogony",
        "personal_dialogue",
        "grief_lament",
        "discovery",
    ]

    STATIC_FALLBACK_BLOCK = """STYLE GUIDE (fallback):
- Flowing, rhythmic prose with Anglo-Saxon cadence
- Rich nature imagery and attention to landscape
- Formal dialogue appropriate to each character's race and status
- Mythic, omniscient narrative voice
- Show don't tell - let actions and dialogue reveal character"""

    _SCENE_HINTS: dict[str, tuple[str, ...]] = {
        "battle": (
            "battle", "war", "siege", "charge", "assault", "duel", "orc", "host", "ambush", "attack",
        ),
        "council": (
            "council", "debate", "speak", "speaks", "deliberation", "parley", "oath", "decree", "valar",
        ),
        "journey": (
            "journey", "road", "path", "travel", "ride", "crossing", "wilderness", "march",
        ),
        "myth_cosmogony": (
            "creation", "before time", "music", "ainur", "cosmogony", "world began", "eä", "valar made",
        ),
        "personal_dialogue": (
            "confession", "private", "quiet talk", "spoke softly", "between them", "heart", "friend",
        ),
        "grief_lament": (
            "lament", "grief", "mourn", "wept", "funeral", "loss", "sorrow",
        ),
        "discovery": (
            "discover", "found", "reveal", "hidden", "strange", "unknown", "uncovered",
        ),
    }

    _SCENE_QUERY_TAGS: dict[str, list[str]] = {
        "battle": ["battle", "action", "combat"],
        "council": ["council", "dialogue", "deliberation", "exposition"],
        "journey": ["journey", "travel", "description"],
        "myth_cosmogony": ["myth_cosmogony", "exposition", "description"],
        "personal_dialogue": ["personal_dialogue", "dialogue", "reflection"],
        "grief_lament": ["grief_lament", "reflection", "description"],
        "discovery": ["discovery", "description", "action"],
    }

    def __init__(self, driver=None, min_samples: int = 5):
        self.driver = driver
        self.min_samples = min_samples

    def classify_scene_type(self, chapter_beat: str) -> str:
        text = (chapter_beat or "").lower()
        tokens = set(re.findall(r"[a-z']+", text))
        best_type = "journey"
        best_hits = 0
        for scene_type, keywords in self._SCENE_HINTS.items():
            hits = 0
            for k in keywords:
                if " " in k:
                    if k in text:
                        hits += 1
                elif k in tokens:
                    hits += 1
            if hits > best_hits:
                best_hits = hits
                best_type = scene_type
        return best_type

    def get_style_constraints(self, scene_type: str) -> Optional[StyleConstraints]:
        if not self.driver:
            return None

        query_tags = self._SCENE_QUERY_TAGS.get(scene_type, [scene_type])
        with self.driver.session() as session:
            rows = session.run(
                """
                MATCH (p:Passage)
                WHERE p.scene_type IS NOT NULL
                  AND toLower(p.scene_type) IN [tag IN $scene_tags | toLower(tag)]
                  AND p.sentence_count IS NOT NULL
                  AND p.avg_sentence_length IS NOT NULL
                RETURN
                    p.avg_sentence_length AS avg_sentence_length,
                    coalesce(p.dialogue_density, 0.0) AS dialogue_density,
                    coalesce(p.passive_ratio, 0.0) AS passive_ratio,
                    coalesce(p.archaic_word_count, 0) AS archaic_word_count,
                    coalesce(p.sentence_count, 0) AS sentence_count,
                    p.text AS text
                """,
                scene_tags=query_tags,
            )
            data = [dict(r) for r in rows]

        if len(data) < self.min_samples:
            return None

        avg_sentence_length = mean(float(r["avg_sentence_length"]) for r in data)
        dialogue_ratio = mean(float(r["dialogue_density"]) for r in data)
        passive_ratio = mean(float(r["passive_ratio"]) for r in data)

        total_words = 0.0
        total_archaic = 0.0
        vocab_counts: dict[str, int] = {}
        for row in data:
            words = str(row.get("text") or "").split()
            word_count = len(words)
            total_words += word_count
            total_archaic += float(row.get("archaic_word_count") or 0)
            for w in words:
                lw = w.strip(".,;:!?\"'()[]{}").lower()
                if len(lw) >= 6:
                    vocab_counts[lw] = vocab_counts.get(lw, 0) + 1

        archaic_density = (total_archaic / total_words) if total_words > 0 else 0.0
        characteristic_vocab = [
            token for token, _ in sorted(vocab_counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
        ]

        return StyleConstraints(
            scene_type=scene_type,
            sample_size=len(data),
            avg_sentence_length_words=round(avg_sentence_length, 2),
            dialogue_ratio=round(dialogue_ratio, 3),
            passive_ratio=round(passive_ratio, 3),
            archaic_word_density=round(archaic_density, 3),
            characteristic_vocab=characteristic_vocab,
        )

    def build_style_block(self, constraints: Optional[StyleConstraints]) -> str:
        if not constraints:
            return self.STATIC_FALLBACK_BLOCK

        low = max(6, int(round(constraints.avg_sentence_length_words - 3)))
        high = max(low + 2, int(round(constraints.avg_sentence_length_words + 3)))
        vocab = ", ".join(constraints.characteristic_vocab[:6]) or "(none)"

        return (
            f"STYLE CONSTRAINTS ({constraints.scene_type} scene — derived from corpus passages):\n"
            f"- Sample size: {constraints.sample_size} passages\n"
            f"- Target sentence length: {low}-{high} words on average\n"
            f"- Dialogue ratio target: ~{constraints.dialogue_ratio:.2f}\n"
            f"- Passive voice ratio: ~{constraints.passive_ratio:.2f}\n"
            f"- Archaic word density: ~{constraints.archaic_word_density:.2f}\n"
            f"- Vocabulary markers to prefer: {vocab}"
        )
