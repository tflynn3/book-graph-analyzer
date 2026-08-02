"""Register taxonomy — classifier, canonical templates, and Neo4j writer.

Contains:
  CANONICAL_SCENE_TEMPLATES  — pre-built SceneTemplate nodes for all 7 registers
  RegisterClassifier         — heuristic passage classification (no LLM needed)
  SceneTemplateNeo4jWriter   — write SceneTemplate nodes and EXEMPLIFIES edges
  build_generation_prompt    — assemble a style-grounded generation prompt
"""

from __future__ import annotations

import re
import statistics
from typing import Optional

from ..models.scene_template import (
    SceneTemplate,
    RegisterClassification,
    ExemplifiesEdge,
    ProseRegister,
    REGISTER_DESCRIPTIONS,
    REGISTER_TRIGGERS,
    REGISTER_STRUCTURAL_PATTERNS,
    REGISTER_SIGNATURE_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Canonical SceneTemplate nodes for all 7 registers
# (Style metrics derived from corpus analysis of Tolkien's prose)
# ---------------------------------------------------------------------------

CANONICAL_SCENE_TEMPLATES: dict[str, SceneTemplate] = {

    ProseRegister.ELEGIAC: SceneTemplate(
        id=f"template_{ProseRegister.ELEGIAC}",
        register=ProseRegister.ELEGIAC,
        scene_type="farewell",
        avg_sentence_length=26.5,
        sentence_length_variance=9.0,
        passive_ratio=0.35,
        dialogue_density=0.06,
        archaic_word_rate=0.12,
        lexical_diversity=0.78,
        descriptive_focus=["light", "silver", "gold", "ancient_age", "sound", "fading"],
        common_openings=[
            "Long [had/was/were] ...",
            "Once ..., but now ...",
            "In days of old ...",
            "There was a time when ...",
        ],
        common_closings=[
            "... and so it was that ... passed from the world.",
            "... but that too has now passed.",
            "... and none now remain who remember.",
        ],
        structural_pattern=REGISTER_STRUCTURAL_PATTERNS[ProseRegister.ELEGIAC],
        description=REGISTER_DESCRIPTIONS[ProseRegister.ELEGIAC],
        trigger_conditions=REGISTER_TRIGGERS[ProseRegister.ELEGIAC],
        example_passages=[
            (
                "I passed over Caradhras and Zirakzigil and Bundushathûr. "
                "Long I fell, and he fell with me. His fire was quenched, but mine burns yet. "
                "I was sent back — for a brief time, until my task is done."
            ),
            (
                "Galadriel looked upon them with clear eye. 'I know what it is that you saw, "
                "'for that is also in my mind. Do not be afraid! I will not turn to shadow. "
                "'I pass the test. I will diminish, and go into the West, and remain Galadriel.'"
            ),
        ],
    ),

    ProseRegister.EUCATASTROPHIC: SceneTemplate(
        id=f"template_{ProseRegister.EUCATASTROPHIC}",
        register=ProseRegister.EUCATASTROPHIC,
        scene_type="battle",
        avg_sentence_length=11.0,
        sentence_length_variance=8.5,
        passive_ratio=0.10,
        dialogue_density=0.15,
        archaic_word_rate=0.05,
        lexical_diversity=0.65,
        descriptive_focus=["light", "sound", "speed", "movement"],
        common_openings=[
            "But then ...",
            "And in that moment ...",
            "Suddenly ...",
            "Yet even as ...",
        ],
        common_closings=[
            "And the shadow passed.",
            "And hope was renewed.",
            "And the darkness broke.",
        ],
        structural_pattern=REGISTER_STRUCTURAL_PATTERNS[ProseRegister.EUCATASTROPHIC],
        description=REGISTER_DESCRIPTIONS[ProseRegister.EUCATASTROPHIC],
        trigger_conditions=REGISTER_TRIGGERS[ProseRegister.EUCATASTROPHIC],
        example_passages=[
            (
                "And as if in answer there came from far away another note. "
                "Horns, horns, horns. In dark Mindolluin's sides they dimly echoed. "
                "Great horns of the North wildly blowing. Riders of Rohan!"
            ),
        ],
    ),

    ProseRegister.COZY: SceneTemplate(
        id=f"template_{ProseRegister.COZY}",
        register=ProseRegister.COZY,
        scene_type="domestic",
        avg_sentence_length=13.5,
        sentence_length_variance=5.0,
        passive_ratio=0.09,
        dialogue_density=0.42,
        archaic_word_rate=0.02,
        lexical_diversity=0.60,
        descriptive_focus=["food", "warmth", "texture", "humor", "domestic"],
        common_openings=[
            "It was a bright cold day ...",
            "The [room/kitchen/hole] was ...",
            "'Have some more!' said ...",
            "After a good meal ...",
        ],
        common_closings=[
            "... and they were very well content.",
            "... and the evening passed pleasantly.",
            "... and soon all was quiet.",
        ],
        structural_pattern=REGISTER_STRUCTURAL_PATTERNS[ProseRegister.COZY],
        description=REGISTER_DESCRIPTIONS[ProseRegister.COZY],
        trigger_conditions=REGISTER_TRIGGERS[ProseRegister.COZY],
        example_passages=[
            (
                "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, "
                "filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole "
                "with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort."
            ),
        ],
    ),

    ProseRegister.DREAD: SceneTemplate(
        id=f"template_{ProseRegister.DREAD}",
        register=ProseRegister.DREAD,
        scene_type="encounter",
        avg_sentence_length=9.5,
        sentence_length_variance=4.0,
        passive_ratio=0.14,
        dialogue_density=0.04,
        archaic_word_rate=0.07,
        lexical_diversity=0.58,
        descriptive_focus=["darkness", "cold", "silence", "weight", "shadow"],
        common_openings=[
            "The darkness was ...",
            "There was no sound ...",
            "Something moved ...",
            "A cold wind ...",
        ],
        common_closings=[
            "And then there was silence.",
            "Nothing moved. Nothing spoke.",
            "And they did not breathe.",
        ],
        structural_pattern=REGISTER_STRUCTURAL_PATTERNS[ProseRegister.DREAD],
        description=REGISTER_DESCRIPTIONS[ProseRegister.DREAD],
        trigger_conditions=REGISTER_TRIGGERS[ProseRegister.DREAD],
        example_passages=[
            (
                "Something was climbing the cliff. Then Frodo saw it. A figure, "
                "black against the starlit sky. It was creeping upward, silent, "
                "hand over hand."
            ),
        ],
    ),

    ProseRegister.WONDER: SceneTemplate(
        id=f"template_{ProseRegister.WONDER}",
        register=ProseRegister.WONDER,
        scene_type="arrival",
        avg_sentence_length=19.5,
        sentence_length_variance=7.5,
        passive_ratio=0.20,
        dialogue_density=0.10,
        archaic_word_rate=0.08,
        lexical_diversity=0.76,
        descriptive_focus=["light", "sound", "silver", "gold", "music", "stars"],
        common_openings=[
            "Never before had he ...",
            "He could not tell ...",
            "It seemed to [him/her] ...",
            "Then [he/she] saw ...",
        ],
        common_closings=[
            "... and for a long time [he/she] could not speak.",
            "... and [he/she] stood still, forgetting everything.",
            "... and wondered if [he/she] was dreaming.",
        ],
        structural_pattern=REGISTER_STRUCTURAL_PATTERNS[ProseRegister.WONDER],
        description=REGISTER_DESCRIPTIONS[ProseRegister.WONDER],
        trigger_conditions=REGISTER_TRIGGERS[ProseRegister.WONDER],
        example_passages=[
            (
                "As Frodo looked out he saw that the sky had grown clear again and "
                "the wind had brought a cold sharp night. There was a star shining "
                "on the edge of the world. He could hear the sound of running water."
            ),
        ],
    ),

    ProseRegister.LORE_REVEAL: SceneTemplate(
        id=f"template_{ProseRegister.LORE_REVEAL}",
        register=ProseRegister.LORE_REVEAL,
        scene_type="council",
        avg_sentence_length=31.0,
        sentence_length_variance=12.0,
        passive_ratio=0.40,
        dialogue_density=0.20,
        archaic_word_rate=0.15,
        lexical_diversity=0.82,
        descriptive_focus=["ancient_age", "genealogy", "history", "power", "lineage"],
        common_openings=[
            "In the Second Age of the World ...",
            "It is said that ...",
            "Long ago, in the days when ...",
            "Of the [making/history/origin] of ...",
        ],
        common_closings=[
            "... and so it has been until this day.",
            "... and this is the history of ...",
            "... of which little else is recorded.",
        ],
        structural_pattern=REGISTER_STRUCTURAL_PATTERNS[ProseRegister.LORE_REVEAL],
        description=REGISTER_DESCRIPTIONS[ProseRegister.LORE_REVEAL],
        trigger_conditions=REGISTER_TRIGGERS[ProseRegister.LORE_REVEAL],
        example_passages=[
            (
                "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords "
                "in their halls of stone, Nine for Mortal Men doomed to die, One for the "
                "Dark Lord on his dark throne..."
            ),
        ],
    ),

    ProseRegister.FELLOWSHIP: SceneTemplate(
        id=f"template_{ProseRegister.FELLOWSHIP}",
        register=ProseRegister.FELLOWSHIP,
        scene_type="journey",
        avg_sentence_length=11.5,
        sentence_length_variance=4.5,
        passive_ratio=0.08,
        dialogue_density=0.55,
        archaic_word_rate=0.03,
        lexical_diversity=0.62,
        descriptive_focus=["physical", "humor", "warmth", "action", "companionship"],
        common_openings=[
            "'Come on!' said ...",
            "They walked ...",
            "After a while ...",
            "Sam [pulled/looked/said] ...",
        ],
        common_closings=[
            "... and they laughed.",
            "... and went on.",
            "... and felt better for it.",
        ],
        structural_pattern=REGISTER_STRUCTURAL_PATTERNS[ProseRegister.FELLOWSHIP],
        description=REGISTER_DESCRIPTIONS[ProseRegister.FELLOWSHIP],
        trigger_conditions=REGISTER_TRIGGERS[ProseRegister.FELLOWSHIP],
        example_passages=[
            (
                "'I don't know half of you half as well as I should like,' said Bilbo, "
                "'and I like less than half of you half as well as you deserve.' "
                "There was some scattered clapping."
            ),
        ],
    ),
}


# ---------------------------------------------------------------------------
# RegisterClassifier — heuristic, no LLM required
# ---------------------------------------------------------------------------

class RegisterClassifier:
    """Classifies passages by prose register using heuristic text analysis.

    No LLM required — uses keyword scoring, structural analysis, and style metrics.
    For LLM-enhanced classification, subclass and override classify().
    """

    def __init__(
        self,
        templates: Optional[dict[str, SceneTemplate]] = None,
    ) -> None:
        self._templates = templates or CANONICAL_SCENE_TEMPLATES

    @property
    def templates(self) -> dict[str, SceneTemplate]:
        return self._templates

    def classify(
        self,
        text: str,
        passage_id: str = "inline",
        threshold: float = 0.2,
    ) -> RegisterClassification:
        """Classify a passage by prose register.

        Args:
            text: The passage text to classify.
            passage_id: Identifier for display.
            threshold: Minimum confidence to include a register.

        Returns:
            RegisterClassification with all registers above threshold.
        """
        scores = self._score_all_registers(text)
        # Filter to those above threshold and sort descending
        classifications = [
            (reg, conf)
            for reg, conf in sorted(scores.items(), key=lambda x: -x[1])
            if conf >= threshold
        ]
        return RegisterClassification(
            passage_id=passage_id,
            passage_text_snippet=text[:120] + ("..." if len(text) > 120 else ""),
            classifications=classifications,
        )

    def classify_from_passage(self, passage) -> RegisterClassification:
        """Classify from a Passage model object."""
        return self.classify(
            text=passage.text,
            passage_id=passage.id,
        )

    def _score_all_registers(self, text: str) -> dict[str, float]:
        """Score text against all registers. Returns dict of {register: score 0-1}."""
        text_lower = text.lower()
        word_count = max(1, len(text.split()))

        raw_scores: dict[str, float] = {}
        for register in ProseRegister:
            raw_scores[register] = self._score_register(
                text, text_lower, word_count, register
            )

        # Also apply structural/metric scoring
        metrics = self._extract_metrics(text)
        structural_boosts = self._structural_boost(metrics)
        for reg, boost in structural_boosts.items():
            raw_scores[reg] = raw_scores.get(reg, 0.0) + boost

        # Normalize to [0, 1]
        max_score = max(raw_scores.values()) if raw_scores else 1.0
        if max_score > 0:
            return {
                reg: min(1.0, score / max_score)
                for reg, score in raw_scores.items()
            }
        return {reg: 0.0 for reg in ProseRegister}

    def _score_register(
        self,
        text: str,
        text_lower: str,
        word_count: int,
        register: str,
    ) -> float:
        """Score text against a single register using keyword matching."""
        keywords = REGISTER_SIGNATURE_KEYWORDS.get(register, [])
        if not keywords:
            return 0.0

        matches = sum(
            1 for kw in keywords
            if kw.lower() in text_lower
        )
        # Normalize: max out at 3 keyword matches
        keyword_score = min(1.0, matches / 3.0)

        # Trigger phrases boost score
        trigger_phrases = REGISTER_TRIGGERS.get(register, [])
        trigger_hits = sum(
            1 for phrase in trigger_phrases
            if phrase.lower() in text_lower
        )
        trigger_score = min(0.4, trigger_hits * 0.2)

        return keyword_score + trigger_score

    def _extract_metrics(self, text: str) -> dict:
        """Extract style metrics from text."""
        sentences = re.split(r'[.!?]+\s+', text.strip())
        sentences = [s for s in sentences if s.strip()]

        if not sentences:
            return {
                "avg_sentence_length": 15.0,
                "dialogue_density": 0.0,
                "passive_ratio": 0.0,
            }

        # Average sentence length in words
        lengths = [len(s.split()) for s in sentences]
        avg_length = statistics.mean(lengths) if lengths else 15.0

        # Dialogue density
        in_quotes = 0
        in_q = False
        for ch in text:
            if ch in ('"', '\u201c'):
                in_q = True
            elif ch in ('"', '\u201d'):
                in_q = False
            elif in_q:
                in_quotes += 1
        dialogue_density = in_quotes / max(1, len(text))

        # Passive ratio (heuristic)
        passive_pat = re.compile(r'\b(was|were|been|is|are|be)\s+\w+ed\b', re.I)
        passive_hits = len(passive_pat.findall(text))
        passive_ratio = passive_hits / max(1, len(sentences))

        return {
            "avg_sentence_length": avg_length,
            "dialogue_density": dialogue_density,
            "passive_ratio": passive_ratio,
        }

    def _structural_boost(self, metrics: dict) -> dict[str, float]:
        """Boost register scores based on structural metrics."""
        boosts: dict[str, float] = {}
        avg = metrics.get("avg_sentence_length", 15.0)
        dialogue = metrics.get("dialogue_density", 0.0)
        passive = metrics.get("passive_ratio", 0.0)

        # Long sentences → elegiac / lore_reveal
        if avg > 22:
            boosts[ProseRegister.ELEGIAC] = boosts.get(ProseRegister.ELEGIAC, 0) + 0.3
            boosts[ProseRegister.LORE_REVEAL] = boosts.get(ProseRegister.LORE_REVEAL, 0) + 0.3

        # Short sentences → dread / eucatastrophic
        if avg < 11:
            boosts[ProseRegister.DREAD] = boosts.get(ProseRegister.DREAD, 0) + 0.2
            boosts[ProseRegister.EUCATASTROPHIC] = boosts.get(ProseRegister.EUCATASTROPHIC, 0) + 0.2

        # High dialogue → fellowship / cozy
        if dialogue > 0.35:
            boosts[ProseRegister.FELLOWSHIP] = boosts.get(ProseRegister.FELLOWSHIP, 0) + 0.35
            boosts[ProseRegister.COZY] = boosts.get(ProseRegister.COZY, 0) + 0.2

        # High passive ratio → elegiac / lore_reveal
        if passive > 0.25:
            boosts[ProseRegister.ELEGIAC] = boosts.get(ProseRegister.ELEGIAC, 0) + 0.25
            boosts[ProseRegister.LORE_REVEAL] = boosts.get(ProseRegister.LORE_REVEAL, 0) + 0.25

        return boosts

    def get_template(self, register: str) -> Optional[SceneTemplate]:
        """Return the canonical SceneTemplate for a register."""
        return self._templates.get(register)

    def describe_register(self, register: str) -> str:
        """Return a full description of a register."""
        tmpl = self.get_template(register)
        if not tmpl:
            return f"No template found for register '{register}'"
        lines = [
            f"Register: {register}",
            f"Description: {tmpl.description}",
            f"Structural pattern: {tmpl.structural_pattern}",
            "Style metrics:",
            f"  Avg sentence length: {tmpl.avg_sentence_length:.1f} words",
            f"  Passive ratio: {tmpl.passive_ratio:.0%}",
            f"  Dialogue density: {tmpl.dialogue_density:.0%}",
            f"  Archaic word rate: {tmpl.archaic_word_rate:.0%}",
            f"Descriptive focus: {', '.join(tmpl.descriptive_focus)}",
        ]
        if tmpl.trigger_conditions:
            lines.append(f"Triggered by: {', '.join(tmpl.trigger_conditions[:5])}")
        return "\n".join(lines)

    def classify_batch(
        self,
        texts: list[tuple[str, str]],  # (passage_id, text) pairs
        threshold: float = 0.2,
    ) -> list[RegisterClassification]:
        """Classify a batch of passages."""
        return [
            self.classify(text, pid, threshold)
            for pid, text in texts
        ]


# ---------------------------------------------------------------------------
# Generation prompt assembly
# ---------------------------------------------------------------------------

def build_generation_prompt(
    register: str,
    anchor_passages: list[str],
    scene_context: str = "",
) -> str:
    """Build a style-grounded generation prompt for a given register.

    Args:
        register: The target ProseRegister.
        anchor_passages: Actual Tolkien passages exemplifying this register.
        scene_context: Brief description of what the scene should be about.

    Returns:
        A prompt fragment ready for injection into an LLM generation call.
    """
    tmpl = CANONICAL_SCENE_TEMPLATES.get(register)
    if not tmpl:
        return f"Write a scene in the '{register}' register."

    lines = [
        "=== STYLE INSTRUCTION ===",
        "",
        tmpl.generation_prompt_fragment(),
        "",
    ]

    if anchor_passages:
        lines += [
            f"=== ANCHOR PASSAGES ({len(anchor_passages)} exemplars from source corpus) ===",
            "",
            "These are actual passages from the source text exemplifying this register.",
            "Match their structural and stylistic patterns, not their content.",
            "",
        ]
        for i, passage in enumerate(anchor_passages[:5], 1):
            lines.append(f"[{i}] {passage.strip()[:300]}")
            lines.append("")

    if scene_context:
        lines += [
            "=== SCENE TO GENERATE ===",
            "",
            scene_context,
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SceneTemplateNeo4jWriter
# ---------------------------------------------------------------------------

class SceneTemplateNeo4jWriter:
    """Write SceneTemplate nodes and EXEMPLIFIES edges to Neo4j."""

    def __init__(self, driver=None) -> None:
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

    def upsert_template(self, template: SceneTemplate) -> None:
        """Create or update a SceneTemplate node."""
        with self.driver.session() as session:
            session.run(
                "MERGE (t:SceneTemplate {id: $id}) SET t += $props",
                id=template.id,
                props=template.to_neo4j_props(),
            )

    def upsert_all_canonical(self) -> int:
        """Write all 7 canonical templates. Returns count written."""
        count = 0
        for template in CANONICAL_SCENE_TEMPLATES.values():
            self.upsert_template(template)
            count += 1
        return count

    def upsert_exemplifies_edge(self, edge: ExemplifiesEdge) -> None:
        """Create or update an EXEMPLIFIES edge from Passage to SceneTemplate."""
        template_id = f"template_{edge.template_id}"
        with self.driver.session() as session:
            session.run(
                """
                MATCH (p:Passage {id: $pid})
                MERGE (t:SceneTemplate {id: $tid})
                MERGE (p)-[r:EXEMPLIFIES {template_id: $tid}]->(t)
                SET r += $props
                """,
                pid=edge.passage_id,
                tid=template_id,
                props=edge.to_neo4j_props(),
            )

    def query_anchor_passages(
        self,
        register: str,
        min_confidence: float = 0.7,
        limit: int = 5,
    ) -> list[dict]:
        """Query exemplar passages for a register.

        These are used as style anchors in generation prompts.
        """
        template_id = f"template_{register}"
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Passage)-[r:EXEMPLIFIES]->(t:SceneTemplate {id: $tid})
                WHERE r.confidence >= $min_conf
                RETURN p.id AS id, p.text AS text, r.confidence AS confidence
                ORDER BY rand()
                LIMIT $limit
                """,
                tid=template_id,
                min_conf=min_confidence,
                limit=limit,
            )
            return [dict(row) for row in result]

    def query_classified_passages(
        self, register: str, limit: int = 20
    ) -> list[dict]:
        """Query all passages classified as a given register."""
        template_id = f"template_{register}"
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Passage)-[r:EXEMPLIFIES]->(t:SceneTemplate {id: $tid})
                RETURN p.id AS id, p.text AS text, r.confidence AS confidence
                ORDER BY r.confidence DESC
                LIMIT $limit
                """,
                tid=template_id,
                limit=limit,
            )
            return [dict(row) for row in result]
