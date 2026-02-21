"""SceneTemplate model — Author Register Taxonomy for prose style generation.

Issue #9: The 7 Tolkien prose registers are structural-emotional modes that
define HOW a passage is written, not just what it's about. Each SceneTemplate
node stores measured style metrics and serves as a generation target.

Unlike TolkienRegister (issue #8, emotional state of characters), ProseRegister
describes the *prose mode* — useful for style injection into generation prompts.

Node: (:SceneTemplate { register, scene_type, avg_sentence_length, ... })
Edge: (Passage)-[:EXEMPLIFIES { confidence: float }]->(SceneTemplate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# The 7 Tolkien prose registers
# ---------------------------------------------------------------------------

class ProseRegister(str, Enum):
    """The 7 Tolkien structural-emotional prose modes."""
    ELEGIAC        = "elegiac"      # Loss, fading, ancient beauty
    EUCATASTROPHIC = "eucatastrophic"  # Sudden joyful turn
    COZY           = "cozy"         # Shire-warmth, domesticity
    DREAD          = "dread"        # Shadow, doom, approaching evil
    WONDER         = "wonder"       # First encounter with the sublime
    LORE_REVEAL    = "lore_reveal"  # Deep history as narrative
    FELLOWSHIP     = "fellowship"   # Companions under pressure, loyalty, humor

    def __str__(self) -> str:
        return self.value

    def __format__(self, format_spec: str) -> str:
        return format(self.value, format_spec)


# Canonical description of each register's prose characteristics
REGISTER_DESCRIPTIONS: dict[str, str] = {
    ProseRegister.ELEGIAC: (
        "Long sentences, passive voice, archaic diction, silver/gold imagery. "
        "Triggered by loss, diminishment, ancient beauty fading. "
        "Example: Galadriel's speech about the decline of the Elves."
    ),
    ProseRegister.EUCATASTROPHIC: (
        "Short explosive sentences after long buildup; light breaking through darkness. "
        "Tolkien's defining narrative move — the sudden joyful turn after despair. "
        "Example: The Eagles arriving at the Pelennor."
    ),
    ProseRegister.COZY: (
        "Short sentences, humor, Anglo-Saxon vocabulary, warmth and food imagery. "
        "Triggered by Shire-warmth, domesticity, safety. "
        "Example: Opening chapters of FotR in the Shire."
    ),
    ProseRegister.DREAD: (
        "Sparse syntax, darkness/cold imagery, weight and silence, short declaratives. "
        "Triggered by shadow approaching, doom, evil. "
        "Example: Entering Moria, the Nazgûl passing."
    ),
    ProseRegister.WONDER: (
        "Simile-heavy, light/music/silver/gold imagery, verbs of seeing and hearing. "
        "Triggered by first encounter with the sublime. "
        "Example: Frodo's first sight of Elves, arriving at Rivendell."
    ),
    ProseRegister.LORE_REVEAL: (
        "Long complex sentences, genealogical depth, omniscient narrator tone, "
        "passive constructions, encyclopedic precision. "
        "Triggered by deep history delivered as narrative. "
        "Example: Council of Elrond backstory, Silmarillion prose."
    ),
    ProseRegister.FELLOWSHIP: (
        "Dialogue-heavy, short paragraphs, physical action, banter and warmth, "
        "companions supporting each other under pressure. "
        "Triggered by companions on a shared mission. "
        "Example: The Company's journey, Sam and Frodo's exchanges."
    ),
}

# Trigger events/conditions for each register
REGISTER_TRIGGERS: dict[str, list[str]] = {
    ProseRegister.ELEGIAC: [
        "departure", "farewell", "loss", "fading", "memory of greatness",
        "diminishment", "end of an age", "last of their kind",
    ],
    ProseRegister.EUCATASTROPHIC: [
        "rescue", "sudden salvation", "eagles", "light breaking",
        "enemy routed", "despair overcome", "unexpected victory",
    ],
    ProseRegister.COZY: [
        "meal", "fireside", "hobbit-hole", "inn", "rest after journey",
        "hospitality", "pipe", "songs and stories",
    ],
    ProseRegister.DREAD: [
        "dark power approaching", "nazgul", "fell creatures",
        "entering enemy territory", "shadow deepening", "waiting for doom",
    ],
    ProseRegister.WONDER: [
        "first sight of elves", "arriving at a great hall", "ancient artifact",
        "star-lit sky", "great music", "deep magic revealed",
    ],
    ProseRegister.LORE_REVEAL: [
        "council", "history explained", "lineage revealed", "ancient war recounted",
        "prophecy explained", "genealogy given",
    ],
    ProseRegister.FELLOWSHIP: [
        "companions traveling", "dialogue between friends", "banter",
        "making camp", "shared danger", "comrades in arms",
    ],
}

# Structural patterns (how each register typically progresses)
REGISTER_STRUCTURAL_PATTERNS: dict[str, str] = {
    ProseRegister.ELEGIAC: (
        "long reflective setup → evocation of lost beauty → "
        "sorrowful acceptance → lingering final image"
    ),
    ProseRegister.EUCATASTROPHIC: (
        "extended darkness and despair → single pivot sentence → "
        "short explosive bursts → wonder and relief washing over"
    ),
    ProseRegister.COZY: (
        "domestic detail (food/fire/comfort) → easy dialogue → "
        "humor or warmth → satisfying closure"
    ),
    ProseRegister.DREAD: (
        "observation → ominous detail → silence/stillness → "
        "short declarative shock → held breath ending"
    ),
    ProseRegister.WONDER: (
        "arrival/first sight → sensory overwhelm (light, sound) → "
        "simile cascade → awe that renders the character speechless"
    ),
    ProseRegister.LORE_REVEAL: (
        "historical preamble → genealogical depth → key event → "
        "consequence for present → connection to now"
    ),
    ProseRegister.FELLOWSHIP: (
        "physical action or movement → dialogue exchange → "
        "character reveals loyalty/humor → bond deepened"
    ),
}

# Tone-signature keywords used for heuristic classification
REGISTER_SIGNATURE_KEYWORDS: dict[str, list[str]] = {
    ProseRegister.ELEGIAC: [
        "fading", "diminished", "long ago", "once was", "shadow of what was",
        "passed away", "no longer", "lament", "memory", "ages past",
        "gold and silver", "ancient", "glory", "waned",
    ],
    ProseRegister.EUCATASTROPHIC: [
        "suddenly", "but lo", "eagles", "dawn", "light broke", "turn",
        "against all hope", "yet", "at the last", "joy",
        "victory", "saved", "eagles are coming",
    ],
    ProseRegister.COZY: [
        "supper", "pipe", "warm", "fire", "comfortable", "cheerful",
        "tea", "food", "laughed", "ale", "cozy", "home",
        "hobbit", "hearth", "round door",
    ],
    ProseRegister.DREAD: [
        "dark", "shadow", "cold", "silence", "weight", "dread",
        "fear", "terrible", "fell", "evil", "black",
        "creeping", "doom", "void",
    ],
    ProseRegister.WONDER: [
        "marvelled", "beautiful", "stars", "silver", "music",
        "radiant", "fair", "wonder", "glittering", "light",
        "astonished", "never had he seen",
    ],
    ProseRegister.LORE_REVEAL: [
        "in the elder days", "ages ago", "of old", "it was said",
        "the history of", "first made", "who was", "born of",
        "whose father was", "realm of", "years before",
    ],
    ProseRegister.FELLOWSHIP: [
        "said sam", "said frodo", "said gandalf", "laughed", "replied",
        "together", "companions", "the company", "beside him",
        "going on", "road", "ahead", "camp",
    ],
}


# ---------------------------------------------------------------------------
# SceneTemplate — the full style node
# ---------------------------------------------------------------------------

@dataclass
class SceneTemplate:
    """A prose style template for a specific register.

    Stored as (:SceneTemplate) in Neo4j.
    Connected to passages via (Passage)-[:EXEMPLIFIES { confidence }]->(SceneTemplate).
    """
    id: str
    register: str                          # ProseRegister value
    scene_type: str = "general"            # 'battle' | 'journey' | 'council' | 'dialogue' | etc.

    # Tolkien's measured style metrics for this register (from corpus analysis)
    avg_sentence_length: float = 15.0      # words per sentence
    sentence_length_variance: float = 5.0  # standard deviation in words
    passive_ratio: float = 0.15            # fraction of sentences with passive voice
    dialogue_density: float = 0.20         # fraction of text in quotation marks
    archaic_word_rate: float = 0.05        # archaic words per 100 words
    lexical_diversity: float = 0.70        # type-token ratio

    # Focus and structure
    descriptive_focus: list[str] = field(default_factory=list)
    common_openings: list[str] = field(default_factory=list)
    common_closings: list[str] = field(default_factory=list)
    structural_pattern: str = ""

    # Canonical description
    description: str = ""
    trigger_conditions: list[str] = field(default_factory=list)
    example_passages: list[str] = field(default_factory=list)  # Canonical excerpts

    def to_neo4j_props(self) -> dict:
        return {
            "id": self.id,
            "register": self.register,
            "scene_type": self.scene_type,
            "avg_sentence_length": self.avg_sentence_length,
            "sentence_length_variance": self.sentence_length_variance,
            "passive_ratio": self.passive_ratio,
            "dialogue_density": self.dialogue_density,
            "archaic_word_rate": self.archaic_word_rate,
            "lexical_diversity": self.lexical_diversity,
            "descriptive_focus": self.descriptive_focus,
            "common_openings": self.common_openings,
            "common_closings": self.common_closings,
            "structural_pattern": self.structural_pattern,
            "description": self.description,
            "trigger_conditions": self.trigger_conditions,
        }

    def to_dict(self) -> dict:
        d = self.to_neo4j_props()
        d["example_passages"] = self.example_passages
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SceneTemplate":
        return cls(
            id=d.get("id", ""),
            register=d.get("register", ""),
            scene_type=d.get("scene_type", "general"),
            avg_sentence_length=float(d.get("avg_sentence_length", 15.0)),
            sentence_length_variance=float(d.get("sentence_length_variance", 5.0)),
            passive_ratio=float(d.get("passive_ratio", 0.15)),
            dialogue_density=float(d.get("dialogue_density", 0.20)),
            archaic_word_rate=float(d.get("archaic_word_rate", 0.05)),
            lexical_diversity=float(d.get("lexical_diversity", 0.70)),
            descriptive_focus=list(d.get("descriptive_focus", [])),
            common_openings=list(d.get("common_openings", [])),
            common_closings=list(d.get("common_closings", [])),
            structural_pattern=d.get("structural_pattern", ""),
            description=d.get("description", ""),
            trigger_conditions=list(d.get("trigger_conditions", [])),
            example_passages=list(d.get("example_passages", [])),
        )

    def generation_prompt_fragment(self) -> str:
        """Return a prompt fragment for generation that injects this register's style."""
        lines = [
            f"Write in the '{self.register}' register.",
            f"Structural pattern: {self.structural_pattern}",
            f"Target sentence length: ~{self.avg_sentence_length:.0f} words "
            f"(variance ±{self.sentence_length_variance:.0f}).",
        ]
        if self.passive_ratio > 0.25:
            lines.append(f"Use passive constructions freely ({self.passive_ratio:.0%} passive rate).")
        elif self.passive_ratio < 0.12:
            lines.append("Prefer active voice and direct syntax.")
        if self.dialogue_density > 0.4:
            lines.append(f"This is dialogue-rich ({self.dialogue_density:.0%} dialogue density) — let characters speak.")
        elif self.dialogue_density < 0.05:
            lines.append("This is narration-heavy — minimize dialogue.")
        if self.archaic_word_rate > 0.08:
            lines.append("Use archaic vocabulary: thee, thou, spake, wrought, nigh, ere, belike.")
        if self.descriptive_focus:
            lines.append(f"Imagery focus: {', '.join(self.descriptive_focus)}.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RegisterClassification — result of classifying a passage
# ---------------------------------------------------------------------------

@dataclass
class RegisterClassification:
    """The result of classifying a single passage by prose register.

    A passage can be tagged with multiple registers (with different confidences).
    """
    passage_id: str
    passage_text_snippet: str    # First 120 chars for display
    classifications: list[tuple[str, float]] = field(default_factory=list)
    # List of (register, confidence) tuples, sorted by confidence desc

    # Optional LLM reasoning
    reasoning: Optional[str] = None

    def primary_register(self) -> Optional[str]:
        """Return the highest-confidence register."""
        if not self.classifications:
            return None
        return max(self.classifications, key=lambda x: x[1])[0]

    def confident_registers(self, threshold: float = 0.5) -> list[str]:
        """Return registers above the confidence threshold."""
        return [r for r, c in self.classifications if c >= threshold]

    def to_dict(self) -> dict:
        return {
            "passage_id": self.passage_id,
            "snippet": self.passage_text_snippet,
            "classifications": [
                {"register": r, "confidence": c}
                for r, c in self.classifications
            ],
            "primary_register": self.primary_register(),
        }

    def summary(self) -> str:
        lines = [f"Register classification: {self.passage_id}"]
        if not self.classifications:
            lines.append("  No register detected.")
        else:
            for register, conf in sorted(self.classifications, key=lambda x: -x[1]):
                bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
                lines.append(f"  {register:<16} {bar} {conf:.2f}")
        if self.reasoning:
            lines.append(f"\n  Reasoning: {self.reasoning[:200]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ExemplifiesEdge — EXEMPLIFIES relationship
# ---------------------------------------------------------------------------

@dataclass
class ExemplifiesEdge:
    """Represents (Passage)-[:EXEMPLIFIES { confidence }]->(SceneTemplate)."""
    passage_id: str
    template_id: str       # SceneTemplate.id (= register name)
    confidence: float      # 0.0 - 1.0
    reasoning: str = ""

    def to_neo4j_props(self) -> dict:
        props = {"confidence": round(self.confidence, 3)}
        if self.reasoning:
            props["reasoning"] = self.reasoning[:500]
        return props
