"""NarrativeWeight — composite interestingness metric for Passage and Scene nodes.

NarrativeWeight is a float 0.0-1.0 computed per Passage (and per generated Scene).
It has named components so you can see *why* something scores high or low, and use
those components as generation targets.

Also contains ThemeNode model and TOLKIEN_THEMES taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Component weights for the overall composite score
# ---------------------------------------------------------------------------

COMPONENT_WEIGHTS: dict[str, float] = {
    "temporal_depth":            0.10,
    "era_reference_count":       0.07,
    "lore_density":              0.10,
    "entity_reference_count":    0.07,
    "thematic_threads":          0.12,
    "theme_coherence":           0.05,
    "revelation_count":          0.10,
    "callback_density":          0.07,
    "foreshadowing_count":       0.05,
    "dramatic_irony":            0.08,
    "character_revelation":      0.06,
    "voice_distinctiveness":     0.04,
    "emotional_register_count":  0.05,
    "emotional_contrast":        0.04,
}

# Verify weights sum to 1.0 (within float tolerance)
assert abs(sum(COMPONENT_WEIGHTS.values()) - 1.0) < 1e-9, \
    f"Component weights must sum to 1.0, got {sum(COMPONENT_WEIGHTS.values())}"

# Human-readable improvement suggestions per low-scoring component
COMPONENT_SUGGESTIONS: dict[str, str] = {
    "temporal_depth":
        "Add references to older eras — have a character speak of ancient history or primordial lore.",
    "era_reference_count":
        "Weave in references to multiple eras — the scene can touch the Second Age, First Age, or even Before Time.",
    "lore_density":
        "Ground the passage with more named lore facts: artifacts, events, places, or laws of the world.",
    "entity_reference_count":
        "Name more characters, places, or objects — 'the sword that Elendil carried' beats 'a sword'.",
    "thematic_threads":
        "Layer in a Tolkien theme explicitly: diminishment, the long defeat, hope vs. despair, or eucatastrophe.",
    "theme_coherence":
        "Ensure the themes you've chosen reinforce each other rather than pulling in opposite directions.",
    "revelation_count":
        "Give the reader something new — a secret revealed, a character's true nature, a hidden history.",
    "callback_density":
        "Callback to earlier established events or details to create structural resonance.",
    "foreshadowing_count":
        "Plant a seed — a detail that will matter later, stated without underscoring.",
    "dramatic_irony":
        "Create asymmetry: the reader knows something the character doesn't (or vice versa).",
    "character_revelation":
        "Reveal something about a character's inner life — not just what they do, but who they are.",
    "voice_distinctiveness":
        "Make each speaker's voice unmistakable — Gandalf's gravitas, Sam's plainspokenness, Frodo's reflective quality.",
    "emotional_register_count":
        "Layer multiple emotional registers simultaneously — sorrow and wonder at once, or dread and courage.",
    "emotional_contrast":
        "Juxtapose light and dark in the same passage — a moment of beauty within fear, or grief within victory.",
}


@dataclass
class NarrativeWeight:
    """Composite narrative interestingness score for a Passage or Scene.

    All component fields are floats in [0.0, 1.0].
    `overall` is the weighted average of all components.
    """

    # Temporal complexity
    temporal_depth: float = 0.0         # How far back does this reach?
    era_reference_count: float = 0.0    # How many distinct eras referenced?

    # Lore density
    lore_density: float = 0.0           # Distinct lore facts per 100 words
    entity_reference_count: float = 0.0 # Named entities referenced

    # Thematic resonance
    thematic_threads: float = 0.0       # How many major themes touched?
    theme_coherence: float = 0.0        # Do the themes reinforce each other?

    # Narrative structure
    revelation_count: float = 0.0       # New information delivered to reader
    callback_density: float = 0.0       # References to earlier established events
    foreshadowing_count: float = 0.0    # Seeds planted for future payoff
    dramatic_irony: float = 0.0         # Reader/character knowledge asymmetry

    # Character depth
    character_revelation: float = 0.0   # New insight into a character's nature
    voice_distinctiveness: float = 0.0  # How characteristically each speaker sounds

    # Emotional complexity
    emotional_register_count: float = 0.0  # How many Tolkien registers active
    emotional_contrast: float = 0.0        # Light and dark in same passage

    # Composite
    overall: float = 0.0  # Weighted average of all above

    def compute_overall(self) -> "NarrativeWeight":
        """Return a new NarrativeWeight with `overall` recomputed from components."""
        overall = sum(
            getattr(self, comp) * weight
            for comp, weight in COMPONENT_WEIGHTS.items()
        )
        return NarrativeWeight(**{**asdict(self), "overall": round(overall, 4)})

    def weakest_components(self, n: int = 3) -> list[tuple[str, float]]:
        """Return the n weakest component (name, score) pairs, sorted lowest-first."""
        scores = [
            (comp, getattr(self, comp))
            for comp in COMPONENT_WEIGHTS
        ]
        scores.sort(key=lambda x: x[1])
        return scores[:n]

    def improvement_suggestions(self, n: int = 3) -> list[str]:
        """Return improvement suggestions for the n weakest components."""
        weak = self.weakest_components(n)
        return [
            f"[{comp} = {score:.2f}] {COMPONENT_SUGGESTIONS[comp]}"
            for comp, score in weak
        ]

    def to_dict(self) -> dict:
        """Serialise to a flat dict (suitable for Neo4j node properties)."""
        return {f"nw_{k}": round(v, 4) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "NarrativeWeight":
        """Deserialise from a dict that may have 'nw_' prefixed keys or plain keys."""
        fields = {f for f in cls.__dataclass_fields__}
        kwargs = {}
        for k, v in d.items():
            key = k[3:] if k.startswith("nw_") else k
            if key in fields:
                kwargs[key] = float(v)
        return cls(**kwargs)

    def summary(self, passage_id: str = "") -> str:
        """Return a human-readable summary table."""
        lines = []
        if passage_id:
            lines.append(f"NarrativeWeight for: {passage_id}")
        lines.append(f"  Overall: {self.overall:.3f}")
        lines.append("")
        lines.append("  Components:")
        for comp, weight in COMPONENT_WEIGHTS.items():
            score = getattr(self, comp)
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(f"    {comp:<28} {bar} {score:.2f}  (w={weight:.2f})")
        lines.append("")
        lines.append("  Improvement suggestions:")
        for suggestion in self.improvement_suggestions(3):
            lines.append(f"    • {suggestion}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ThemeNode
# ---------------------------------------------------------------------------

@dataclass
class ThemeNode:
    """A thematic concept that can be tagged on Passages, Events, and LoreRules."""

    id: str                     # e.g. 'eucatastrophe'
    name: str                   # Human-readable name
    description: str            # What this theme means
    tolkien_specific: bool = True  # Is this a distinctively Tolkien theme?

    # Keywords used for rule-based detection in passage text
    detection_keywords: list[str] = field(default_factory=list)

    def to_neo4j_props(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tolkien_specific": self.tolkien_specific,
        }


# ---------------------------------------------------------------------------
# Tolkien Theme Taxonomy — 10 core themes (Acceptance Criterion)
# ---------------------------------------------------------------------------

TOLKIEN_THEMES: list[ThemeNode] = [
    ThemeNode(
        id="eucatastrophe",
        name="Eucatastrophe",
        description=(
            "The sudden joyous turn — the unexpected good event that comes at the darkest hour "
            "and gives a fleeting glimpse of a Joy beyond the walls of the world."
        ),
        tolkien_specific=True,
        detection_keywords=[
            "hope", "sudden", "turn", "joy", "despite all", "unexpected salvation",
            "eagles", "against all hope", "dawn", "saved", "victory", "eucatastrophe",
        ],
    ),
    ThemeNode(
        id="the_long_defeat",
        name="The Long Defeat",
        description=(
            "Virtuous struggle against evil that nevertheless fades and is lost over ages — "
            "the courage to fight knowing you will ultimately lose."
        ),
        tolkien_specific=True,
        detection_keywords=[
            "defeat", "fading", "losing", "diminish", "lost", "age after age",
            "fighting the long defeat", "ever losing", "long struggle", "in vain",
        ],
    ),
    ThemeNode(
        id="diminishment",
        name="Diminishment",
        description=(
            "The irreversible fading of great things over time — beauty, power, and glory "
            "that once blazed now grown pale and thin."
        ),
        tolkien_specific=True,
        detection_keywords=[
            "fading", "diminished", "lesser", "shadow of what was", "pale", "thin",
            "shadow", "waning", "glory", "what once was", "grown old", "passing",
        ],
    ),
    ThemeNode(
        id="the_past_pressing_on_present",
        name="The Past Pressing on the Present",
        description=(
            "Ancient history making itself felt in the present moment — the weight of deep time "
            "active in every contemporary event."
        ),
        tolkien_specific=True,
        detection_keywords=[
            "long ago", "ages past", "in the elder days", "since the world was young",
            "ancient", "of old", "in days of yore", "before your time",
            "history", "ages", "remember when", "first age", "second age",
        ],
    ),
    ThemeNode(
        id="power_corrupts",
        name="Power Corrupts",
        description=(
            "The possession of great power corrupts those who seek or hold it — "
            "the One Ring as the purest expression of this truth."
        ),
        tolkien_specific=False,
        detection_keywords=[
            "power", "ring", "corrupted", "possessed", "consumed", "dominated",
            "will to dominate", "mastery", "control", "enslaved", "fallen",
        ],
    ),
    ThemeNode(
        id="hope_vs_despair",
        name="Hope vs. Despair",
        description=(
            "The fundamental tension between hope that good will endure and despair "
            "that evil is too great — often embodied in the same character or moment."
        ),
        tolkien_specific=False,
        detection_keywords=[
            "hope", "despair", "dark", "light", "no hope", "all is lost",
            "must go on", "while there is hope", "give up", "endure", "cannot be done",
            "seemingly hopeless",
        ],
    ),
    ThemeNode(
        id="mortality",
        name="Mortality",
        description=(
            "The gift and burden of mortality — the Men's fate of death contrasted "
            "with the Elves' immortality, and the meaning found in a mortal life."
        ),
        tolkien_specific=False,
        detection_keywords=[
            "death", "mortal", "immortal", "die", "dying", "last", "end",
            "after death", "beyond the circles of the world", "gift of men",
            "grief", "parting", "farewell",
        ],
    ),
    ThemeNode(
        id="loyalty",
        name="Loyalty and Faithful Service",
        description=(
            "Steadfast fidelity and service freely given — Sam's devotion to Frodo, "
            "Aragorn's loyalty to the Fellowship, the Ents' slow commitment."
        ),
        tolkien_specific=False,
        detection_keywords=[
            "loyal", "faithful", "serve", "devoted", "together", "will not leave",
            "follow", "master", "friend", "beside you", "sworn", "oath",
        ],
    ),
    ThemeNode(
        id="mercy",
        name="Mercy and Pity",
        description=(
            "The unexpected power of pity and mercy — Bilbo's pity for Gollum, "
            "Frodo's mercy, the eucatastrophe enabled by a chain of unmerited grace."
        ),
        tolkien_specific=False,
        detection_keywords=[
            "mercy", "pity", "spare", "forgive", "forgiveness", "compassion",
            "could not kill", "let him go", "grace", "kind", "gentle",
        ],
    ),
    ThemeNode(
        id="stewardship",
        name="Stewardship",
        description=(
            "The responsibility to preserve and protect rather than dominate — "
            "Faramir vs. Boromir, the Valar's guardianship, the Ents' care for the forest."
        ),
        tolkien_specific=True,
        detection_keywords=[
            "steward", "guard", "protect", "preserve", "custodian", "care for",
            "not dominate", "servant not master", "keep", "tend", "oversee",
        ],
    ),
]

# Lookup dict for fast access
THEME_BY_ID: dict[str, ThemeNode] = {t.id: t for t in TOLKIEN_THEMES}
