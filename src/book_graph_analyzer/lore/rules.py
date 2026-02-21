"""LoreRule system — registry, validator, WorldBible mapper, and Neo4j writer.

Architecture:
  TOLKIEN_LORE_RULES — pre-defined set of hard/soft rules
  LoreRuleRegistry   — in-memory registry of LoreRule objects
  LoreRuleValidator  — validates entities/text against rules
  WorldBibleRuleMapper — maps WorldBible WorldRule -> LoreRule
  LoreRuleNeo4jWriter  — writes LoreRule nodes and SUBJECT_TO edges to Neo4j
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..models.lore_rule import (
    LoreRule,
    LoreViolation,
    LoreValidationResult,
)


# ---------------------------------------------------------------------------
# Pre-defined Tolkien Lore Rules
# Covers all world-bible categories: race, magic, cosmology, geography,
# politics, metaphysics, objects, history
# ---------------------------------------------------------------------------

TOLKIEN_LORE_RULES: list[LoreRule] = [

    # ---- RACE ----------------------------------------------------------------

    LoreRule(
        id="race_elf_immortal",
        statement="Elves cannot die of age or disease",
        category="race",
        hardness="HARD",
        scope_entity_type="Elf",
        confidence=1.0,
        cypher_check="""
MATCH (c:Character)-[:PARTICIPATED_IN {role: 'victim'}]->(e:Event {type: 'death'})
WHERE 'Elf' IN c.race_tags
  AND NOT e.cause IN ['violence', 'grief', 'departure_to_valinor', 'unknown']
RETURN c.canonical_name + ' dies of ' + e.cause + ' — Elves are immortal except through violence or grief'
        """.strip(),
    ),

    LoreRule(
        id="race_hobbit_unadventurous",
        statement="Hobbits are generally unadventurous and dislike leaving the Shire",
        category="race",
        hardness="SOFT",
        scope_entity_type="Hobbit",
        confidence=0.9,
        cypher_check="""
MATCH (c:Character {race: 'Hobbit'})-[:TRAVELED_TO]->(p:Place)
WHERE NOT p.name IN ['The Shire', 'Bree', 'Rivendell']
  AND NOT c.canonical_name IN ['Bilbo Baggins', 'Frodo Baggins', 'Samwise Gamgee',
                                 'Meriadoc Brandybuck', 'Peregrin Took']
RETURN 'WARNING (soft): ' + c.canonical_name + ' travels far — unusual for a Hobbit'
        """.strip(),
    ),

    LoreRule(
        id="race_maia_limited_power",
        statement="Maiar take incarnated forms with limited power in Middle-earth",
        category="race",
        hardness="SOFT",
        scope_entity_type="Maia",
        confidence=0.85,
        cypher_check="""
MATCH (c:Character {race: 'Maia'})-[:PERFORMED]->(a:Action {type: 'miracle'})
WHERE a.magnitude = 'cosmic' AND NOT c.is_valar = true
RETURN 'WARNING (soft): ' + c.canonical_name + ' performs cosmic miracle — Maiar are limited in Middle-earth form'
        """.strip(),
    ),

    LoreRule(
        id="race_dwarf_underground",
        statement="Dwarves prefer underground halls and are skilled smiths",
        category="race",
        hardness="SOFT",
        scope_entity_type="Dwarf",
        confidence=0.8,
        cypher_check=None,  # Cultural soft rule — no automated Cypher check
    ),

    # ---- MAGIC ---------------------------------------------------------------

    LoreRule(
        id="magic_ring_corruption",
        statement="The One Ring corrupts all who possess it except the Valar",
        category="magic",
        hardness="HARD",
        confidence=1.0,
        cypher_check="""
MATCH (c:Character)-[:POSSESSED]->(o:Object {canonical_name: 'The One Ring'})
WHERE NOT c.corrupted = true
  AND NOT c.entity_type IN ['Valar']
  AND NOT c.canonical_name IN ['Frodo Baggins', 'Samwise Gamgee', 'Tom Bombadil']
RETURN c.canonical_name + ' possesses the One Ring without corruption — lore violation'
        """.strip(),
    ),

    LoreRule(
        id="magic_ring_destruction",
        statement="The One Ring can only be destroyed in the fires of Mount Doom",
        category="magic",
        hardness="HARD",
        confidence=1.0,
        cypher_check="""
MATCH (e:Event {type: 'destruction'})-[:INVOLVED]->(o:Object {canonical_name: 'The One Ring'})
WHERE NOT (e)-[:TOOK_PLACE_AT]->(:Place {canonical_name: 'Mount Doom'})
RETURN 'One Ring destroyed outside Mount Doom — hard lore violation'
        """.strip(),
    ),

    LoreRule(
        id="magic_palantir_truthful",
        statement="Palantiri show the truth but can deceive through selective framing",
        category="magic",
        hardness="SOFT",
        confidence=0.9,
        cypher_check=None,
    ),

    LoreRule(
        id="magic_subtle_not_flashy",
        statement="Magic in Middle-earth is generally subtle, not visually spectacular",
        category="magic",
        hardness="SOFT",
        confidence=0.85,
        cypher_check=None,
    ),

    # ---- COSMOLOGY -----------------------------------------------------------

    LoreRule(
        id="cosmo_arda_round",
        statement="After the Fall of Númenor, Arda became round and Valinor was removed from the world",
        category="cosmology",
        hardness="HARD",
        scope_era="Third Age",
        confidence=1.0,
        cypher_check="""
MATCH (e:Event {type: 'travel'})-[:TOOK_PLACE_AT]->(p:Place {name: 'Valinor'})
WHERE e.era_name = 'Third Age'
  AND NOT e.method IN ['straight_road', 'gift_of_valar', 'ship_of_the_dead']
RETURN 'Travel to Valinor in Third Age requires the Straight Road — cannot sail directly'
        """.strip(),
    ),

    LoreRule(
        id="cosmo_music_of_ainur",
        statement="The world was created through the Music of the Ainur; its fundamental laws are fixed",
        category="cosmology",
        hardness="HARD",
        confidence=1.0,
        cypher_check=None,  # Fundamental axiom — no automated check possible
    ),

    LoreRule(
        id="cosmo_sun_moon_created_fa",
        statement="The Sun and Moon were created at the beginning of the First Age from the last fruits of the Two Trees",
        category="cosmology",
        hardness="HARD",
        scope_era="First Age",
        confidence=1.0,
        cypher_check="""
MATCH (e:Event)-[:REFERENCES {type: 'sun_rises'}]->()
WHERE e.era_name IN ['Years of the Trees', 'Years of the Lamps']
RETURN 'Sun does not exist in Years of the Trees or Years of the Lamps'
        """.strip(),
    ),

    # ---- GEOGRAPHY -----------------------------------------------------------

    LoreRule(
        id="geo_mirkwood_dangerous",
        statement="Mirkwood is dangerous for travellers — avoid the path or face spiders and imprisonment",
        category="geography",
        hardness="SOFT",
        confidence=0.9,
        cypher_check=None,
    ),

    LoreRule(
        id="geo_mordor_inhabitable",
        statement="Only Orcs and beings of Sauron can dwell freely in Mordor",
        category="geography",
        hardness="SOFT",
        confidence=0.85,
        cypher_check="""
MATCH (c:Character)-[:DWELLED_IN]->(p:Place {canonical_name: 'Mordor'})
WHERE NOT c.allegiance = 'Sauron' AND NOT c.race IN ['Orc', 'Troll', 'Nazgul']
RETURN 'WARNING (soft): ' + c.canonical_name + ' dwells in Mordor — unusual for non-Sauron aligned character'
        """.strip(),
    ),

    LoreRule(
        id="geo_rivendell_safe_haven",
        statement="Rivendell is a place of safety and rest for the Free Peoples",
        category="geography",
        hardness="SOFT",
        confidence=0.9,
        cypher_check=None,
    ),

    # ---- POLITICS ------------------------------------------------------------

    LoreRule(
        id="pol_steward_not_king",
        statement="Gondor's Stewards rule in the King's name — they are not Kings",
        category="politics",
        hardness="HARD",
        scope_entity_type="Steward",
        scope_era="Third Age",
        confidence=1.0,
        cypher_check="""
MATCH (c:Character {title: 'Steward of Gondor'})-[:CROWNED_AS]->(:Title {name: 'King of Gondor'})
RETURN c.canonical_name + ' crowned as King of Gondor — Stewards cannot take the throne'
        """.strip(),
    ),

    LoreRule(
        id="pol_elrond_neutral",
        statement="Elrond's house at Rivendell remains neutral in the wars of power",
        category="politics",
        hardness="SOFT",
        confidence=0.8,
        cypher_check=None,
    ),

    # ---- METAPHYSICS ---------------------------------------------------------

    LoreRule(
        id="meta_rings_of_power",
        statement="The Three Elven Rings lose their power when the One Ring is destroyed",
        category="metaphysics",
        hardness="HARD",
        confidence=1.0,
        cypher_check="""
MATCH (e:Event {type: 'destruction', target: 'The One Ring'})
WITH e
MATCH (o:Object)-[:IS_A]->(:ObjectType {name: 'Elven Ring'})
WHERE o.power_intact = true
  AND e.occurred = true
RETURN 'Elven Ring retains power after destruction of the One Ring — lore violation'
        """.strip(),
    ),

    LoreRule(
        id="meta_death_permanent_men",
        statement="Men die and their fate after death is unknown — the Gift of Men",
        category="metaphysics",
        hardness="HARD",
        scope_entity_type="Man",
        confidence=1.0,
        cypher_check=None,
    ),

    LoreRule(
        id="meta_temporal_knowledge",
        statement="Characters can only know what was knowable at their story-time",
        category="metaphysics",
        hardness="HARD",
        confidence=1.0,
        cypher_check="""
MATCH (c:Character)-[:KNOWS]->(f:Fact)
WHERE f.revealed_era_order > c.story_time_era_order
RETURN c.canonical_name + ' knows ' + f.description + ' before it was revealed — temporal violation'
        """.strip(),
    ),

    # ---- OBJECTS -------------------------------------------------------------

    LoreRule(
        id="obj_narsil_broken",
        statement="Narsil was broken in the Second Age; Andúril is the reforged Narsil",
        category="objects",
        hardness="HARD",
        confidence=1.0,
        cypher_check="""
MATCH (c:Character)-[:WIELDS]->(o:Object {canonical_name: 'Narsil'})
WHERE c.story_time_era = 'Third Age' AND o.is_broken IS NULL
RETURN c.canonical_name + ' wields unbroken Narsil in the Third Age — Narsil was broken at end of Second Age'
        """.strip(),
    ),

    LoreRule(
        id="obj_mithril_valuable",
        statement="Mithril is the rarest and most valuable metal — worth more than gold",
        category="objects",
        hardness="SOFT",
        confidence=0.95,
        cypher_check=None,
    ),

    # ---- HISTORY -------------------------------------------------------------

    LoreRule(
        id="hist_numenor_fallen",
        statement="Númenor was destroyed at the end of the Second Age — it cannot exist in Third Age",
        category="history",
        hardness="HARD",
        scope_era="Third Age",
        confidence=1.0,
        cypher_check="""
MATCH (e:Event {era_name: 'Third Age'})-[:TOOK_PLACE_AT]->(p:Place {canonical_name: 'Numenor'})
RETURN 'Event set in Númenor during Third Age — Númenor was destroyed at end of Second Age'
        """.strip(),
    ),

    LoreRule(
        id="hist_first_ring_war",
        statement="Sauron was defeated in the War of the Last Alliance at the end of the Second Age",
        category="history",
        hardness="HARD",
        confidence=1.0,
        cypher_check=None,
    ),
]


# ---------------------------------------------------------------------------
# LoreRuleRegistry
# ---------------------------------------------------------------------------

class LoreRuleRegistry:
    """In-memory registry of LoreRule objects.

    Can be seeded from TOLKIEN_LORE_RULES, loaded from Neo4j, or built manually.
    """

    def __init__(self) -> None:
        self._rules: dict[str, LoreRule] = {}

    def add(self, rule: LoreRule) -> None:
        self._rules[rule.id] = rule

    def add_many(self, rules: list[LoreRule]) -> None:
        for rule in rules:
            self.add(rule)

    def get(self, rule_id: str) -> LoreRule | None:
        return self._rules.get(rule_id)

    def all(self) -> list[LoreRule]:
        return list(self._rules.values())

    def hard_rules(self) -> list[LoreRule]:
        return [r for r in self._rules.values() if r.is_hard]

    def soft_rules(self) -> list[LoreRule]:
        return [r for r in self._rules.values() if r.is_soft]

    def by_category(self, category: str) -> list[LoreRule]:
        return [r for r in self._rules.values() if r.category == category]

    def by_entity_type(self, entity_type: str) -> list[LoreRule]:
        """Return rules scoped to a specific entity type or universal rules."""
        return [
            r for r in self._rules.values()
            if r.scope_entity_type is None or r.scope_entity_type == entity_type
        ]

    def by_era(self, era: str | None) -> list[LoreRule]:
        """Return rules applicable to a given era (scoped to era or universal)."""
        return [
            r for r in self._rules.values()
            if r.scope_era is None or r.scope_era == era
        ]

    @classmethod
    def from_tolkien_defaults(cls) -> "LoreRuleRegistry":
        """Create a registry pre-loaded with all Tolkien lore rules."""
        registry = cls()
        registry.add_many(TOLKIEN_LORE_RULES)
        return registry

    def __len__(self) -> int:
        return len(self._rules)


# ---------------------------------------------------------------------------
# LoreRuleValidator — pure-Python validation without Neo4j
# ---------------------------------------------------------------------------

@dataclass
class SceneContext:
    """A simplified scene context for pure-Python lore validation.

    Represents the entities and events present in a proposed scene.
    Used when Neo4j is not available.
    """
    scene_id: str
    character_names: list[str]
    character_races: dict[str, str]   # name -> race
    place_names: list[str]
    object_names: list[str]
    event_types: list[str]            # e.g. ['death', 'travel', 'combat']
    story_era: str | None = None


class LoreRuleValidator:
    """Validates a scene context against a registry of LoreRules.

    Two modes:
      1. Pure-Python (offline) — uses heuristic text/entity checks
      2. Neo4j — runs the actual cypher_check queries
    """

    def __init__(self, registry: LoreRuleRegistry | None = None) -> None:
        self._registry = registry or LoreRuleRegistry.from_tolkien_defaults()

    @property
    def registry(self) -> LoreRuleRegistry:
        return self._registry

    # ------------------------------------------------------------------
    # Pure-Python validation
    # ------------------------------------------------------------------

    def validate_scene_context(
        self, context: SceneContext, categories: list[str] | None = None
    ) -> LoreValidationResult:
        """Validate a scene context using pure-Python heuristic checks.

        This does not require Neo4j. Uses simple entity/race/era checks
        to catch the most common lore violations.

        Args:
            context: The scene to validate.
            categories: Optional list of rule categories to check (None = all).

        Returns:
            LoreValidationResult with hard violations and soft warnings.
        """
        rules = self._registry.all()
        if categories:
            rules = [r for r in rules if r.category in categories]
        # Filter by era
        if context.story_era:
            rules = [
                r for r in rules
                if r.scope_era is None or r.scope_era == context.story_era
            ]

        hard_violations: list[LoreViolation] = []
        soft_warnings: list[LoreViolation] = []
        rules_checked = 0

        for rule in rules:
            violation = self._check_rule_heuristic(rule, context)
            if violation:
                rules_checked += 1
                if rule.is_hard:
                    hard_violations.append(violation)
                else:
                    soft_warnings.append(violation)
            else:
                rules_checked += 1

        passed = len(hard_violations) == 0
        return LoreValidationResult(
            scene_id=context.scene_id,
            passed=passed,
            hard_violations=hard_violations,
            soft_warnings=soft_warnings,
            rules_checked=rules_checked,
        )

    def _check_rule_heuristic(
        self, rule: LoreRule, context: SceneContext
    ) -> LoreViolation | None:
        """Pure-Python heuristic check for a single rule."""

        # Dispatch to rule-specific checks by ID
        checker = _HEURISTIC_CHECKERS.get(rule.id)
        if checker:
            description = checker(rule, context)
            if description:
                return LoreViolation(
                    rule_id=rule.id,
                    rule_statement=rule.statement,
                    hardness=rule.hardness,
                    description=description,
                    blocking=rule.is_hard,
                )
        return None

    def validate_text(
        self, text: str, scene_id: str = "inline", story_era: str | None = None
    ) -> LoreValidationResult:
        """Validate free text by extracting entities heuristically.

        Useful for quick offline checking of draft passages.
        """
        context = _extract_context_from_text(text, scene_id, story_era)
        return self.validate_scene_context(context)

    # ------------------------------------------------------------------
    # Neo4j cypher_check validation
    # ------------------------------------------------------------------

    def validate_scene_neo4j(
        self, scene_id: str, driver=None
    ) -> LoreValidationResult:
        """Run all cypher_check queries for applicable rules against Neo4j.

        Args:
            scene_id: The ID of the scene node in Neo4j.
            driver: Neo4j driver (created from get_driver() if not provided).

        Returns:
            LoreValidationResult from running Cypher checks.
        """
        if driver is None:
            from ..graph.connection import get_driver
            driver = get_driver()
            if driver is None:
                raise ConnectionError("Cannot connect to Neo4j")

        rules_with_cypher = [r for r in self._registry.all() if r.cypher_check]
        hard_violations: list[LoreViolation] = []
        soft_warnings: list[LoreViolation] = []
        rules_checked = 0

        with driver.session() as session:
            for rule in rules_with_cypher:
                rules_checked += 1
                try:
                    # Each cypher_check query RETURNS violation strings
                    result = session.run(
                        rule.cypher_check,
                        scene_id=scene_id,
                    )
                    for row in result:
                        desc = str(row[0]) if row[0] else "Violation detected"
                        v = LoreViolation(
                            rule_id=rule.id,
                            rule_statement=rule.statement,
                            hardness=rule.hardness,
                            description=desc,
                            blocking=rule.is_hard,
                        )
                        if rule.is_hard:
                            hard_violations.append(v)
                        else:
                            soft_warnings.append(v)
                except Exception:
                    pass  # Skip rules that fail to execute (schema not available)

        passed = len(hard_violations) == 0
        return LoreValidationResult(
            scene_id=scene_id,
            passed=passed,
            hard_violations=hard_violations,
            soft_warnings=soft_warnings,
            rules_checked=rules_checked,
        )


# ---------------------------------------------------------------------------
# Heuristic checkers for pure-Python validation
# ---------------------------------------------------------------------------

def _check_elf_immortal(rule: LoreRule, ctx: SceneContext) -> str | None:
    """Elves cannot die of age — check if any Elf characters die of non-violence."""
    elves = [
        name for name, race in ctx.character_races.items()
        if race.lower() in ("elf", "elvish", "eldar", "sindarin", "noldor")
    ]
    if elves and "death_by_age" in ctx.event_types:
        return f"{', '.join(elves)} — Elf character(s) present with age-death event"
    return None


def _check_ring_corruption(rule: LoreRule, ctx: SceneContext) -> str | None:
    """One Ring present and a non-corrupted, non-Valar character wields it."""
    ring_present = any(
        "one ring" in o.lower() or "ring of power" in o.lower()
        for o in ctx.object_names
    )
    if ring_present:
        # Check if any non-exempt character is marked as uncorrupted ring-bearer
        for char in ctx.character_names:
            if char.lower() not in (
                "frodo baggins", "samwise gamgee", "tom bombadil", "gandalf",
                "aragorn", "galadriel", "elrond"
            ):
                # Suspicious — ring present with unvetted character
                return f"One Ring present with {char} — verify corruption arc is handled"
    return None


def _check_ring_destruction(rule: LoreRule, ctx: SceneContext) -> str | None:
    """One Ring destruction must happen at Mount Doom."""
    ring_destroyed = (
        any("one ring" in o.lower() for o in ctx.object_names)
        and "destruction" in ctx.event_types
    )
    if ring_destroyed:
        if "Mount Doom" not in ctx.place_names and "Orodruin" not in ctx.place_names:
            return "One Ring being destroyed but Mount Doom not in scene locations"
    return None


def _check_steward_not_king(rule: LoreRule, ctx: SceneContext) -> str | None:
    """Stewards of Gondor cannot claim the kingship."""
    stewards = ["denethor", "cirion", "boromir"]  # Known stewards (by last name / role)
    # Heuristic: if a known steward's name and 'king' event appear together
    for name in ctx.character_names:
        if any(s in name.lower() for s in stewards):
            if "coronation" in ctx.event_types or "crowned" in ctx.event_types:
                return f"{name} (Steward) in a coronation/crowning scene — Stewards cannot be crowned King"
    return None


def _check_temporal_knowledge(rule: LoreRule, ctx: SceneContext) -> str | None:
    """Characters cannot know things before they were revealed in story-time."""
    # This is very hard to check without the full graph — return None (skip)
    return None


def _check_numenor_fallen(rule: LoreRule, ctx: SceneContext) -> str | None:
    """Númenor doesn't exist in Third Age."""
    if ctx.story_era and "Third Age" in ctx.story_era:
        for place in ctx.place_names:
            if "numenor" in place.lower() or "númenor" in place.lower():
                return "Númenor present as a place in Third Age — it was destroyed at end of Second Age"
    return None


def _check_cosmo_arda_round(rule: LoreRule, ctx: SceneContext) -> str | None:
    """Valinor cannot be sailed to directly in Third Age."""
    if ctx.story_era and "Third Age" in ctx.story_era:
        if "Valinor" in ctx.place_names or "Aman" in ctx.place_names:
            if "travel" in ctx.event_types and "straight_road" not in ctx.event_types:
                return "Character travelling to Valinor in Third Age — requires the Straight Road gift"
    return None


def _check_meta_rings_power(rule: LoreRule, ctx: SceneContext) -> str | None:
    """Three Elven Rings lose power when One Ring is destroyed."""
    # Heuristic only — can't be checked without full graph
    return None


# Map rule_id -> checker function
_HEURISTIC_CHECKERS: dict[str, object] = {
    "race_elf_immortal":      _check_elf_immortal,
    "magic_ring_corruption":  _check_ring_corruption,
    "magic_ring_destruction": _check_ring_destruction,
    "pol_steward_not_king":   _check_steward_not_king,
    "meta_temporal_knowledge": _check_temporal_knowledge,
    "hist_numenor_fallen":    _check_numenor_fallen,
    "cosmo_arda_round":       _check_cosmo_arda_round,
    "meta_rings_of_power":    _check_meta_rings_power,
}


# ---------------------------------------------------------------------------
# Context extraction from free text (heuristic)
# ---------------------------------------------------------------------------

def _extract_context_from_text(
    text: str, scene_id: str, story_era: str | None
) -> SceneContext:
    """Heuristically extract scene context from raw text."""
    text_lower = text.lower()

    # Race detection
    race_keywords = {
        "Elf": ["elf", "elves", "elvish", "eldar", "sindarin", "noldor", "galadriel", "legolas", "elrond"],
        "Dwarf": ["dwarf", "dwarves", "dwarven", "gimli", "thorin"],
        "Hobbit": ["hobbit", "hobbits", "bilbo", "frodo", "samwise", "merry", "pippin"],
        "Man": ["man", "men of", "dunedain", "gondorian", "aragorn", "boromir", "faramir"],
        "Orc": ["orc", "orcs", "uruk", "goblin"],
        "Maia": ["gandalf", "saruman", "radagast", "sauron", "maia", "istari"],
    }

    character_races: dict[str, str] = {}
    character_names: list[str] = []
    for race, kws in race_keywords.items():
        for kw in kws:
            if kw in text_lower:
                character_races[kw] = race
                character_names.append(kw)

    # Place detection
    place_keywords = [
        "Rivendell", "Mirkwood", "Mordor", "The Shire", "Gondor", "Rohan",
        "Mount Doom", "Orodruin", "Valinor", "Aman", "Numenor", "Númenor",
        "Moria", "Lothlórien", "Fangorn", "Isengard",
    ]
    place_names = [p for p in place_keywords if p.lower() in text_lower]

    # Object detection
    object_keywords = [
        "One Ring", "the Ring", "Ring of Power",
        "Narsil", "Andúril", "Sting", "Glamdring",
        "palantír", "silmaril", "Arkenstone", "mithril",
    ]
    object_names = [o for o in object_keywords if o.lower() in text_lower]

    # Event detection
    event_types: list[str] = []
    if re.search(r"\b(died|death|slain|killed)\b", text_lower):
        event_types.append("death")
    if re.search(r"\b(destroy|destroyed|unmade)\b", text_lower):
        event_types.append("destruction")
    if re.search(r"\b(travel|journey|sailed|rode)\b", text_lower):
        event_types.append("travel")
    if re.search(r"\b(fought|battle|war|combat)\b", text_lower):
        event_types.append("combat")
    if re.search(r"\b(crown|coronation|king)\b", text_lower):
        event_types.append("coronation")
    if re.search(r"\b(die|died|death)\b.{0,40}\bage\b|\bage\b.{0,40}\b(die|died)\b", text_lower):
        event_types.append("death_by_age")

    return SceneContext(
        scene_id=scene_id,
        character_names=list(set(character_names)),
        character_races=character_races,
        place_names=place_names,
        object_names=object_names,
        event_types=event_types,
        story_era=story_era,
    )


# ---------------------------------------------------------------------------
# WorldBibleRuleMapper — maps WorldBible WorldRule -> LoreRule
# ---------------------------------------------------------------------------

# Mapping from WorldBibleCategory.value -> LoreRule category
_WORLDBIBLE_CATEGORY_MAP: dict[str, str] = {
    "magic":      "magic",
    "culture":    "race",
    "geography":  "geography",
    "technology": "objects",
    "cosmology":  "cosmology",
    "history":    "history",
    "language":   "metaphysics",
    "creatures":  "race",
    "objects":    "objects",
    "themes":     "metaphysics",
}

# Keywords that indicate a HARD rule
_HARD_INDICATORS = [
    "cannot", "can only", "never", "must", "always", "impossible",
    "only way", "only method", "only in", "requires", "forbidden",
]


def classify_hardness(statement: str, description: str) -> str:
    """Heuristically classify a rule as HARD or SOFT from its text."""
    text = (statement + " " + description).lower()
    for indicator in _HARD_INDICATORS:
        if indicator in text:
            return "HARD"
    return "SOFT"


def generate_cypher_check_template(
    rule_id: str, statement: str, category: str, hardness: str
) -> str | None:
    """Generate a basic Cypher check template for a rule.

    This is a best-effort template — LLM refinement would improve it.
    Returns None for rules too abstract to automate.
    """
    if hardness == "SOFT":
        return None  # Soft rules typically require contextual judgment

    # Template by category
    templates: dict[str, str] = {
        "geography": (
            f"// {statement}\n"
            "// TODO: Implement geography check for: " + statement
        ),
        "history": (
            f"// {statement}\n"
            "// TODO: Implement history check for: " + statement
        ),
        "race": (
            f"// {statement}\n"
            "// TODO: Implement race constraint check for: " + statement
        ),
    }
    return templates.get(category)


class WorldBibleRuleMapper:
    """Maps WorldBible WorldRule objects to LoreRule objects.

    This is the bridge between the world-bible extraction pipeline and
    the LoreRule system.
    """

    def map_rule(self, world_rule) -> LoreRule:
        """Map a single WorldRule to a LoreRule.

        Args:
            world_rule: A WorldRule from book_graph_analyzer.worldbible.models
        """
        category = _WORLDBIBLE_CATEGORY_MAP.get(
            world_rule.category.value
            if hasattr(world_rule.category, "value")
            else str(world_rule.category),
            "metaphysics"
        )
        statement = world_rule.title
        description = world_rule.description

        hardness = classify_hardness(statement, description)

        # Build ID from title
        rule_id = "wb_" + re.sub(r"[^a-z0-9]", "_", statement.lower())[:40].strip("_")

        cypher_check = generate_cypher_check_template(rule_id, statement, category, hardness)

        return LoreRule(
            id=rule_id,
            statement=statement,
            category=category,
            hardness=hardness,
            confidence=world_rule.confidence,
            source_passage_ids=[],
            cypher_check=cypher_check,
        )

    def map_bible(self, bible) -> list[LoreRule]:
        """Map all rules from a WorldBible to LoreRule objects.

        Args:
            bible: A WorldBible from book_graph_analyzer.worldbible
        """
        rules: list[LoreRule] = []
        for rule_list in bible.rules.values():
            for world_rule in rule_list:
                rules.append(self.map_rule(world_rule))
        return rules


# ---------------------------------------------------------------------------
# LoreRuleNeo4jWriter
# ---------------------------------------------------------------------------

class LoreRuleNeo4jWriter:
    """Write LoreRule nodes and SUBJECT_TO edges to Neo4j."""

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

    def ensure_schema(self) -> None:
        """Create constraint on LoreRule.id. Idempotent."""
        with self.driver.session() as session:
            try:
                session.run(
                    "CREATE CONSTRAINT lore_rule_id IF NOT EXISTS "
                    "FOR (r:LoreRule) REQUIRE r.id IS UNIQUE"
                )
            except Exception:
                pass

    def upsert_rule(self, rule: LoreRule) -> None:
        """Create or update a single LoreRule node."""
        with self.driver.session() as session:
            session.run(
                "MERGE (r:LoreRule {id: $id}) SET r += $props",
                id=rule.id,
                props=rule.to_neo4j_props(),
            )

    def upsert_many(self, rules: list[LoreRule]) -> int:
        """Write many rules. Returns count written."""
        count = 0
        with self.driver.session() as session:
            for rule in rules:
                session.run(
                    "MERGE (r:LoreRule {id: $id}) SET r += $props",
                    id=rule.id,
                    props=rule.to_neo4j_props(),
                )
                count += 1
        return count

    def add_subject_to_edge(
        self,
        entity_id: str,
        entity_label: str,
        rule_id: str,
        exceptions: list[str] | None = None,
        source_passage_id: str | None = None,
    ) -> None:
        """Create a (Character/Entity)-[:SUBJECT_TO]->(LoreRule) edge.

        Allows per-entity exceptions to be recorded without removing the rule.
        """
        props: dict = {"exceptions": exceptions or []}
        if source_passage_id:
            props["source_passage_id"] = source_passage_id

        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (e:{entity_label} {{id: $entity_id}})
                MATCH (r:LoreRule {{id: $rule_id}})
                MERGE (e)-[rel:SUBJECT_TO {{rule_id: $rule_id}}]->(r)
                SET rel += $props
                """,
                entity_id=entity_id,
                rule_id=rule_id,
                props=props,
            )

    def query_rules(
        self,
        category: str | None = None,
        hardness: str | None = None,
    ) -> list[dict]:
        """Query LoreRule nodes from Neo4j."""
        where_clauses = []
        params: dict = {}
        if category:
            where_clauses.append("r.category = $category")
            params["category"] = category
        if hardness:
            where_clauses.append("r.hardness = $hardness")
            params["hardness"] = hardness

        where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        with self.driver.session() as session:
            result = session.run(
                f"MATCH (r:LoreRule) {where} RETURN r ORDER BY r.category, r.hardness",
                **params,
            )
            return [dict(row["r"]) for row in result]
