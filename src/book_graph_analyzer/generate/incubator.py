"""Lore Incubator — pre-drafting invention engine.

Runs BEFORE the Outliner generates a chapter. Invents new entities
(characters, locations, artifacts) to populate whitespace gaps in the
canonical record, then commits them to the Shadow Graph as local canon.

The key principle: separate INVENTION from PROSE-WRITING.
The Drafter never has to be creative — it just executes the Shadow Graph's facts.
"""

import json
import random
import re
from dataclasses import dataclass, field
from typing import Optional

from ..llm import LLMClient
from .shadow.graph import ShadowGraph


# ─── Trope Dictionary ────────────────────────────────────────────────────────

class TropeDictionary:
    """
    Built-in mythic trope seeds.

    Tolkien's "serendipitous" moments weren't random — they were deep
    structural patterns from mythology. Injecting one trope per chapter
    gives the LLM a narrative skeleton to build toward, producing moments
    that feel inevitable in retrospect.
    """

    TROPES: list[dict] = [
        {
            "name": "The Relic in the Dark",
            "description": (
                "The protagonist falls or is trapped in a dark, enclosed place. "
                "While there, they discover a forgotten item of power that will aid them later."
            ),
            "required_elements": ["dark_enclosed_place", "ancient_item"],
            "scene_type_hint": "discovery",
            "tolkien_examples": [
                "Bilbo finds the One Ring in Gollum's cave",
                "The Fellowship discovers Balin's tomb in Moria",
            ],
        },
        {
            "name": "The Threshold Guardian",
            "description": (
                "A figure — hostile, wise, or both — blocks the path and must be "
                "confronted, bargained with, or outwitted before progress can continue."
            ),
            "required_elements": ["obstacle_figure", "threshold_location"],
            "scene_type_hint": "council",
            "tolkien_examples": [
                "The Balrog at the Bridge of Khazad-dûm",
                "Tom Bombadil in the Old Forest",
                "Haldir at the borders of Lórien",
            ],
        },
        {
            "name": "The Hidden Valley",
            "description": (
                "A secret, protected place is revealed to the protagonist — a refuge "
                "that exists outside normal geography, accessible only to the worthy."
            ),
            "required_elements": ["hidden_entrance", "protected_interior"],
            "scene_type_hint": "discovery",
            "tolkien_examples": [
                "The revelation of Rivendell",
                "The first sight of Gondolin from the Dry River",
                "Lothlórien seen from the Nimrodel",
            ],
        },
        {
            "name": "Eucatastrophe",
            "description": (
                "At the moment of greatest despair, an entirely unexpected turn "
                "reverses fortune — a sudden, joyous salvation the protagonist did not engineer."
            ),
            "required_elements": ["crisis_point", "unexpected_salvation"],
            "scene_type_hint": "battle",
            "tolkien_examples": [
                "The Eagles at the Battle of Morannon",
                "Gollum's fall destroying the Ring",
                "The Rohirrim arriving at the Pelennor",
            ],
        },
        {
            "name": "The Reluctant Guide",
            "description": (
                "A morally ambiguous or unwilling figure leads the protagonist through "
                "danger they could not navigate alone — their motives are unclear but necessary."
            ),
            "required_elements": ["guide_figure", "dangerous_path"],
            "scene_type_hint": "journey",
            "tolkien_examples": [
                "Gollum leading Frodo and Sam to Mordor",
                "Strider revealing himself at Bree",
            ],
        },
        {
            "name": "The Divine Messenger",
            "description": (
                "A figure of divine or Maia origin appears — often in disguise — "
                "to deliver a task, warning, or gift that sets the protagonist on their path."
            ),
            "required_elements": ["messenger_figure", "sacred_instruction"],
            "scene_type_hint": "myth_cosmogony",
            "tolkien_examples": [
                "Ulmo appears to Tuor at Vinyamar",
                "Gandalf's true nature as a Maia revealed",
                "The vision of Manwë given to Frodo on Amon Hen",
            ],
        },
        {
            "name": "The Corrupted Place",
            "description": (
                "A location that was once fair is now twisted. Its history of evil "
                "is written in the landscape — beauty and ruin coexist, and the past speaks."
            ),
            "required_elements": ["ruined_location", "history_of_evil"],
            "scene_type_hint": "journey",
            "tolkien_examples": [
                "Mordor (once a fair valley before Sauron came)",
                "Dol Guldur in the dark wood",
                "Angband beneath the Iron Mountains",
            ],
        },
        {
            "name": "The Last of Their Kind",
            "description": (
                "The protagonist encounters the sole survivor of a vanished people or "
                "order, who carries irreplaceable knowledge — and unbearable grief."
            ),
            "required_elements": ["survivor_figure", "lost_people_or_order"],
            "scene_type_hint": "personal_dialogue",
            "tolkien_examples": [
                "Treebeard as the last speaking tree",
                "A lone Noldor lord in a ruined city of the First Age",
            ],
        },
    ]

    def select_trope(self, chapter_beat: str = "", used_tropes: list[str] = None) -> dict:
        """
        Select an appropriate trope for the given chapter beat.
        Avoids repeating recently used tropes.
        """
        used = set(used_tropes or [])
        available = [t for t in self.TROPES if t["name"] not in used]

        if not available:
            # All tropes used — reset pool
            available = self.TROPES

        # Simple keyword match to prefer contextually appropriate tropes
        if chapter_beat:
            beat_lower = chapter_beat.lower()
            scored = []
            for trope in available:
                score = 0
                hint = trope.get("scene_type_hint", "")
                # Rough keyword matching
                if hint == "battle" and any(w in beat_lower for w in ["battle", "fight", "attack", "flee", "danger"]):
                    score += 2
                elif hint == "journey" and any(w in beat_lower for w in ["travel", "journey", "road", "path", "wilderness"]):
                    score += 2
                elif hint == "discovery" and any(w in beat_lower for w in ["find", "discover", "reveal", "ruin", "ancient"]):
                    score += 2
                elif hint == "council" and any(w in beat_lower for w in ["speak", "counsel", "meet", "parley", "gate"]):
                    score += 2
                scored.append((score, trope))

            scored.sort(key=lambda x: x[0], reverse=True)
            # Pick from top candidates with some randomness
            top_score = scored[0][0]
            top_candidates = [t for s, t in scored if s >= top_score]
            return random.choice(top_candidates)

        return random.choice(available)

    def get_all(self) -> list[dict]:
        return self.TROPES


# ─── Incubation result ───────────────────────────────────────────────────────

@dataclass
class IncubationResult:
    """Output of a single Lore Incubator run."""
    invented_entities: list[dict] = field(default_factory=list)
    narrative_seeds: list[str] = field(default_factory=list)
    trope_used: dict = field(default_factory=dict)
    raw_response: str = ""

    def summary(self) -> str:
        lines = [f"Trope: {self.trope_used.get('name', 'None')}"]
        for entity in self.invented_entities:
            lines.append(f"  [{entity.get('type')}] {entity.get('name')}: {entity.get('description', '')[:80]}")
        if self.narrative_seeds:
            lines.append("Seeds:")
            for seed in self.narrative_seeds:
                lines.append(f"  - {seed}")
        return "\n".join(lines)


# ─── Lore Incubator ──────────────────────────────────────────────────────────

_INVENTION_PROMPT = """You are a sub-creator working in J.R.R. Tolkien's legendarium.

A character is traveling through uncharted territory. Your task is NOT to write prose.
Your task is to INVENT new lore entities that will populate this whitespace.

JOURNEY CONTEXT:
{journey_context}

WHITESPACE DESCRIPTION:
{whitespace_description}

WORLD BIBLE CONSTRAINTS (all invented entities must comply):
{world_bible_rules}

TROPE SEED (incorporate this structural element):
Name: {trope_name}
Description: {trope_description}
Required elements: {trope_required}

EXISTING SHADOW CANON (do not duplicate these):
{existing_shadow_entities}

Invent EXACTLY these entity types:
1. One MINOR_CHARACTER (a living being — mortal, Elf, creature, or spirit)
2. One RUINED_LOCATION (a named place with tragic or mysterious history)
3. One ARTIFACT (a named object with First or Second Age provenance)

Naming conventions:
- Elven characters/places: Sindarin or Quenya roots (e.g., Cabed, Tol, Annon, Esgal, Dol, Bar)
- Mannish characters/places: Anglo-Saxon or Norse roots (e.g., Helm, Beorn, Aldburg, Wulf)
- Artifacts: evocative compound names (e.g., Mornbrand, Aeglos-minor, Caladring)

All entities must:
- Have a tragic or mysterious history consistent with the Age
- Connect thematically to the trope seed
- Never contradict the World Bible rules

Return ONLY valid JSON (no markdown fences, no preamble):
{{
  "invented_entities": [
    {{
      "type": "MINOR_CHARACTER",
      "name": "...",
      "race": "...",
      "description": "...",
      "tragic_history": "...",
      "role_in_story": "...",
      "trope_connection": "..."
    }},
    {{
      "type": "RUINED_LOCATION",
      "name": "...",
      "region": "...",
      "description": "...",
      "former_purpose": "...",
      "how_it_fell": "...",
      "what_remains": "..."
    }},
    {{
      "type": "ARTIFACT",
      "name": "...",
      "material": "...",
      "description": "...",
      "age_of_origin": "...",
      "tragic_history": "...",
      "power_or_property": "...",
      "trope_connection": "..."
    }}
  ],
  "narrative_seeds": [
    "One sentence: how the MINOR_CHARACTER enters the story",
    "One sentence: how the RUINED_LOCATION is discovered",
    "One sentence: how the ARTIFACT is found (tied to the trope seed)"
  ]
}}"""


class LoreIncubator:
    """
    Pre-drafting invention phase.

    Runs before the Outliner generates chapter beats. Invents new entities
    — minor characters, ruined locations, artifacts — and commits them to
    the Shadow Graph as local canon before the Drafter ever sees a prompt.

    The key insight: you give the LLM explicit *permission* to invent new
    things here. This is the only place in the pipeline where hallucination
    is the desired outcome.
    """

    def __init__(
        self,
        shadow_graph: ShadowGraph,
        world_bible=None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.shadow_graph = shadow_graph
        self.world_bible = world_bible
        self.llm = llm_client or LLMClient()
        self.tropes = TropeDictionary()
        self._used_tropes: list[str] = []

    def incubate(
        self,
        journey_context: str,
        whitespace_description: str,
        trope: dict = None,
        entity_types: list[str] = None,  # Future: override default 3-type set
    ) -> IncubationResult:
        """
        Run the invention phase for a chapter.

        Args:
            journey_context: e.g. "Tuor traveling from Nevrast to Gondolin"
            whitespace_description: e.g. "400-mile gap of unnamed wilderness between coastal cliffs and the Echoriath"
            trope: Override the auto-selected trope (from TropeDictionary)
            entity_types: Override the entity types to invent (default: CHARACTER, LOCATION, ARTIFACT)

        Returns:
            IncubationResult with invented entities and narrative seeds
        """
        selected_trope = trope or self.tropes.select_trope(
            chapter_beat=whitespace_description,
            used_tropes=self._used_tropes,
        )

        world_rules = self._get_world_bible_rules()
        existing_entities = self._get_existing_entity_names()

        prompt = _INVENTION_PROMPT.format(
            journey_context=journey_context,
            whitespace_description=whitespace_description,
            world_bible_rules=world_rules,
            trope_name=selected_trope["name"],
            trope_description=selected_trope["description"],
            trope_required=", ".join(selected_trope.get("required_elements", [])),
            existing_shadow_entities=existing_entities or "None yet.",
        )

        try:
            response = self.llm.generate(prompt, temperature=0.85)
            data = self._parse_response(response)

            result = IncubationResult(
                invented_entities=data.get("invented_entities", []),
                narrative_seeds=data.get("narrative_seeds", []),
                trope_used=selected_trope,
                raw_response=response,
            )

            # Track used tropes to avoid repeats
            self._used_tropes.append(selected_trope["name"])

            return result

        except Exception as e:
            print(f"[LoreIncubator] Warning: incubation failed: {e}")
            return IncubationResult(trope_used=selected_trope, raw_response="")

    def commit_to_shadow(self, result: IncubationResult, scene_id: str = "incubation") -> None:
        """
        Write all invented entities to the Shadow Graph.
        After this call, the Drafter treats them as immutable local canon.
        """
        for entity in result.invented_entities:
            self.shadow_graph.commit_invented_entity(entity, scene_id)

    def incubate_and_commit(
        self,
        journey_context: str,
        whitespace_description: str,
        chapter_id: str = "pre-draft",
    ) -> IncubationResult:
        """Convenience: incubate + commit in one call."""
        result = self.incubate(journey_context, whitespace_description)
        self.commit_to_shadow(result, scene_id=chapter_id)
        return result

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _get_world_bible_rules(self, max_rules: int = 15) -> str:
        if not self.world_bible:
            return "No world bible loaded — invent freely but keep within Tolkien's First Age conventions."

        rules = []
        for category, rule_list in self.world_bible.rules.items():
            for rule in rule_list[:3]:  # Sample from each category
                rules.append(f"[{category.value}] {rule.title}: {rule.description[:120]}")
                if len(rules) >= max_rules:
                    break
            if len(rules) >= max_rules:
                break

        return "\n".join(rules) if rules else "No rules found."

    def _get_existing_entity_names(self) -> str:
        entities = self.shadow_graph.get_invented_entities()
        if not entities:
            return ""
        return ", ".join(f"{e.name} ({e.type})" for e in entities[:20])

    def _parse_response(self, response: str) -> dict:
        """Extract JSON from LLM response, handling various formats."""
        # Strip markdown fences if present
        response = re.sub(r'```(?:json)?\s*', '', response).strip()

        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            raise ValueError("No JSON object found in incubator response")

        return json.loads(json_match.group())
