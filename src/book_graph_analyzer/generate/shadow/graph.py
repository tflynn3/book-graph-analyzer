"""Shadow Graph — Neo4j-backed mutable story state for a generated novel.

Uses Shadow_* label namespace to stay completely separate from canonical lore.
All nodes are scoped to a story_id so multiple stories can run in parallel.
"""

import json
import re
from typing import Optional

from ...llm import LLMClient
from ...graph.connection import get_driver
from .models import CharacterState, InventedEntity, SceneState, StateDelta


# ─── Cypher helpers ──────────────────────────────────────────────────────────

_MERGE_CHARACTER = """
MERGE (c:Shadow_Character {name: $name, story_id: $story_id})
RETURN c
"""

_SET_LOCATION = """
MATCH (c:Shadow_Character {name: $name, story_id: $story_id})
OPTIONAL MATCH (c)-[existing:LOCATED_AT]->(:Shadow_Place)
WITH c, collect(existing) AS existing_locations
FOREACH (relation IN existing_locations | DELETE relation)
MERGE (p:Shadow_Place {name: $location, story_id: $story_id})
MERGE (c)-[:LOCATED_AT]->(p)
"""

_ADD_POSSESSION = """
MATCH (c:Shadow_Character {name: $name, story_id: $story_id})
MERGE (o:Shadow_Object {name: $item, story_id: $story_id})
MERGE (c)-[:POSSESSES {acquired_in_scene: $scene_id}]->(o)
"""

_REMOVE_POSSESSION = """
MATCH (c:Shadow_Character {name: $name, story_id: $story_id})-[r:POSSESSES]->(o:Shadow_Object {name: $item, story_id: $story_id})
DELETE r
"""

_ADD_CONDITION = """
MATCH (c:Shadow_Character {name: $name, story_id: $story_id})
MERGE (cond:Shadow_Condition {value: $condition, story_id: $story_id})
MERGE (c)-[:HAS_CONDITION {last_updated_scene: $scene_id}]->(cond)
"""

_REMOVE_CONDITION = """
MATCH (c:Shadow_Character {name: $name, story_id: $story_id})-[r:HAS_CONDITION]->(cond:Shadow_Condition {value: $condition, story_id: $story_id})
DELETE r
"""

_RECORD_SCENE = """
MERGE (s:Shadow_Scene {id: $scene_id, story_id: $story_id})
SET s.summary = $summary, s.chapter_num = $chapter_num, s.scene_num = $scene_num
WITH s
OPTIONAL MATCH (p:Shadow_Place {name: $place, story_id: $story_id})
FOREACH (x IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
    MERGE (s)-[:OCCURRED_AT]->(p)
)
"""

_LINK_CHARACTER_SCENE = """
MATCH (c:Shadow_Character {name: $name, story_id: $story_id})
MATCH (s:Shadow_Scene {id: $scene_id, story_id: $story_id})
MERGE (s)-[:INVOLVED]->(c)
"""

_MERGE_INVENTED_ENTITY = """
MERGE (e:Shadow_Entity {name: $name, story_id: $story_id})
SET e.type = $type,
    e.description = $description,
    e.scene_id = $scene_id,
    e.properties = $properties
"""

_GET_CHARACTER = """
MATCH (c:Shadow_Character {name: $name, story_id: $story_id})
OPTIONAL MATCH (c)-[:LOCATED_AT]->(p:Shadow_Place)
OPTIONAL MATCH (c)-[:POSSESSES]->(o:Shadow_Object)
OPTIONAL MATCH (c)-[:HAS_CONDITION]->(cond:Shadow_Condition)
RETURN c.name as name,
       p.name as location,
       collect(DISTINCT o.name) as possessions,
       collect(DISTINCT cond.value) as conditions
"""

_GET_RECENT_SUMMARIES = """
MATCH (s:Shadow_Scene {story_id: $story_id})
WHERE s.summary IS NOT NULL AND s.summary <> ""
  AND (
    $chapter_num IS NULL
    OR coalesce(s.chapter_num, 0) < $chapter_num
    OR (
      coalesce(s.chapter_num, 0) = $chapter_num
      AND ($scene_num IS NULL OR coalesce(s.scene_num, 0) < $scene_num)
    )
  )
RETURN s.summary AS summary, s.chapter_num AS chapter_num, s.scene_num AS scene_num
ORDER BY coalesce(s.chapter_num, 0) DESC,
         coalesce(s.scene_num, 0) DESC,
         s.id DESC
LIMIT $limit
"""

_GET_INVENTED_ENTITIES = """
MATCH (e:Shadow_Entity {story_id: $story_id})
WHERE $type IS NULL OR e.type = $type
RETURN e.name as name, e.type as type, e.description as description,
       e.scene_id as scene_id, e.properties as properties
"""

_DELETE_STORY = """
MATCH (n)
WHERE n.story_id = $story_id
  AND any(label IN labels(n) WHERE label STARTS WITH 'Shadow_')
DETACH DELETE n
"""


# ─── State delta extraction prompt ───────────────────────────────────────────

_EXTRACT_DELTA_PROMPT = """Extract structured state changes from this scene.

SCENE TEXT:
\"\"\"{scene_text}\"\"\"

CHARACTERS IN SCENE: {characters}

For each character, identify:
- location_change: where they are now (null if unchanged / unclear)
- possessions_gained: items they picked up, received, or now carry
- possessions_lost: items they dropped, gave away, or lost
- conditions_added: new physical/mental states (weary, injured, afraid, resolute, etc.)
- conditions_removed: states that have resolved

Also identify any NEW named entities introduced in the scene:
- Named characters (minor)
- Named locations
- Named artifacts or objects

Return ONLY valid JSON:
{{
  "characters": {{
    "<name>": {{
      "location_change": "<place name or null>",
      "possessions_gained": ["<item>"],
      "possessions_lost": ["<item>"],
      "conditions_added": ["<condition>"],
      "conditions_removed": ["<condition>"]
    }}
  }},
  "new_entities": [
    {{
      "type": "MINOR_CHARACTER | RUINED_LOCATION | ARTIFACT | PLACE",
      "name": "<name>",
      "description": "<one sentence>"
    }}
  ],
  "scene_summary": "<one sentence summary of what happened>"
}}"""


class ShadowGraph:
    """
    Mutable story-state graph stored in Neo4j under Shadow_* labels.

    Tracks what is TRUE in the GENERATED novel — separate from Tolkien canon.
    All nodes are scoped to a story_id for clean multi-story isolation.
    """

    def __init__(
        self,
        story_id: str,
        driver=None,
        delta_max_chars: int | None = None,
    ):
        self.story_id = story_id
        self._driver = driver or get_driver()
        self._llm = LLMClient()
        self._delta_max_chars = delta_max_chars

    # ─── Write operations ────────────────────────────────────────────────────

    def commit_state_delta(self, delta: StateDelta) -> None:
        """Write a StateDelta to Neo4j. Non-blocking on failure."""
        if not self._driver:
            return

        try:
            with self._driver.session() as session:
                for char_name, updates in delta.character_updates.items():
                    # Ensure character node exists
                    session.run(_MERGE_CHARACTER, name=char_name, story_id=self.story_id)

                    if updates.get("location_change"):
                        session.run(_SET_LOCATION, name=char_name,
                                    story_id=self.story_id,
                                    location=updates["location_change"])

                    for item in updates.get("possessions_gained", []):
                        session.run(_ADD_POSSESSION, name=char_name,
                                    story_id=self.story_id,
                                    item=item, scene_id=delta.scene_id)

                    for item in updates.get("possessions_lost", []):
                        session.run(_REMOVE_POSSESSION, name=char_name,
                                    story_id=self.story_id, item=item)

                    for cond in updates.get("conditions_added", []):
                        session.run(_ADD_CONDITION, name=char_name,
                                    story_id=self.story_id,
                                    condition=cond, scene_id=delta.scene_id)

                    for cond in updates.get("conditions_removed", []):
                        session.run(_REMOVE_CONDITION, name=char_name,
                                    story_id=self.story_id, condition=cond)

                # Record the scene node
                primary_char = next(iter(delta.character_updates), None)
                primary_location = None
                if primary_char:
                    primary_location = delta.character_updates[primary_char].get("location_change")

                session.run(_RECORD_SCENE,
                            scene_id=delta.scene_id,
                            story_id=self.story_id,
                            summary=delta.scene_summary,
                            chapter_num=delta.chapter_num,
                            scene_num=delta.scene_num,
                            place=primary_location or "")

                for char_name in delta.character_updates:
                    session.run(_LINK_CHARACTER_SCENE,
                                name=char_name, story_id=self.story_id,
                                scene_id=delta.scene_id)

                # Commit any invented entities in this delta
                for entity in delta.invented_entities:
                    self.commit_invented_entity(entity, delta.scene_id)

        except Exception as e:
            # Non-blocking: log and continue — never crash scene generation
            print(f"[ShadowGraph] Warning: state commit failed for scene {delta.scene_id}: {e}")

    def commit_invented_entity(self, entity: dict, scene_id: str) -> None:
        """
        Write an invented entity to the Shadow Graph as local canon.
        Once committed, the Drafter treats it as immutable fact.
        """
        if not self._driver:
            return

        try:
            properties = {k: v for k, v in entity.items()
                          if k not in ("type", "name", "description")}
            with self._driver.session() as session:
                session.run(
                    _MERGE_INVENTED_ENTITY,
                    name=entity.get("name", "Unknown"),
                    story_id=self.story_id,
                    type=entity.get("type", "UNKNOWN"),
                    description=entity.get("description", ""),
                    scene_id=scene_id,
                    properties=json.dumps(properties),
                )
        except Exception as e:
            print(f"[ShadowGraph] Warning: entity commit failed: {e}")

    # ─── Read operations ─────────────────────────────────────────────────────

    def get_character_state(self, character_name: str) -> Optional[CharacterState]:
        """Return current location, possessions, and conditions for a character."""
        if not self._driver:
            return None

        try:
            with self._driver.session() as session:
                result = session.run(_GET_CHARACTER,
                                     name=character_name, story_id=self.story_id)
                record = result.single()
                if not record:
                    return None

                return CharacterState(
                    name=record["name"],
                    story_id=self.story_id,
                    location=record["location"],
                    possessions=[p for p in record["possessions"] if p],
                    conditions=[c for c in record["conditions"] if c],
                )
        except Exception as e:
            print(f"[ShadowGraph] Warning: character query failed: {e}")
            return None

    def get_scene_state(
        self,
        characters: list[str],
        place: str,
        chapter_num: int | None = None,
        scene_num: int | None = None,
    ) -> SceneState:
        """
        Assemble a full SceneState for the context assembler.
        Queries character states, recent summaries, and invented entities.
        """
        char_states = []
        for name in characters:
            state = self.get_character_state(name)
            if state:
                char_states.append(state)
            else:
                # Character not yet in Shadow Graph — return a blank state
                char_states.append(CharacterState(
                    name=name,
                    story_id=self.story_id,
                ))

        summaries = self._get_recent_summaries(
            chapter_num=chapter_num,
            scene_num=scene_num,
        )
        invented = self.get_invented_entities()

        return SceneState(
            characters=char_states,
            recent_summaries=summaries,
            invented_entities=invented,
        )

    def get_invented_entities(self, entity_type: str = None) -> list[InventedEntity]:
        """Return all shadow-canon entities, optionally filtered by type."""
        if not self._driver:
            return []

        try:
            with self._driver.session() as session:
                result = session.run(_GET_INVENTED_ENTITIES,
                                     story_id=self.story_id, type=entity_type)
                entities = []
                for record in result:
                    props = {}
                    if record["properties"]:
                        try:
                            props = json.loads(record["properties"])
                        except (json.JSONDecodeError, TypeError):
                            pass
                    entities.append(InventedEntity(
                        type=record["type"],
                        name=record["name"],
                        description=record["description"],
                        story_id=self.story_id,
                        scene_id=record["scene_id"] or "",
                        properties=props,
                    ))
                return entities
        except Exception as e:
            print(f"[ShadowGraph] Warning: entity query failed: {e}")
            return []

    def _get_recent_summaries(
        self,
        limit: int = 3,
        chapter_num: int | None = None,
        scene_num: int | None = None,
    ) -> list[str]:
        if not self._driver:
            return []

        try:
            with self._driver.session() as session:
                result = session.run(
                    _GET_RECENT_SUMMARIES,
                    story_id=self.story_id,
                    chapter_num=chapter_num,
                    scene_num=scene_num,
                    limit=max(0, limit),
                )
                # Neo4j selects newest-first so LIMIT keeps the truly latest scenes;
                # present that small window in chronological order to the drafter.
                summaries = [record["summary"] for record in result if record["summary"]]
                summaries.reverse()
                return summaries
        except Exception as e:
            print(f"[ShadowGraph] Warning: summary query failed: {e}")
            return []

    # ─── Extraction ──────────────────────────────────────────────────────────

    def extract_delta_from_scene(
        self,
        scene_text: str,
        characters: list[str],
        scene_id: str,
        chapter_num: int = 0,
        scene_num: int = 0,
    ) -> StateDelta:
        """
        Ask the LLM to extract a StateDelta from a generated scene.
        Falls back gracefully to an empty delta on parse failure.
        """
        extraction_text = scene_text
        if self._delta_max_chars is not None:
            extraction_text = scene_text[: max(0, self._delta_max_chars)]

        prompt = _EXTRACT_DELTA_PROMPT.format(
            scene_text=extraction_text,
            characters=", ".join(characters),
        )

        try:
            response = self._llm.generate(prompt, temperature=0.1)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                raise ValueError("No JSON in response")

            data = json.loads(json_match.group())

            character_updates = {}
            for char_name, updates in data.get("characters", {}).items():
                character_updates[char_name] = {
                    "location_change": updates.get("location_change"),
                    "possessions_gained": updates.get("possessions_gained", []),
                    "possessions_lost": updates.get("possessions_lost", []),
                    "conditions_added": updates.get("conditions_added", []),
                    "conditions_removed": updates.get("conditions_removed", []),
                }

            return StateDelta(
                story_id=self.story_id,
                scene_id=scene_id,
                character_updates=character_updates,
                invented_entities=data.get("new_entities", []),
                scene_summary=data.get("scene_summary", ""),
                chapter_num=chapter_num,
                scene_num=scene_num,
            )

        except Exception as e:
            print(f"[ShadowGraph] Warning: delta extraction failed, using empty delta: {e}")
            return StateDelta(
                story_id=self.story_id,
                scene_id=scene_id,
                chapter_num=chapter_num,
                scene_num=scene_num,
            )

    # ─── Maintenance ─────────────────────────────────────────────────────────

    def reset_story(self) -> None:
        """Delete all Shadow_* nodes for this story_id. Irreversible."""
        if not self._driver:
            return

        try:
            with self._driver.session() as session:
                session.run(_DELETE_STORY, story_id=self.story_id)
            print(f"[ShadowGraph] Story {self.story_id} state cleared.")
        except Exception as e:
            print(f"[ShadowGraph] Warning: reset failed: {e}")
