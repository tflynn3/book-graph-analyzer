"""Context assembly engine for scene generation."""

from dataclasses import dataclass, field
from typing import Optional

from .shadow.graph import ShadowGraph
from .shadow.models import CharacterState


@dataclass
class AssembledContext:
    """Structured current-state context used by the scene generator."""

    character_states: list[CharacterState] = field(default_factory=list)
    recent_summaries: list[str] = field(default_factory=list)
    place_facts: dict = field(default_factory=dict)
    active_plot_threads: list[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        """Serialize to a compact, LLM-readable context block."""
        def _clip(text: str, max_len: int = 110) -> str:
            if len(text) <= max_len:
                return text
            return text[: max_len - 3] + "..."

        lines: list[str] = ["CURRENT STATE:"]

        if self.character_states:
            for character in self.character_states:
                lines.append(f"- {_clip(character.to_prompt_fragment(), 100)}")
        else:
            lines.append("- Character state not yet established.")

        if self.recent_summaries:
            lines.append("\nRECENT EVENTS (last 3 scenes):")
            for summary in self.recent_summaries[:3]:
                lines.append(f"- {_clip(summary, 95)}")

        place_name = self.place_facts.get("name") or "Unknown"
        place_region = self.place_facts.get("region")
        place_header = f"\nCURRENT PLACE — {place_name}"
        if place_region:
            place_header += f" ({place_region})"
        place_header += ":"
        lines.append(place_header)

        description = self.place_facts.get("description")
        if description:
            lines.append(f"- {_clip(description, 95)}")

        key_facts = self.place_facts.get("facts") or []
        for fact in key_facts[:3]:
            lines.append(f"- {_clip(str(fact), 90)}")

        if self.active_plot_threads:
            lines.append("\nACTIVE PLOT THREADS:")
            for thread in self.active_plot_threads[:4]:
                lines.append(f"- {_clip(thread, 90)}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "character_states": [
                {
                    "name": s.name,
                    "location": s.location,
                    "possessions": s.possessions,
                    "conditions": s.conditions,
                }
                for s in self.character_states
            ],
            "recent_summaries": self.recent_summaries,
            "place_facts": self.place_facts,
            "active_plot_threads": self.active_plot_threads,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AssembledContext":
        return cls(
            character_states=[],
            recent_summaries=list(data.get("recent_summaries", [])),
            place_facts=dict(data.get("place_facts", {})),
            active_plot_threads=list(data.get("active_plot_threads", [])),
        )


class ContextAssembler:
    """Assembles a structured context block from Shadow Graph + Neo4j."""

    def __init__(self, shadow_graph: Optional[ShadowGraph], neo4j_driver=None):
        self.shadow_graph = shadow_graph
        self.driver = neo4j_driver

    def assemble(
        self,
        story_id: str,
        characters: list[str],
        place: str,
        chapter_num: int,
        scene_num: int,
    ) -> AssembledContext:
        _ = scene_num  # reserved for future ranking and retrieval

        shadow = self._resolve_shadow_graph(story_id)
        scene_state = shadow.get_scene_state(characters=characters, place=place) if shadow else None

        return AssembledContext(
            character_states=(scene_state.characters if scene_state else []),
            recent_summaries=(scene_state.recent_summaries if scene_state else []),
            place_facts=self._get_place_facts(place),
            active_plot_threads=self._get_active_plot_threads(story_id, chapter_num),
        )

    def _resolve_shadow_graph(self, story_id: str) -> Optional[ShadowGraph]:
        if self.shadow_graph and self.shadow_graph.story_id == story_id:
            return self.shadow_graph

        if not self.driver:
            return None

        return ShadowGraph(story_id=story_id, driver=self.driver)

    def _get_place_facts(self, place: str) -> dict:
        if not self.driver or not place:
            return {}

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Place)
                WHERE toLower(p.name) CONTAINS toLower($name)
                RETURN p.name as name,
                       p.description as description,
                       p.region as region,
                       p.type as type,
                       p.history as history
                LIMIT 1
                """,
                name=place,
            )
            record = result.single()

        if not record:
            return {"name": place}

        facts = []
        if record.get("type"):
            facts.append(f"Type: {record['type']}")
        if record.get("history"):
            facts.append(str(record["history"])[:180])

        return {
            "name": record.get("name") or place,
            "description": (record.get("description") or "")[:220],
            "region": record.get("region"),
            "facts": facts,
        }

    def _get_active_plot_threads(self, story_id: str, chapter_num: int) -> list[str]:
        if not self.driver:
            return []

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Story {id: $story_id})
                OPTIONAL MATCH (s)-[:CONTAINS]->(c:Chapter)
                WHERE c.number <= $chapter_num
                RETURN s.outline as story_outline, c.number as chapter_num, c.outline as chapter_outline
                ORDER BY c.number DESC
                LIMIT 2
                """,
                story_id=story_id,
                chapter_num=chapter_num,
            )
            rows = list(result)

        if not rows:
            return []

        chunks = []
        story_outline = rows[0].get("story_outline")
        if story_outline:
            chunks.append(story_outline)

        for row in rows:
            chapter_outline = row.get("chapter_outline")
            if chapter_outline:
                chunks.append(chapter_outline)

        return self._extract_threads(chunks)

    @staticmethod
    def _extract_threads(chunks: list[str]) -> list[str]:
        threads: list[str] = []
        for chunk in chunks:
            for raw in chunk.splitlines():
                line = raw.strip().lstrip("-*0123456789. ").strip()
                if not line:
                    continue
                if len(line) > 140:
                    line = line[:137] + "..."
                threads.append(line)

        seen = set()
        deduped = []
        for thread in threads:
            key = thread.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(thread)
            if len(deduped) >= 4:
                break

        return deduped
