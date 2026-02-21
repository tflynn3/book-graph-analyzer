"""Data models for Shadow Graph story state."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InventedEntity:
    """A new lore entity created by the Incubator — local canon for this story."""
    type: str               # MINOR_CHARACTER, RUINED_LOCATION, ARTIFACT
    name: str
    description: str
    story_id: str
    scene_id: str           # Scene in which this entity was introduced
    properties: dict = field(default_factory=dict)  # type-specific fields

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "story_id": self.story_id,
            "scene_id": self.scene_id,
            "properties": self.properties,
        }


@dataclass
class CharacterState:
    """Current state of a character in the generated story."""
    name: str
    story_id: str
    location: Optional[str] = None
    possessions: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)  # e.g. ["weary", "injured_left_arm"]
    last_scene: str = ""

    def to_prompt_fragment(self) -> str:
        parts = [f"{self.name}"]
        if self.location:
            parts.append(f"located in {self.location}")
        if self.possessions:
            parts.append(f"carries: {', '.join(self.possessions)}")
        if self.conditions:
            parts.append(f"condition: {', '.join(self.conditions)}")
        return ". ".join(parts) + "."


@dataclass
class SceneState:
    """Full assembled current state fed to scene generation."""
    characters: list[CharacterState] = field(default_factory=list)
    recent_summaries: list[str] = field(default_factory=list)       # Last 3 scene summaries
    invented_entities: list[InventedEntity] = field(default_factory=list)  # Shadow-canon entities

    def to_prompt_block(self) -> str:
        """
        Compact, LLM-readable current-state block.
        Kept under ~400 tokens by design.
        """
        lines = []

        if self.characters:
            lines.append("CURRENT STATE:")
            for ch in self.characters:
                lines.append(f"  - {ch.to_prompt_fragment()}")

        if self.recent_summaries:
            lines.append("\nRECENT EVENTS (last scenes):")
            for summary in self.recent_summaries[-3:]:
                lines.append(f"  - {summary}")

        if self.invented_entities:
            lines.append("\nESTABLISHED (local canon for this story):")
            for entity in self.invented_entities:
                lines.append(f"  - [{entity.type}] {entity.name}: {entity.description[:100]}")

        return "\n".join(lines)


@dataclass
class StateDelta:
    """
    Structured state changes extracted from a generated scene.
    Written to the Shadow Graph after every scene.
    """
    story_id: str
    scene_id: str

    # {character_name: {location, possessions_gained, possessions_lost, conditions_added, conditions_removed}}
    character_updates: dict[str, dict] = field(default_factory=dict)

    # New entities introduced or discovered in the scene
    invented_entities: list[dict] = field(default_factory=list)

    # One-sentence summary of the scene (stored in Shadow_Scene node)
    scene_summary: str = ""
    chapter_num: int = 0
    scene_num: int = 0
