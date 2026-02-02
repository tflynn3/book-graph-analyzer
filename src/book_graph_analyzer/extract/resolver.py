"""Entity resolution - mapping extracted names to canonical entities.

Handles aliases, titles, and fuzzy matching to identify the canonical
entity that an extracted mention refers to.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rapidfuzz import fuzz, process

from ..config import get_settings
from ..models.entities import Character, Place, Object
from .ner import ExtractedEntity


@dataclass
class ResolvedEntity:
    """An extracted entity resolved to a canonical entry (or marked as new)."""

    extracted: ExtractedEntity
    canonical_id: str | None = None
    canonical_name: str | None = None
    entity_type: Literal["character", "place", "object", "event", "unknown"] = "unknown"
    confidence: float = 0.0
    is_new: bool = False  # True if not found in seed database


@dataclass
class EntityDatabase:
    """In-memory database of known entities for resolution."""

    characters: dict[str, Character] = field(default_factory=dict)
    places: dict[str, Place] = field(default_factory=dict)
    objects: dict[str, Object] = field(default_factory=dict)

    # Lookup tables for fast alias resolution
    _alias_to_id: dict[str, str] = field(default_factory=dict)
    _name_to_type: dict[str, str] = field(default_factory=dict)

    def add_character(self, char: Character) -> None:
        """Add a character to the database."""
        self.characters[char.id] = char
        self._index_entity(char, "character")

    def add_place(self, place: Place) -> None:
        """Add a place to the database."""
        self.places[place.id] = place
        self._index_entity(place, "place")

    def add_object(self, obj: Object) -> None:
        """Add an object to the database."""
        self.objects[obj.id] = obj
        self._index_entity(obj, "object")

    def _index_entity(self, entity, entity_type: str) -> None:
        """Index an entity's name and aliases for lookup."""
        # Index canonical name
        name_lower = entity.canonical_name.lower()
        self._alias_to_id[name_lower] = entity.id
        self._name_to_type[name_lower] = entity_type

        # Index aliases
        for alias in entity.aliases:
            alias_lower = alias.lower()
            self._alias_to_id[alias_lower] = entity.id
            self._name_to_type[alias_lower] = entity_type

    def lookup(self, text: str) -> tuple[str | None, str | None, float]:
        """Look up an entity by name or alias.

        Returns:
            Tuple of (entity_id, entity_type, confidence)
        """
        text_lower = text.lower().strip()

        # Exact match
        if text_lower in self._alias_to_id:
            return self._alias_to_id[text_lower], self._name_to_type[text_lower], 1.0

        # Try without common articles
        for prefix in ["the ", "a ", "an "]:
            if text_lower.startswith(prefix):
                stripped = text_lower[len(prefix) :]
                if stripped in self._alias_to_id:
                    return self._alias_to_id[stripped], self._name_to_type[stripped], 0.95

        # Fuzzy match
        if self._alias_to_id:
            result = process.extractOne(
                text_lower,
                self._alias_to_id.keys(),
                scorer=fuzz.ratio,
            )
            if result and result[1] >= 85:  # 85% similarity threshold
                matched_name = result[0]
                return (
                    self._alias_to_id[matched_name],
                    self._name_to_type[matched_name],
                    result[1] / 100,
                )

        return None, None, 0.0

    def get_entity(self, entity_id: str) -> Character | Place | Object | None:
        """Get an entity by ID."""
        if entity_id in self.characters:
            return self.characters[entity_id]
        if entity_id in self.places:
            return self.places[entity_id]
        if entity_id in self.objects:
            return self.objects[entity_id]
        return None


class EntityResolver:
    """Resolves extracted entities to canonical database entries."""

    def __init__(self, seed_dir: Path | None = None):
        """Initialize the resolver with optional seed directory.

        Args:
            seed_dir: Directory containing seed JSON files (characters.json, etc.)
        """
        self.settings = get_settings()
        self.seed_dir = seed_dir or self.settings.seeds_dir
        self.db = EntityDatabase()
        self._load_seeds()

    def _load_seeds(self) -> None:
        """Load seed data from JSON files."""
        if not self.seed_dir.exists():
            return

        # Load characters
        chars_file = self.seed_dir / "characters.json"
        if chars_file.exists():
            with open(chars_file) as f:
                data = json.load(f)
                for item in data:
                    char = Character(**item)
                    self.db.add_character(char)

        # Load places
        places_file = self.seed_dir / "places.json"
        if places_file.exists():
            with open(places_file) as f:
                data = json.load(f)
                for item in data:
                    place = Place(**item)
                    self.db.add_place(place)

        # Load objects
        objects_file = self.seed_dir / "objects.json"
        if objects_file.exists():
            with open(objects_file) as f:
                data = json.load(f)
                for item in data:
                    obj = Object(**item)
                    self.db.add_object(obj)

    def resolve(self, entity: ExtractedEntity) -> ResolvedEntity:
        """Resolve an extracted entity to a canonical entry.

        Args:
            entity: The extracted entity to resolve

        Returns:
            ResolvedEntity with canonical information if found
        """
        # Clean the extracted text
        text = self._clean_text(entity.text)

        # Look up in database
        entity_id, entity_type, confidence = self.db.lookup(text)

        if entity_id:
            canonical = self.db.get_entity(entity_id)
            return ResolvedEntity(
                extracted=entity,
                canonical_id=entity_id,
                canonical_name=canonical.canonical_name if canonical else None,
                entity_type=entity_type,
                confidence=confidence,
                is_new=False,
            )

        # Not found - create a new entity suggestion
        suggested_type = self._infer_type(entity)
        return ResolvedEntity(
            extracted=entity,
            canonical_id=None,
            canonical_name=None,
            entity_type=suggested_type,
            confidence=0.0,
            is_new=True,
        )

    def resolve_all(self, entities: list[ExtractedEntity]) -> list[ResolvedEntity]:
        """Resolve a list of extracted entities.

        Args:
            entities: List of extracted entities

        Returns:
            List of resolved entities
        """
        return [self.resolve(e) for e in entities]

    def _clean_text(self, text: str) -> str:
        """Clean extracted text for matching."""
        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove possessive suffixes
        text = re.sub(r"'s$", "", text)

        return text.strip()

    def _infer_type(self, entity: ExtractedEntity) -> str:
        """Infer entity type from the extraction label and text patterns."""
        # Map extraction labels to our types
        label_map = {
            "PERSON": "character",
            "PLACE": "place",
            "OBJECT": "object",
            "EVENT": "event",
            "ORG": "character",  # Organizations often map to peoples/races
        }

        if entity.label in label_map:
            return label_map[entity.label]

        # Pattern-based inference
        text_lower = entity.text.lower()

        # Place indicators
        place_indicators = [
            "mountain", "forest", "river", "lake", "sea", "land",
            "realm", "kingdom", "city", "tower", "hall", "gate",
            "pass", "valley", "plain", "wood", "dale", "shire",
        ]
        if any(ind in text_lower for ind in place_indicators):
            return "place"

        # Object indicators
        object_indicators = [
            "ring", "sword", "staff", "jewel", "stone", "gem",
            "crown", "armor", "helm", "blade", "bow", "arrow",
        ]
        if any(ind in text_lower for ind in object_indicators):
            return "object"

        return "unknown"

    def add_entity(
        self,
        canonical_name: str,
        entity_type: str,
        aliases: list[str] | None = None,
        **kwargs,
    ) -> str:
        """Add a new entity to the database.

        Args:
            canonical_name: The primary name for the entity
            entity_type: One of 'character', 'place', 'object'
            aliases: List of alternative names
            **kwargs: Additional entity attributes

        Returns:
            The generated entity ID
        """
        # Generate ID from canonical name
        entity_id = canonical_name.lower().replace(" ", "_").replace("'", "")
        entity_id = re.sub(r"[^a-z0-9_]", "", entity_id)

        aliases = aliases or []

        if entity_type == "character":
            char = Character(
                id=entity_id,
                canonical_name=canonical_name,
                aliases=aliases,
                **kwargs,
            )
            self.db.add_character(char)
        elif entity_type == "place":
            place = Place(
                id=entity_id,
                canonical_name=canonical_name,
                aliases=aliases,
                **kwargs,
            )
            self.db.add_place(place)
        elif entity_type == "object":
            obj = Object(
                id=entity_id,
                canonical_name=canonical_name,
                aliases=aliases,
                **kwargs,
            )
            self.db.add_object(obj)

        return entity_id

    def export_seeds(self, output_dir: Path | None = None) -> None:
        """Export the current database to seed files.

        Args:
            output_dir: Directory to write seed files to
        """
        output_dir = output_dir or self.seed_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export characters
        if self.db.characters:
            with open(output_dir / "characters.json", "w") as f:
                json.dump(
                    [c.model_dump() for c in self.db.characters.values()],
                    f,
                    indent=2,
                )

        # Export places
        if self.db.places:
            with open(output_dir / "places.json", "w") as f:
                json.dump(
                    [p.model_dump() for p in self.db.places.values()],
                    f,
                    indent=2,
                )

        # Export objects
        if self.db.objects:
            with open(output_dir / "objects.json", "w") as f:
                json.dump(
                    [o.model_dump() for o in self.db.objects.values()],
                    f,
                    indent=2,
                )

    @property
    def stats(self) -> dict:
        """Get statistics about the entity database."""
        return {
            "characters": len(self.db.characters),
            "places": len(self.db.places),
            "objects": len(self.db.objects),
            "total_aliases": len(self.db._alias_to_id),
        }
