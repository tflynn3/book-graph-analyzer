"""Data models for World Bible extraction.

Defines the structure for storing world-building rules,
cultural profiles, magic systems, and other lore.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json
from pathlib import Path


class WorldBibleCategory(Enum):
    """Categories of world-building information."""
    MAGIC = "magic"
    CULTURE = "culture"
    GEOGRAPHY = "geography"
    TECHNOLOGY = "technology"
    COSMOLOGY = "cosmology"
    HISTORY = "history"
    LANGUAGE = "language"
    CREATURES = "creatures"
    OBJECTS = "objects"
    THEMES = "themes"


@dataclass
class SourcePassage:
    """A passage that supports a world bible entry."""
    text: str
    book: str
    location: str  # e.g., "Chapter 3, paragraph 12"
    relevance: float = 1.0  # How relevant is this passage to the rule?


@dataclass
class WorldRule:
    """A rule or pattern extracted from the world.
    
    Examples:
    - "Elves do not age and are immortal unless slain"
    - "The One Ring corrupts all who possess it"
    - "Only those with Elvish or Numenorean blood can wield certain swords"
    """
    id: str
    title: str
    description: str
    category: WorldBibleCategory
    
    # Evidence
    source_passages: list[SourcePassage] = field(default_factory=list)
    confidence: float = 1.0
    
    # Metadata
    keywords: list[str] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)
    
    # Constraints and exceptions
    constraints: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "source_passages": [
                {"text": p.text, "book": p.book, "location": p.location, "relevance": p.relevance}
                for p in self.source_passages
            ],
            "confidence": self.confidence,
            "keywords": self.keywords,
            "related_entities": self.related_entities,
            "constraints": self.constraints,
            "exceptions": self.exceptions,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "WorldRule":
        return cls(
            id=d["id"],
            title=d["title"],
            description=d["description"],
            category=WorldBibleCategory(d["category"]),
            source_passages=[
                SourcePassage(**p) for p in d.get("source_passages", [])
            ],
            confidence=d.get("confidence", 1.0),
            keywords=d.get("keywords", []),
            related_entities=d.get("related_entities", []),
            constraints=d.get("constraints", []),
            exceptions=d.get("exceptions", []),
        )


@dataclass
class CulturalProfile:
    """Profile of a culture, race, or people in the world."""
    id: str
    name: str  # e.g., "Hobbits", "Elves of Rivendell", "Dwarves of Erebor"
    
    # Cultural attributes
    values: list[str] = field(default_factory=list)  # What they value
    customs: list[str] = field(default_factory=list)  # Traditions and practices
    taboos: list[str] = field(default_factory=list)  # Things they avoid/forbid
    
    # Physical/practical
    homeland: Optional[str] = None
    government: Optional[str] = None
    language: Optional[str] = None
    lifespan: Optional[str] = None
    
    # Relationships
    allies: list[str] = field(default_factory=list)
    enemies: list[str] = field(default_factory=list)
    
    # Evidence
    source_passages: list[SourcePassage] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "values": self.values,
            "customs": self.customs,
            "taboos": self.taboos,
            "homeland": self.homeland,
            "government": self.government,
            "language": self.language,
            "lifespan": self.lifespan,
            "allies": self.allies,
            "enemies": self.enemies,
            "source_passages": [
                {"text": p.text, "book": p.book, "location": p.location}
                for p in self.source_passages
            ],
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "CulturalProfile":
        return cls(
            id=d["id"],
            name=d["name"],
            values=d.get("values", []),
            customs=d.get("customs", []),
            taboos=d.get("taboos", []),
            homeland=d.get("homeland"),
            government=d.get("government"),
            language=d.get("language"),
            lifespan=d.get("lifespan"),
            allies=d.get("allies", []),
            enemies=d.get("enemies", []),
            source_passages=[
                SourcePassage(**p) for p in d.get("source_passages", [])
            ],
        )


@dataclass
class MagicSystem:
    """Description of a magic or power system in the world."""
    id: str
    name: str  # e.g., "Ring Magic", "Elven Magic", "Wizardry"
    
    # How it works
    source: Optional[str] = None  # Where does the power come from?
    practitioners: list[str] = field(default_factory=list)  # Who can use it?
    abilities: list[str] = field(default_factory=list)  # What can it do?
    
    # Constraints
    costs: list[str] = field(default_factory=list)  # What does it cost to use?
    limitations: list[str] = field(default_factory=list)  # What can't it do?
    dangers: list[str] = field(default_factory=list)  # What are the risks?
    
    # Evidence
    source_passages: list[SourcePassage] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "practitioners": self.practitioners,
            "abilities": self.abilities,
            "costs": self.costs,
            "limitations": self.limitations,
            "dangers": self.dangers,
            "source_passages": [
                {"text": p.text, "book": p.book, "location": p.location}
                for p in self.source_passages
            ],
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "MagicSystem":
        return cls(
            id=d["id"],
            name=d["name"],
            source=d.get("source"),
            practitioners=d.get("practitioners", []),
            abilities=d.get("abilities", []),
            costs=d.get("costs", []),
            limitations=d.get("limitations", []),
            dangers=d.get("dangers", []),
            source_passages=[
                SourcePassage(**p) for p in d.get("source_passages", [])
            ],
        )


@dataclass
class GeographyEntry:
    """A location or geographical feature in the world."""
    id: str
    name: str
    
    # Description
    type: str = "location"  # location, region, building, landmark
    description: Optional[str] = None
    
    # Relationships
    parent_region: Optional[str] = None  # What larger area contains this?
    notable_features: list[str] = field(default_factory=list)
    inhabitants: list[str] = field(default_factory=list)
    
    # Travel
    travel_notes: list[str] = field(default_factory=list)  # How to get there, dangers, etc.
    distances: dict[str, str] = field(default_factory=dict)  # name -> distance description
    
    # Evidence
    source_passages: list[SourcePassage] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "parent_region": self.parent_region,
            "notable_features": self.notable_features,
            "inhabitants": self.inhabitants,
            "travel_notes": self.travel_notes,
            "distances": self.distances,
            "source_passages": [
                {"text": p.text, "book": p.book, "location": p.location}
                for p in self.source_passages
            ],
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "GeographyEntry":
        return cls(
            id=d["id"],
            name=d["name"],
            type=d.get("type", "location"),
            description=d.get("description"),
            parent_region=d.get("parent_region"),
            notable_features=d.get("notable_features", []),
            inhabitants=d.get("inhabitants", []),
            travel_notes=d.get("travel_notes", []),
            distances=d.get("distances", {}),
            source_passages=[
                SourcePassage(**p) for p in d.get("source_passages", [])
            ],
        )


@dataclass
class WorldBible:
    """Complete world bible for a fictional world."""
    name: str  # e.g., "Middle-earth"
    
    # Core content
    rules: dict[WorldBibleCategory, list[WorldRule]] = field(default_factory=dict)
    cultures: dict[str, CulturalProfile] = field(default_factory=dict)
    magic_systems: dict[str, MagicSystem] = field(default_factory=dict)
    geography: dict[str, GeographyEntry] = field(default_factory=dict)
    
    # Metadata
    sources: list[str] = field(default_factory=list)  # Books processed
    last_updated: Optional[str] = None
    
    def add_rule(self, rule: WorldRule) -> None:
        """Add a rule to the appropriate category."""
        if rule.category not in self.rules:
            self.rules[rule.category] = []
        self.rules[rule.category].append(rule)
    
    def get_rules(self, category: WorldBibleCategory) -> list[WorldRule]:
        """Get all rules in a category."""
        return self.rules.get(category, [])
    
    def search_rules(self, query: str) -> list[WorldRule]:
        """Search rules by keyword."""
        query_lower = query.lower()
        results = []
        for rules in self.rules.values():
            for rule in rules:
                if (query_lower in rule.title.lower() or 
                    query_lower in rule.description.lower() or
                    any(query_lower in kw.lower() for kw in rule.keywords)):
                    results.append(rule)
        return results
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rules": {
                cat.value: [r.to_dict() for r in rules]
                for cat, rules in self.rules.items()
            },
            "cultures": {k: v.to_dict() for k, v in self.cultures.items()},
            "magic_systems": {k: v.to_dict() for k, v in self.magic_systems.items()},
            "geography": {k: v.to_dict() for k, v in self.geography.items()},
            "sources": self.sources,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "WorldBible":
        bible = cls(
            name=d["name"],
            sources=d.get("sources", []),
            last_updated=d.get("last_updated"),
        )
        
        # Load rules
        for cat_str, rules_data in d.get("rules", {}).items():
            cat = WorldBibleCategory(cat_str)
            bible.rules[cat] = [WorldRule.from_dict(r) for r in rules_data]
        
        # Load cultures
        for k, v in d.get("cultures", {}).items():
            bible.cultures[k] = CulturalProfile.from_dict(v)
        
        # Load magic systems
        for k, v in d.get("magic_systems", {}).items():
            bible.magic_systems[k] = MagicSystem.from_dict(v)
        
        # Load geography
        for k, v in d.get("geography", {}).items():
            bible.geography[k] = GeographyEntry.from_dict(v)
        
        return bible
    
    def summary(self) -> str:
        """Generate a summary of the world bible contents."""
        total_rules = sum(len(rules) for rules in self.rules.values())
        
        lines = [
            f"=== World Bible: {self.name} ===",
            f"Sources: {', '.join(self.sources) if self.sources else 'None'}",
            f"",
            f"[Content Summary]",
            f"  Total rules: {total_rules}",
            f"  Cultures: {len(self.cultures)}",
            f"  Magic systems: {len(self.magic_systems)}",
            f"  Geography entries: {len(self.geography)}",
            f"",
        ]
        
        if self.rules:
            lines.append("[Rules by Category]")
            for cat, rules in sorted(self.rules.items(), key=lambda x: -len(x[1])):
                lines.append(f"  {cat.value}: {len(rules)}")
        
        return "\n".join(lines)
    
    def save(self, path: Path) -> None:
        """Save world bible to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "WorldBible":
        """Load world bible from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))
