"""
World Bible Data Models

Data structures for world rules, cultural profiles, and the complete world bible.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json

from .categories import WorldBibleCategory


@dataclass
class SourcePassage:
    """A passage that supports a world rule."""
    passage_id: str
    text: str
    book: str
    relevance_score: float = 1.0  # How relevant this passage is to the rule


@dataclass
class WorldRule:
    """
    A rule or fact about the fictional world.
    
    Rules are extracted from text and linked to their sources.
    """
    id: str
    category: WorldBibleCategory
    title: str                      # Short description
    description: str                # Full explanation
    
    # Source tracking
    source_passages: list[SourcePassage] = field(default_factory=list)
    
    # Confidence and validation
    confidence: float = 0.5         # 0-1, how confident we are
    validated: bool = False         # Human-validated?
    validator_notes: str = ""       # Notes from validation
    
    # Relationships
    related_entities: list[str] = field(default_factory=list)  # Entity IDs
    contradicts: list[str] = field(default_factory=list)       # Other rule IDs that conflict
    supports: list[str] = field(default_factory=list)          # Other rule IDs this supports
    
    # Metadata
    extraction_method: str = "llm"  # llm, pattern, manual
    
    def add_source(self, passage_id: str, text: str, book: str, relevance: float = 1.0):
        """Add a source passage for this rule."""
        self.source_passages.append(SourcePassage(
            passage_id=passage_id,
            text=text[:500],  # Truncate for storage
            book=book,
            relevance_score=relevance,
        ))
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['category'] = self.category.value
        d['source_passages'] = [asdict(sp) for sp in self.source_passages]
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> "WorldRule":
        d['category'] = WorldBibleCategory(d['category'])
        d['source_passages'] = [SourcePassage(**sp) for sp in d.get('source_passages', [])]
        return cls(**d)


@dataclass
class CulturalProfile:
    """
    Profile of a culture, race, or people in the world.
    """
    id: str
    name: str                       # e.g., "Elves", "Dwarves", "Hobbits"
    
    # Core traits
    values: list[str] = field(default_factory=list)         # What they value
    customs: list[str] = field(default_factory=list)        # Traditions, rituals
    taboos: list[str] = field(default_factory=list)         # What's forbidden
    
    # Physical/material
    homeland: Optional[str] = None
    appearance: Optional[str] = None
    lifespan: Optional[str] = None
    
    # Social
    government: Optional[str] = None
    social_structure: Optional[str] = None
    relations: dict[str, str] = field(default_factory=dict)  # other_culture -> relationship
    
    # Language
    language: Optional[str] = None
    naming_conventions: Optional[str] = None
    
    # Source tracking
    source_passages: list[SourcePassage] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['source_passages'] = [asdict(sp) for sp in self.source_passages]
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> "CulturalProfile":
        d['source_passages'] = [SourcePassage(**sp) for sp in d.get('source_passages', [])]
        return cls(**d)


@dataclass
class WorldBible:
    """
    Complete world bible for a fictional world.
    
    Contains all extracted rules, cultures, and world-building information.
    """
    name: str                       # e.g., "Middle-earth", "Westeros"
    source_books: list[str] = field(default_factory=list)
    
    # Rules by category
    rules: dict[str, list[WorldRule]] = field(default_factory=dict)
    
    # Cultural profiles
    cultures: dict[str, CulturalProfile] = field(default_factory=dict)
    
    # Summary information
    magic_system_summary: str = ""
    geography_summary: str = ""
    history_summary: str = ""
    cosmology_summary: str = ""
    
    def add_rule(self, rule: WorldRule) -> None:
        """Add a rule to the world bible."""
        category = rule.category.value
        if category not in self.rules:
            self.rules[category] = []
        self.rules[category].append(rule)
    
    def get_rules(self, category: WorldBibleCategory) -> list[WorldRule]:
        """Get all rules for a category."""
        return self.rules.get(category.value, [])
    
    def add_culture(self, culture: CulturalProfile) -> None:
        """Add a cultural profile."""
        self.cultures[culture.id] = culture
    
    def get_culture(self, culture_id: str) -> Optional[CulturalProfile]:
        """Get a cultural profile by ID."""
        return self.cultures.get(culture_id)
    
    def total_rules(self) -> int:
        """Get total rule count."""
        return sum(len(rules) for rules in self.rules.values())
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source_books": self.source_books,
            "rules": {
                cat: [r.to_dict() for r in rules]
                for cat, rules in self.rules.items()
            },
            "cultures": {
                cid: c.to_dict() for cid, c in self.cultures.items()
            },
            "magic_system_summary": self.magic_system_summary,
            "geography_summary": self.geography_summary,
            "history_summary": self.history_summary,
            "cosmology_summary": self.cosmology_summary,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, d: dict) -> "WorldBible":
        wb = cls(name=d["name"], source_books=d.get("source_books", []))
        
        for cat, rules in d.get("rules", {}).items():
            wb.rules[cat] = [WorldRule.from_dict(r) for r in rules]
        
        for cid, cdata in d.get("cultures", {}).items():
            wb.cultures[cid] = CulturalProfile.from_dict(cdata)
        
        wb.magic_system_summary = d.get("magic_system_summary", "")
        wb.geography_summary = d.get("geography_summary", "")
        wb.history_summary = d.get("history_summary", "")
        wb.cosmology_summary = d.get("cosmology_summary", "")
        
        return wb
    
    @classmethod
    def from_json(cls, json_str: str) -> "WorldBible":
        return cls.from_dict(json.loads(json_str))
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== World Bible: {self.name} ===",
            f"",
            f"Source books: {', '.join(self.source_books)}",
            f"Total rules: {self.total_rules()}",
            f"Cultures documented: {len(self.cultures)}",
            f"",
            f"[Rules by Category]",
        ]
        
        for cat in WorldBibleCategory:
            rules = self.get_rules(cat)
            if rules:
                lines.append(f"  {cat.value}: {len(rules)} rules")
        
        if self.cultures:
            lines.append(f"")
            lines.append(f"[Cultures]")
            for culture in self.cultures.values():
                lines.append(f"  - {culture.name}")
        
        return "\n".join(lines)
