"""Temporal reasoning for lore checking.

Handles:
- Age/era validation (First Age, Second Age, etc.)
- Event ordering (X happened before Y)
- Lifespan checking (character alive during event)
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..llm import LLMClient


class Era(Enum):
    """Major time periods in fantasy worlds."""
    BEFORE_TIME = "before_time"
    YEARS_OF_TREES = "years_of_trees"
    FIRST_AGE = "first_age"
    SECOND_AGE = "second_age"
    THIRD_AGE = "third_age"
    FOURTH_AGE = "fourth_age"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_text(cls, text: str) -> "Era":
        """Parse era from text."""
        text_lower = text.lower()
        
        if "first age" in text_lower or "elder days" in text_lower:
            return cls.FIRST_AGE
        elif "second age" in text_lower:
            return cls.SECOND_AGE
        elif "third age" in text_lower:
            return cls.THIRD_AGE
        elif "fourth age" in text_lower:
            return cls.FOURTH_AGE
        elif "years of the trees" in text_lower:
            return cls.YEARS_OF_TREES
        elif "before" in text_lower and ("time" in text_lower or "sun" in text_lower):
            return cls.BEFORE_TIME
        
        return cls.UNKNOWN
    
    @property
    def order(self) -> int:
        """Numeric order for comparison."""
        return {
            Era.BEFORE_TIME: 0,
            Era.YEARS_OF_TREES: 1,
            Era.FIRST_AGE: 2,
            Era.SECOND_AGE: 3,
            Era.THIRD_AGE: 4,
            Era.FOURTH_AGE: 5,
            Era.UNKNOWN: -1,
        }[self]
    
    def __lt__(self, other: "Era") -> bool:
        return self.order < other.order
    
    def __le__(self, other: "Era") -> bool:
        return self.order <= other.order


@dataclass
class TemporalEntity:
    """An entity with temporal bounds."""
    name: str
    entity_type: str  # character, event, place, object
    
    # When did this entity exist/occur?
    birth_era: Optional[Era] = None
    death_era: Optional[Era] = None
    birth_year: Optional[int] = None  # Year within era
    death_year: Optional[int] = None
    
    # For events
    event_era: Optional[Era] = None
    event_year: Optional[int] = None
    
    # Evidence
    source_text: str = ""
    
    def alive_during(self, era: Era) -> Optional[bool]:
        """Check if entity was alive during an era.
        
        Returns:
            True if definitely alive
            False if definitely not alive
            None if unknown
        """
        if self.entity_type == "event":
            return None  # Events don't have lifespans
        
        if self.birth_era is None and self.death_era is None:
            return None  # Unknown
        
        if self.birth_era and era < self.birth_era:
            return False  # Era before birth
        
        if self.death_era and era > self.death_era:
            return False  # Era after death
        
        if self.birth_era and self.death_era:
            if self.birth_era <= era <= self.death_era:
                return True
        
        return None  # Partially known
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "birth_era": self.birth_era.value if self.birth_era else None,
            "death_era": self.death_era.value if self.death_era else None,
            "birth_year": self.birth_year,
            "death_year": self.death_year,
            "event_era": self.event_era.value if self.event_era else None,
            "event_year": self.event_year,
            "source_text": self.source_text,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "TemporalEntity":
        return cls(
            name=d["name"],
            entity_type=d["entity_type"],
            birth_era=Era(d["birth_era"]) if d.get("birth_era") else None,
            death_era=Era(d["death_era"]) if d.get("death_era") else None,
            birth_year=d.get("birth_year"),
            death_year=d.get("death_year"),
            event_era=Era(d["event_era"]) if d.get("event_era") else None,
            event_year=d.get("event_year"),
            source_text=d.get("source_text", ""),
        )


@dataclass
class TemporalRelation:
    """A temporal relationship between entities/events."""
    subject: str
    relation: str  # "before", "after", "during", "caused", "witnessed"
    object: str
    confidence: float = 1.0
    source_text: str = ""


class Timeline:
    """Timeline of entities and events for a world.
    
    Enables queries like:
    - Was X alive during the Second Age?
    - Did X happen before Y?
    - Who was alive when X occurred?
    """
    
    def __init__(self):
        self.entities: dict[str, TemporalEntity] = {}
        self.relations: list[TemporalRelation] = []
        self._name_index: dict[str, str] = {}  # lowercase -> canonical
    
    def add_entity(self, entity: TemporalEntity) -> None:
        """Add an entity to the timeline."""
        self.entities[entity.name] = entity
        self._name_index[entity.name.lower()] = entity.name
    
    def add_relation(self, relation: TemporalRelation) -> None:
        """Add a temporal relation."""
        self.relations.append(relation)
    
    def get_entity(self, name: str) -> Optional[TemporalEntity]:
        """Get entity by name (case-insensitive)."""
        canonical = self._name_index.get(name.lower())
        if canonical:
            return self.entities.get(canonical)
        return None
    
    def alive_during_era(self, name: str, era: Era) -> Optional[bool]:
        """Check if entity was alive during era."""
        entity = self.get_entity(name)
        if entity:
            return entity.alive_during(era)
        return None
    
    def happened_before(self, event1: str, event2: str) -> Optional[bool]:
        """Check if event1 happened before event2."""
        e1 = self.get_entity(event1)
        e2 = self.get_entity(event2)
        
        if not e1 or not e2:
            return None
        
        # Check explicit relations
        for rel in self.relations:
            if rel.subject.lower() == event1.lower() and rel.object.lower() == event2.lower():
                if rel.relation == "before":
                    return True
                elif rel.relation == "after":
                    return False
            if rel.subject.lower() == event2.lower() and rel.object.lower() == event1.lower():
                if rel.relation == "before":
                    return False
                elif rel.relation == "after":
                    return True
        
        # Check era ordering
        era1 = e1.event_era or e1.birth_era
        era2 = e2.event_era or e2.birth_era
        
        if era1 and era2 and era1 != Era.UNKNOWN and era2 != Era.UNKNOWN:
            if era1 < era2:
                return True
            elif era1 > era2:
                return False
        
        return None
    
    def to_dict(self) -> dict:
        return {
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "relations": [
                {
                    "subject": r.subject,
                    "relation": r.relation,
                    "object": r.object,
                    "confidence": r.confidence,
                }
                for r in self.relations
            ],
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Timeline":
        timeline = cls()
        for k, v in d.get("entities", {}).items():
            timeline.entities[k] = TemporalEntity.from_dict(v)
            timeline._name_index[k.lower()] = k
        for r in d.get("relations", []):
            timeline.relations.append(TemporalRelation(**r))
        return timeline


class TemporalExtractor:
    """Extracts temporal information from text.
    
    Identifies:
    - Birth/death mentions
    - Era references
    - Event ordering
    - Temporal relationships
    """
    
    # Patterns for temporal extraction
    BIRTH_PATTERNS = [
        r"(\w+(?:\s+\w+)?)\s+was\s+born\s+(?:in\s+)?(?:the\s+)?(\w+\s+Age)",
        r"(\w+(?:\s+\w+)?)\s+came\s+into\s+(?:the\s+)?world\s+(?:in\s+)?(?:the\s+)?(\w+\s+Age)",
    ]
    
    DEATH_PATTERNS = [
        r"(\w+(?:\s+\w+)?)\s+(?:died|fell|perished)\s+(?:in\s+)?(?:the\s+)?(\w+\s+Age)",
        r"(?:death|end)\s+of\s+(\w+(?:\s+\w+)?)\s+(?:in\s+)?(?:the\s+)?(\w+\s+Age)",
    ]
    
    LIVED_PATTERNS = [
        r"(\w+(?:\s+\w+)?)\s+(?:lived|dwelt)\s+(?:in\s+)?(?:the\s+)?(\w+\s+Age)",
        r"(\w+(?:\s+\w+)?)\s+(?:of|in)\s+(?:the\s+)?(\w+\s+Age)",
    ]
    
    EVENT_PATTERNS = [
        r"(?:the\s+)?(\w+(?:\s+\w+)*)\s+(?:occurred|happened|took\s+place)\s+(?:in\s+)?(?:the\s+)?(\w+\s+Age)",
        r"(?:in\s+)?(?:the\s+)?(\w+\s+Age)[,\s]+(?:the\s+)?(\w+(?:\s+\w+)*)\s+(?:began|ended|occurred)",
    ]
    
    ORDER_PATTERNS = [
        (r"(\w+(?:\s+\w+)?)\s+(?:came\s+)?before\s+(\w+(?:\s+\w+)?)", "before"),
        (r"(\w+(?:\s+\w+)?)\s+(?:came\s+)?after\s+(\w+(?:\s+\w+)?)", "after"),
        (r"(\w+(?:\s+\w+)?)\s+preceded\s+(\w+(?:\s+\w+)?)", "before"),
        (r"(\w+(?:\s+\w+)?)\s+followed\s+(\w+(?:\s+\w+)?)", "after"),
    ]
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        self._birth_patterns = [re.compile(p, re.IGNORECASE) for p in self.BIRTH_PATTERNS]
        self._death_patterns = [re.compile(p, re.IGNORECASE) for p in self.DEATH_PATTERNS]
        self._lived_patterns = [re.compile(p, re.IGNORECASE) for p in self.LIVED_PATTERNS]
        self._event_patterns = [re.compile(p, re.IGNORECASE) for p in self.EVENT_PATTERNS]
        self._order_patterns = [(re.compile(p, re.IGNORECASE), r) for p, r in self.ORDER_PATTERNS]
    
    def extract_from_text(self, text: str) -> Timeline:
        """Extract timeline from text."""
        timeline = Timeline()
        
        # Extract birth/death
        for pattern in self._birth_patterns:
            for match in pattern.finditer(text):
                name = match.group(1)
                era = Era.from_text(match.group(2))
                entity = timeline.get_entity(name) or TemporalEntity(name=name, entity_type="character")
                entity.birth_era = era
                entity.source_text = match.group(0)
                timeline.add_entity(entity)
        
        for pattern in self._death_patterns:
            for match in pattern.finditer(text):
                name = match.group(1)
                era = Era.from_text(match.group(2))
                entity = timeline.get_entity(name) or TemporalEntity(name=name, entity_type="character")
                entity.death_era = era
                entity.source_text = match.group(0)
                timeline.add_entity(entity)
        
        # Extract "lived in" mentions
        for pattern in self._lived_patterns:
            for match in pattern.finditer(text):
                name = match.group(1)
                era = Era.from_text(match.group(2))
                if not timeline.get_entity(name):
                    entity = TemporalEntity(name=name, entity_type="character")
                    entity.birth_era = era  # Assume they were alive during this era
                    entity.death_era = era
                    entity.source_text = match.group(0)
                    timeline.add_entity(entity)
        
        # Extract temporal ordering
        for pattern, relation in self._order_patterns:
            for match in pattern.finditer(text):
                timeline.add_relation(TemporalRelation(
                    subject=match.group(1),
                    relation=relation,
                    object=match.group(2),
                    source_text=match.group(0),
                ))
        
        # Use LLM for complex extraction
        if self.use_llm:
            llm_entities = self._extract_llm(text)
            for entity in llm_entities:
                if not timeline.get_entity(entity.name):
                    timeline.add_entity(entity)
        
        return timeline
    
    def _extract_llm(self, text: str) -> list[TemporalEntity]:
        """Use LLM to extract temporal information."""
        # Limit text size
        text = text[:3000]
        
        prompt = f"""Extract temporal information about characters and events from this fantasy text.

Text:
{text}

For each character or event with temporal info, provide:
- name: The character or event name
- type: "character" or "event"
- era: Which Age they lived/occurred in (First Age, Second Age, Third Age, etc.)
- born_era: When born (if character)
- died_era: When died (if character)

Return as JSON array of objects. Only include entities with clear temporal information.

JSON:"""

        llm = LLMClient()
        response = llm.generate(prompt, temperature=0.1, max_tokens=1000)
        
        entities = []
        if response:
            data = llm.extract_json(response)
            if data and isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        entity = TemporalEntity(
                            name=item["name"],
                            entity_type=item.get("type", "character"),
                        )
                        if item.get("era"):
                            era = Era.from_text(item["era"])
                            entity.birth_era = era
                            entity.death_era = era
                        if item.get("born_era"):
                            entity.birth_era = Era.from_text(item["born_era"])
                        if item.get("died_era"):
                            entity.death_era = Era.from_text(item["died_era"])
                        entities.append(entity)
        
        return entities
