"""
Cross-Book Entity Resolution

Resolves entities across multiple books to ensure the same character/place/object
gets the same canonical ID even when appearing in different works.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path

from rapidfuzz import fuzz


@dataclass
class CanonicalEntity:
    """A canonical entity that may appear across multiple books."""
    id: str
    canonical_name: str
    entity_type: str  # character, place, object
    aliases: set[str] = field(default_factory=set)
    
    # Track where this entity appears
    appearances: dict[str, int] = field(default_factory=dict)  # book_id -> mention_count
    
    # Optional metadata
    description: Optional[str] = None
    first_appearance: Optional[str] = None  # book_id
    
    def add_alias(self, alias: str) -> None:
        """Add an alias for this entity."""
        if alias.lower() != self.canonical_name.lower():
            self.aliases.add(alias)
    
    def matches(self, name: str, threshold: float = 85.0) -> bool:
        """Check if a name matches this entity."""
        name_lower = name.lower()
        
        # Exact match
        if name_lower == self.canonical_name.lower():
            return True
        
        # Alias match
        for alias in self.aliases:
            if name_lower == alias.lower():
                return True
        
        # Fuzzy match
        if fuzz.ratio(name_lower, self.canonical_name.lower()) >= threshold:
            return True
        
        for alias in self.aliases:
            if fuzz.ratio(name_lower, alias.lower()) >= threshold:
                return True
        
        return False
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "aliases": list(self.aliases),
            "appearances": self.appearances,
            "description": self.description,
            "first_appearance": self.first_appearance,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "CanonicalEntity":
        return cls(
            id=d["id"],
            canonical_name=d["canonical_name"],
            entity_type=d["entity_type"],
            aliases=set(d.get("aliases", [])),
            appearances=d.get("appearances", {}),
            description=d.get("description"),
            first_appearance=d.get("first_appearance"),
        )


class CrossBookResolver:
    """
    Resolves entities across multiple books in a corpus.
    
    Maintains a master entity list and resolves new mentions
    to existing entities or creates new ones.
    
    Usage:
        resolver = CrossBookResolver("tolkien_corpus")
        
        # When processing a book
        canonical_id = resolver.resolve("Gandalf", "character", "the_hobbit")
        # Returns same ID whether from Hobbit, LOTR, or Silmarillion
    """
    
    def __init__(
        self,
        corpus_name: str,
        data_dir: str = "data/corpus",
        fuzzy_threshold: float = 85.0,
    ):
        """
        Initialize the cross-book resolver.
        
        Args:
            corpus_name: Name of the corpus
            data_dir: Directory for data files
            fuzzy_threshold: Threshold for fuzzy matching (0-100)
        """
        self.corpus_name = corpus_name
        self.data_dir = Path(data_dir)
        self.fuzzy_threshold = fuzzy_threshold
        
        self.entities_file = self.data_dir / f"{corpus_name}_entities.json"
        
        # Load or initialize entities
        self.entities: dict[str, CanonicalEntity] = {}  # id -> entity
        self._load_entities()
    
    def _load_entities(self) -> None:
        """Load entities from file."""
        if self.entities_file.exists():
            with open(self.entities_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entity_data in data.get("entities", []):
                    entity = CanonicalEntity.from_dict(entity_data)
                    self.entities[entity.id] = entity
    
    def _save_entities(self) -> None:
        """Save entities to file."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "corpus": self.corpus_name,
            "entity_count": len(self.entities),
            "entities": [e.to_dict() for e in self.entities.values()],
        }
        with open(self.entities_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def resolve(
        self,
        name: str,
        entity_type: str,
        book_id: str,
        create_if_missing: bool = True,
    ) -> Optional[str]:
        """
        Resolve a name to a canonical entity ID.
        
        Args:
            name: The entity name to resolve
            entity_type: Type of entity (character, place, object)
            book_id: ID of the book this mention is from
            create_if_missing: Whether to create new entity if not found
            
        Returns:
            Canonical entity ID, or None if not found and create_if_missing=False
        """
        # First, try exact/fuzzy match against existing entities
        for entity in self.entities.values():
            if entity.entity_type == entity_type and entity.matches(name, self.fuzzy_threshold):
                # Found match - update appearances
                entity.appearances[book_id] = entity.appearances.get(book_id, 0) + 1
                entity.add_alias(name)
                return entity.id
        
        # No match found
        if not create_if_missing:
            return None
        
        # Create new entity
        entity_id = self._generate_id(name, entity_type)
        entity = CanonicalEntity(
            id=entity_id,
            canonical_name=name,
            entity_type=entity_type,
            first_appearance=book_id,
            appearances={book_id: 1},
        )
        self.entities[entity_id] = entity
        
        return entity_id
    
    def _generate_id(self, name: str, entity_type: str) -> str:
        """Generate a unique ID for an entity."""
        base_id = f"{entity_type}_{name.lower().replace(' ', '_').replace("'", '')}"
        
        # Handle duplicates
        if base_id not in self.entities:
            return base_id
        
        counter = 2
        while f"{base_id}_{counter}" in self.entities:
            counter += 1
        return f"{base_id}_{counter}"
    
    def add_alias(self, entity_id: str, alias: str) -> bool:
        """
        Add an alias to an existing entity.
        
        Returns:
            True if alias was added, False if entity not found
        """
        if entity_id not in self.entities:
            return False
        
        self.entities[entity_id].add_alias(alias)
        return True
    
    def merge_entities(self, keep_id: str, merge_id: str) -> bool:
        """
        Merge two entities that are actually the same.
        
        Args:
            keep_id: ID of entity to keep
            merge_id: ID of entity to merge into keep_id
            
        Returns:
            True if merge successful
        """
        if keep_id not in self.entities or merge_id not in self.entities:
            return False
        
        keep = self.entities[keep_id]
        merge = self.entities[merge_id]
        
        # Transfer aliases
        keep.aliases.add(merge.canonical_name)
        keep.aliases.update(merge.aliases)
        
        # Merge appearances
        for book_id, count in merge.appearances.items():
            keep.appearances[book_id] = keep.appearances.get(book_id, 0) + count
        
        # Remove merged entity
        del self.entities[merge_id]
        
        return True
    
    def get_entity(self, entity_id: str) -> Optional[CanonicalEntity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)
    
    def find_entity(self, name: str, entity_type: str) -> Optional[CanonicalEntity]:
        """Find entity by name without creating."""
        for entity in self.entities.values():
            if entity.entity_type == entity_type and entity.matches(name, self.fuzzy_threshold):
                return entity
        return None
    
    def get_entities_by_type(self, entity_type: str) -> list[CanonicalEntity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]
    
    def get_cross_book_entities(self) -> list[CanonicalEntity]:
        """Get entities that appear in multiple books."""
        return [e for e in self.entities.values() if len(e.appearances) > 1]
    
    def save(self) -> None:
        """Save entities to disk."""
        self._save_entities()
    
    def summary(self) -> str:
        """Generate summary of resolved entities."""
        by_type = {}
        cross_book = 0
        
        for entity in self.entities.values():
            by_type[entity.entity_type] = by_type.get(entity.entity_type, 0) + 1
            if len(entity.appearances) > 1:
                cross_book += 1
        
        lines = [
            f"=== Cross-Book Entity Resolution ===",
            f"Corpus: {self.corpus_name}",
            f"Total entities: {len(self.entities)}",
            f"Cross-book entities: {cross_book}",
            f"",
            f"[By Type]",
        ]
        
        for etype, count in sorted(by_type.items()):
            lines.append(f"  {etype}: {count}")
        
        # Show some cross-book entities
        cross_book_entities = self.get_cross_book_entities()
        if cross_book_entities:
            lines.append(f"")
            lines.append(f"[Cross-Book Characters]")
            for entity in sorted(cross_book_entities, key=lambda e: -sum(e.appearances.values()))[:10]:
                books = ", ".join(entity.appearances.keys())
                lines.append(f"  {entity.canonical_name}: {books}")
        
        return "\n".join(lines)
