"""Cross-book entity resolution.

Resolves entities that appear across multiple books in a corpus,
merging them into unified canonical entities while tracking
their per-book appearances and handling conflicts.
"""

from dataclasses import dataclass, field
from typing import Literal
from collections import defaultdict
import json
from pathlib import Path

from rapidfuzz import fuzz
import httpx

from ..config import get_settings
from ..extract.dynamic_resolver import EntityCluster


@dataclass
class CrossBookEntity:
    """An entity that may appear across multiple books."""
    
    id: str
    canonical_name: str
    entity_type: Literal["character", "place", "object", "unknown"]
    
    # Per-book data: book_id -> EntityCluster.id
    book_clusters: dict[str, str] = field(default_factory=dict)
    
    # All known names/aliases across all books
    all_names: set[str] = field(default_factory=set)
    
    # Aggregate stats
    total_mentions: int = 0
    books_appeared_in: int = 0
    
    # Conflict tracking
    # If different books have conflicting info, track it
    conflicts: list[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "book_clusters": self.book_clusters,
            "all_names": list(self.all_names),
            "total_mentions": self.total_mentions,
            "books_appeared_in": self.books_appeared_in,
            "conflicts": self.conflicts,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "CrossBookEntity":
        return cls(
            id=d["id"],
            canonical_name=d["canonical_name"],
            entity_type=d["entity_type"],
            book_clusters=d.get("book_clusters", {}),
            all_names=set(d.get("all_names", [])),
            total_mentions=d.get("total_mentions", 0),
            books_appeared_in=d.get("books_appeared_in", 0),
            conflicts=d.get("conflicts", []),
        )


class CrossBookResolver:
    """
    Resolves entities across multiple books in a corpus.
    
    When processing multiple books by the same author, the same
    characters/places/things often appear. This resolver:
    
    1. Matches entities across books using fuzzy name matching
    2. Uses LLM to verify ambiguous matches
    3. Merges entity data while tracking sources
    4. Handles conflicts (different descriptions, etc.)
    
    Usage:
        resolver = CrossBookResolver("tolkien_corpus")
        
        # After processing each book, register its entities
        resolver.register_book_entities("the_hobbit", hobbit_clusters)
        resolver.register_book_entities("fellowship", fellowship_clusters)
        
        # Resolve cross-book matches
        resolver.resolve_all()
        
        # Get unified entity
        gandalf = resolver.get_entity("gandalf")
        print(gandalf.books_appeared_in)  # 2
    """
    
    def __init__(
        self,
        corpus_name: str,
        data_dir: str = "data/corpus",
        use_llm: bool = True,
        fuzzy_threshold: int = 85,
    ):
        """
        Initialize the cross-book resolver.
        
        Args:
            corpus_name: Name of the corpus
            data_dir: Directory for corpus data
            use_llm: Whether to use LLM for ambiguous matches
            fuzzy_threshold: Minimum fuzzy match score (0-100)
        """
        self.corpus_name = corpus_name
        self.data_dir = Path(data_dir)
        self.use_llm = use_llm
        self.fuzzy_threshold = fuzzy_threshold
        self.settings = get_settings()
        
        # Unified entity database
        self.entities: dict[str, CrossBookEntity] = {}
        
        # Per-book entity clusters (book_id -> cluster_id -> EntityCluster)
        self.book_clusters: dict[str, dict[str, EntityCluster]] = {}
        
        # Name lookup: lowercase name -> entity id
        self._name_index: dict[str, str] = {}
        
        # Track unresolved entities per book
        self._pending: dict[str, list[EntityCluster]] = defaultdict(list)
        
        # Resolution file
        self.resolution_file = self.data_dir / f"{corpus_name}_entities.json"
        self._load_if_exists()
    
    def _load_if_exists(self) -> None:
        """Load existing resolution data if available."""
        if self.resolution_file.exists():
            with open(self.resolution_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entity_data in data.get("entities", []):
                    entity = CrossBookEntity.from_dict(entity_data)
                    self.entities[entity.id] = entity
                    self._index_entity(entity)
    
    def _save(self) -> None:
        """Save resolution data."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "corpus": self.corpus_name,
            "entities": [e.to_dict() for e in self.entities.values()],
        }
        with open(self.resolution_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _index_entity(self, entity: CrossBookEntity) -> None:
        """Add entity to name index."""
        self._name_index[entity.canonical_name.lower()] = entity.id
        for name in entity.all_names:
            self._name_index[name.lower()] = entity.id
    
    def register_book_entities(
        self,
        book_id: str,
        clusters: dict[str, EntityCluster],
    ) -> None:
        """
        Register entities extracted from a book.
        
        Args:
            book_id: ID of the book
            clusters: EntityCluster dict from DynamicEntityResolver
        """
        self.book_clusters[book_id] = clusters
        
        for cluster in clusters.values():
            self._pending[book_id].append(cluster)
    
    def resolve_all(self) -> dict:
        """
        Resolve all pending entities across books.
        
        Returns:
            Resolution stats
        """
        stats = {
            "new_entities": 0,
            "merged_entities": 0,
            "llm_checks": 0,
            "conflicts": 0,
        }
        
        # Process each book's pending entities
        for book_id, clusters in list(self._pending.items()):
            for cluster in clusters:
                match = self._find_matching_entity(cluster)
                
                if match:
                    # Merge into existing entity
                    self._merge_cluster_into_entity(match, cluster, book_id)
                    stats["merged_entities"] += 1
                else:
                    # Create new cross-book entity
                    entity = self._create_entity_from_cluster(cluster, book_id)
                    self.entities[entity.id] = entity
                    self._index_entity(entity)
                    stats["new_entities"] += 1
            
            # Clear pending
            self._pending[book_id] = []
        
        # Second pass: check for cross-entity merges
        merge_count = self._consolidate_entities()
        stats["merged_entities"] += merge_count
        
        self._save()
        return stats
    
    def _find_matching_entity(self, cluster: EntityCluster) -> CrossBookEntity | None:
        """Find an existing cross-book entity that matches this cluster."""
        # Exact match on canonical name
        key = cluster.canonical_name.lower()
        if key in self._name_index:
            return self.entities[self._name_index[key]]
        
        # Check aliases
        for alias in cluster.aliases:
            if alias.lower() in self._name_index:
                return self.entities[self._name_index[alias.lower()]]
        
        # Fuzzy match
        best_match = None
        best_score = 0
        
        for name, entity_id in self._name_index.items():
            score = fuzz.ratio(key, name)
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = self.entities[entity_id]
        
        # If fuzzy match found but not super confident, verify with LLM
        if best_match and best_score < 95 and self.use_llm:
            if not self._llm_verify_match(cluster, best_match):
                return None
        
        return best_match
    
    def _llm_verify_match(self, cluster: EntityCluster, entity: CrossBookEntity) -> bool:
        """Use LLM to verify if a cluster matches an existing entity."""
        # Get context from cluster
        ctx = ""
        if cluster.mentions:
            ctx = cluster.mentions[0].passage_text[:150]
        
        prompt = f"""Are these two references to the same entity (character/place/thing)?

Entity 1: "{cluster.canonical_name}"
- Also known as: {', '.join(list(cluster.aliases)[:5]) or 'no aliases'}
- Type: {cluster.entity_type}
- Context: "{ctx[:100]}..."

Entity 2: "{entity.canonical_name}"
- Also known as: {', '.join(list(entity.all_names)[:5])}
- Type: {entity.entity_type}
- Appears in {entity.books_appeared_in} book(s)

Answer only YES or NO:"""

        try:
            response = httpx.post(
                f"{self.settings.ollama_base_url}/api/generate",
                json={
                    "model": self.settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=15.0,
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "").strip().upper()
                return result.startswith("YES")
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        
        # Default to fuzzy match result if LLM fails
        return True
    
    def _create_entity_from_cluster(
        self,
        cluster: EntityCluster,
        book_id: str,
    ) -> CrossBookEntity:
        """Create a new CrossBookEntity from a cluster."""
        entity = CrossBookEntity(
            id=cluster.id,
            canonical_name=cluster.canonical_name,
            entity_type=cluster.entity_type,
            book_clusters={book_id: cluster.id},
            all_names={cluster.canonical_name} | cluster.aliases,
            total_mentions=cluster.mention_count,
            books_appeared_in=1,
        )
        return entity
    
    def _merge_cluster_into_entity(
        self,
        entity: CrossBookEntity,
        cluster: EntityCluster,
        book_id: str,
    ) -> None:
        """Merge a cluster's data into an existing cross-book entity."""
        # Track book appearance
        entity.book_clusters[book_id] = cluster.id
        entity.books_appeared_in = len(entity.book_clusters)
        
        # Add mentions
        entity.total_mentions += cluster.mention_count
        
        # Add names/aliases
        entity.all_names.add(cluster.canonical_name)
        entity.all_names.update(cluster.aliases)
        
        # Update index
        for name in cluster.aliases:
            if name.lower() not in self._name_index:
                self._name_index[name.lower()] = entity.id
        
        # Check for type conflicts
        if (
            cluster.entity_type != "unknown"
            and entity.entity_type != "unknown"
            and cluster.entity_type != entity.entity_type
        ):
            entity.conflicts.append({
                "type": "entity_type_mismatch",
                "book": book_id,
                "existing": entity.entity_type,
                "new": cluster.entity_type,
            })
    
    def _consolidate_entities(self) -> int:
        """Find and merge cross-book entities that are actually the same."""
        merges = 0
        
        # Group by type first
        by_type: dict[str, list[CrossBookEntity]] = defaultdict(list)
        for entity in self.entities.values():
            by_type[entity.entity_type].append(entity)
        
        # Within each type, look for potential matches
        for entity_type, entities in by_type.items():
            if len(entities) < 2:
                continue
            
            # Compare all pairs
            to_merge: list[tuple[str, str]] = []
            
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1:]:
                    if self._should_merge_entities(e1, e2):
                        to_merge.append((e1.id, e2.id))
        
            # Execute merges
            for keep_id, merge_id in to_merge:
                if keep_id in self.entities and merge_id in self.entities:
                    self._merge_entities(keep_id, merge_id)
                    merges += 1
        
        return merges
    
    def _should_merge_entities(self, e1: CrossBookEntity, e2: CrossBookEntity) -> bool:
        """Determine if two entities should be merged."""
        # Check for name overlap
        names1 = {n.lower() for n in e1.all_names}
        names2 = {n.lower() for n in e2.all_names}
        
        if names1 & names2:
            return True
        
        # Check fuzzy match on canonical names
        score = fuzz.ratio(e1.canonical_name.lower(), e2.canonical_name.lower())
        if score >= 90:
            if self.use_llm:
                return self._llm_verify_entity_merge(e1, e2)
            return True
        
        return False
    
    def _llm_verify_entity_merge(self, e1: CrossBookEntity, e2: CrossBookEntity) -> bool:
        """Use LLM to verify two entities should be merged."""
        prompt = f"""Should these two entities from a book corpus be merged as the same entity?

Entity 1: "{e1.canonical_name}"
- Names: {', '.join(list(e1.all_names)[:5])}
- Type: {e1.entity_type}
- Mentioned {e1.total_mentions} times in {e1.books_appeared_in} book(s)

Entity 2: "{e2.canonical_name}"
- Names: {', '.join(list(e2.all_names)[:5])}
- Type: {e2.entity_type}
- Mentioned {e2.total_mentions} times in {e2.books_appeared_in} book(s)

Answer only YES or NO:"""

        try:
            response = httpx.post(
                f"{self.settings.ollama_base_url}/api/generate",
                json={
                    "model": self.settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=15.0,
            )
            
            if response.status_code == 200:
                result = response.json().get("response", "").strip().upper()
                return result.startswith("YES")
        except (httpx.RequestError, httpx.TimeoutException):
            pass
        
        return False
    
    def _merge_entities(self, keep_id: str, merge_id: str) -> None:
        """Merge one entity into another."""
        keep = self.entities[keep_id]
        merge = self.entities[merge_id]
        
        # Merge data
        keep.book_clusters.update(merge.book_clusters)
        keep.all_names.update(merge.all_names)
        keep.total_mentions += merge.total_mentions
        keep.books_appeared_in = len(keep.book_clusters)
        keep.conflicts.extend(merge.conflicts)
        
        # Update index
        for name in merge.all_names:
            self._name_index[name.lower()] = keep_id
        
        # Remove merged entity
        del self.entities[merge_id]
    
    def get_entity(self, name_or_id: str) -> CrossBookEntity | None:
        """Get an entity by name or ID."""
        # Direct ID lookup
        if name_or_id in self.entities:
            return self.entities[name_or_id]
        
        # Name lookup
        key = name_or_id.lower()
        if key in self._name_index:
            return self.entities[self._name_index[key]]
        
        return None
    
    def get_entities_by_type(self, entity_type: str) -> list[CrossBookEntity]:
        """Get all entities of a given type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]
    
    def get_multi_book_entities(self) -> list[CrossBookEntity]:
        """Get entities that appear in multiple books."""
        return [e for e in self.entities.values() if e.books_appeared_in > 1]
    
    def get_entity_appearances(self, entity_id: str) -> dict[str, int]:
        """Get book-by-book mention counts for an entity."""
        entity = self.entities.get(entity_id)
        if not entity:
            return {}
        
        appearances = {}
        for book_id, cluster_id in entity.book_clusters.items():
            if book_id in self.book_clusters:
                cluster = self.book_clusters[book_id].get(cluster_id)
                if cluster:
                    appearances[book_id] = cluster.mention_count
        
        return appearances
    
    @property
    def stats(self) -> dict:
        """Get resolution statistics."""
        by_type = defaultdict(int)
        multi_book = 0
        total_conflicts = 0
        
        for entity in self.entities.values():
            by_type[entity.entity_type] += 1
            if entity.books_appeared_in > 1:
                multi_book += 1
            total_conflicts += len(entity.conflicts)
        
        return {
            "total_entities": len(self.entities),
            "by_type": dict(by_type),
            "multi_book_entities": multi_book,
            "total_conflicts": total_conflicts,
        }
    
    def summary(self) -> str:
        """Generate a summary of cross-book entity resolution."""
        stats = self.stats
        lines = [
            f"=== Cross-Book Entity Resolution: {self.corpus_name} ===",
            f"Total entities: {stats['total_entities']}",
            f"Multi-book entities: {stats['multi_book_entities']}",
            f"",
            "[By Type]",
        ]
        
        for etype, count in stats["by_type"].items():
            lines.append(f"  {etype}: {count}")
        
        if stats["total_conflicts"] > 0:
            lines.append(f"\n[!] {stats['total_conflicts']} conflict(s) detected")
        
        # Show top multi-book entities
        multi = self.get_multi_book_entities()
        if multi:
            lines.append(f"\n[Top Multi-Book Entities]")
            for entity in sorted(multi, key=lambda e: -e.total_mentions)[:10]:
                books = list(entity.book_clusters.keys())
                lines.append(f"  {entity.canonical_name}: {entity.total_mentions} mentions in {books}")
        
        return "\n".join(lines)
