"""Dynamic entity resolution - builds entity database from text without pre-seeding.

This module enables zero-seed entity extraction for any novel by:
1. Extracting entities via NER
2. Clustering co-occurring mentions
3. Using LLM to detect aliases and merge entities
4. Building a canonical entity database dynamically
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import httpx
from rapidfuzz import fuzz

from ..config import get_settings
from ..models.entities import Character, Place, Object
from .ner import ExtractedEntity


@dataclass
class EntityMention:
    """A single mention of an entity in text."""
    
    text: str
    label: str  # PERSON, PLACE, OBJECT, etc.
    passage_id: str
    passage_text: str
    char_offset: int
    context_before: str = ""  # 50 chars before
    context_after: str = ""   # 50 chars after


@dataclass 
class EntityCluster:
    """A cluster of mentions believed to be the same entity."""
    
    id: str
    canonical_name: str
    entity_type: Literal["character", "place", "object", "unknown"]
    mentions: list[EntityMention] = field(default_factory=list)
    aliases: set[str] = field(default_factory=set)
    confidence: float = 1.0
    
    # Metadata extracted from context
    descriptors: set[str] = field(default_factory=set)  # "the wizard", "old man"
    
    @property
    def mention_count(self) -> int:
        return len(self.mentions)
    
    def add_mention(self, mention: EntityMention) -> None:
        """Add a mention and update aliases."""
        self.mentions.append(mention)
        if mention.text.lower() != self.canonical_name.lower():
            self.aliases.add(mention.text)
    
    def merge_with(self, other: "EntityCluster") -> None:
        """Merge another cluster into this one."""
        self.mentions.extend(other.mentions)
        self.aliases.update(other.aliases)
        self.aliases.add(other.canonical_name)
        self.descriptors.update(other.descriptors)


class DynamicEntityResolver:
    """Builds entity database dynamically from text without pre-seeding."""
    
    def __init__(self, use_llm: bool = True):
        """Initialize the dynamic resolver.
        
        Args:
            use_llm: Whether to use LLM for alias detection
        """
        self.settings = get_settings()
        self.use_llm = use_llm
        
        # Entity clusters by type
        self.clusters: dict[str, EntityCluster] = {}
        
        # Lookup tables for fast matching
        self._name_to_cluster: dict[str, str] = {}  # lowercase name -> cluster id
        
        # Co-occurrence tracking for alias detection
        self._cooccurrence: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Passage context for LLM queries
        self._recent_passages: list[str] = []
    
    def process_mention(
        self,
        entity: ExtractedEntity,
        passage_id: str,
        passage_text: str,
    ) -> EntityCluster:
        """Process a single entity mention, creating or updating clusters.
        
        Args:
            entity: The extracted entity
            passage_id: ID of the source passage
            passage_text: Full text of the passage
            
        Returns:
            The EntityCluster this mention was assigned to
        """
        mention = EntityMention(
            text=entity.text,
            label=entity.label,
            passage_id=passage_id,
            passage_text=passage_text,
            char_offset=entity.start_char,
            context_before=passage_text[max(0, entity.start_char - 50):entity.start_char],
            context_after=passage_text[entity.end_char:entity.end_char + 50],
        )
        
        # Try to find existing cluster
        cluster = self._find_matching_cluster(mention)
        
        if cluster:
            cluster.add_mention(mention)
        else:
            # Create new cluster
            cluster = self._create_cluster(mention)
        
        # Track co-occurrence for later alias detection
        self._track_cooccurrence(mention, passage_id)
        
        return cluster
    
    def _find_matching_cluster(self, mention: EntityMention) -> EntityCluster | None:
        """Find an existing cluster that matches this mention."""
        text_lower = mention.text.lower().strip()
        
        # Exact match on name or alias
        if text_lower in self._name_to_cluster:
            return self.clusters[self._name_to_cluster[text_lower]]
        
        # Try without articles
        for prefix in ["the ", "a ", "an "]:
            if text_lower.startswith(prefix):
                stripped = text_lower[len(prefix):]
                if stripped in self._name_to_cluster:
                    return self.clusters[self._name_to_cluster[stripped]]
        
        # Fuzzy match for typos/variations
        for name, cluster_id in self._name_to_cluster.items():
            if fuzz.ratio(text_lower, name) >= 90:
                return self.clusters[cluster_id]
        
        return None
    
    def _create_cluster(self, mention: EntityMention) -> EntityCluster:
        """Create a new entity cluster from a mention."""
        # Generate ID
        base_name = mention.text.lower().replace(" ", "_")
        base_name = re.sub(r"[^a-z0-9_]", "", base_name)
        cluster_id = f"{base_name}_{len(self.clusters)}"
        
        # Determine entity type from label
        type_map = {
            "PERSON": "character",
            "GPE": "place",
            "LOC": "place", 
            "FAC": "place",
            "ORG": "character",  # Often groups/peoples
            "PRODUCT": "object",
            "WORK_OF_ART": "object",
        }
        entity_type = type_map.get(mention.label, "unknown")
        
        cluster = EntityCluster(
            id=cluster_id,
            canonical_name=mention.text,
            entity_type=entity_type,
            mentions=[mention],
        )
        
        self.clusters[cluster_id] = cluster
        self._name_to_cluster[mention.text.lower()] = cluster_id
        
        return cluster
    
    def _track_cooccurrence(self, mention: EntityMention, passage_id: str) -> None:
        """Track which entities appear together for alias detection."""
        text_lower = mention.text.lower()
        
        # Find other entities in the same passage
        for cluster_id, cluster in self.clusters.items():
            for m in cluster.mentions:
                if m.passage_id == passage_id and m.text.lower() != text_lower:
                    self._cooccurrence[text_lower][m.text.lower()] += 1
    
    def consolidate_clusters(self, min_cooccurrence: int = 3) -> int:
        """Consolidate clusters that likely refer to the same entity.
        
        Uses co-occurrence patterns and optionally LLM to detect aliases.
        
        Args:
            min_cooccurrence: Minimum co-occurrences to consider merging
            
        Returns:
            Number of merges performed
        """
        merges = 0
        
        # Find high co-occurrence pairs
        merge_candidates = []
        for name1, cooccurs in self._cooccurrence.items():
            for name2, count in cooccurs.items():
                if count >= min_cooccurrence and name1 < name2:
                    merge_candidates.append((name1, name2, count))
        
        # Sort by co-occurrence count
        merge_candidates.sort(key=lambda x: -x[2])
        
        # Process candidates
        for name1, name2, count in merge_candidates:
            cluster1_id = self._name_to_cluster.get(name1)
            cluster2_id = self._name_to_cluster.get(name2)
            
            if not cluster1_id or not cluster2_id:
                continue
            if cluster1_id == cluster2_id:
                continue  # Already merged
            
            cluster1 = self.clusters.get(cluster1_id)
            cluster2 = self.clusters.get(cluster2_id)
            
            if not cluster1 or not cluster2:
                continue
            
            # Check if they should be merged
            should_merge = self._should_merge(cluster1, cluster2)
            
            if should_merge:
                self._merge_clusters(cluster1, cluster2)
                merges += 1
        
        return merges
    
    def _should_merge(self, c1: EntityCluster, c2: EntityCluster) -> bool:
        """Determine if two clusters should be merged."""
        # Same type required
        if c1.entity_type != c2.entity_type and c1.entity_type != "unknown" and c2.entity_type != "unknown":
            return False
        
        # If LLM enabled, ask it
        if self.use_llm:
            return self._llm_check_alias(c1, c2)
        
        # Heuristic: if one is a substring of the other, likely same
        n1 = c1.canonical_name.lower()
        n2 = c2.canonical_name.lower()
        if n1 in n2 or n2 in n1:
            return True
        
        # Check for common patterns like "X" and "the X"
        if n1 == f"the {n2}" or n2 == f"the {n1}":
            return True
        
        return False
    
    def _llm_check_alias(self, c1: EntityCluster, c2: EntityCluster) -> bool:
        """Use LLM to check if two entities are the same."""
        # Get sample contexts
        ctx1 = c1.mentions[0].passage_text[:100] if c1.mentions else ""
        ctx2 = c2.mentions[0].passage_text[:100] if c2.mentions else ""
        
        prompt = f"""In this novel, are "{c1.canonical_name}" and "{c2.canonical_name}" the same entity (person/place/thing)?

Context for "{c1.canonical_name}": "{ctx1}"
Context for "{c2.canonical_name}": "{ctx2}"

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
    
    def _merge_clusters(self, keep: EntityCluster, merge: EntityCluster) -> None:
        """Merge one cluster into another."""
        keep.merge_with(merge)
        
        # Update lookup table
        self._name_to_cluster[merge.canonical_name.lower()] = keep.id
        for alias in merge.aliases:
            self._name_to_cluster[alias.lower()] = keep.id
        
        # Remove merged cluster
        del self.clusters[merge.id]
    
    def detect_aliases_from_text(self, text: str, passage_id: str) -> list[tuple[str, str]]:
        """Detect alias relationships from explicit statements in text.
        
        Looks for patterns like:
        - "X, whose real name was Y"
        - "X (also known as Y)"
        - "X, or Y as he was called"
        
        Args:
            text: The passage text
            passage_id: ID of the passage
            
        Returns:
            List of (name1, name2) alias pairs found
        """
        aliases = []
        
        patterns = [
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+whose\s+(?:real\s+)?name\s+was\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*\((?:also\s+)?(?:known|called)\s+(?:as\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+or\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+as\s+(?:he|she|they)\s+(?:was|were)\s+(?:called|known)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s+(?:whom|who)\s+(?:they|we|people)\s+(?:called|named)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                name1, name2 = match.group(1), match.group(2)
                aliases.append((name1, name2))
                
                # Merge if both exist
                c1_id = self._name_to_cluster.get(name1.lower())
                c2_id = self._name_to_cluster.get(name2.lower())
                
                if c1_id and c2_id and c1_id != c2_id:
                    c1 = self.clusters.get(c1_id)
                    c2 = self.clusters.get(c2_id)
                    if c1 and c2:
                        self._merge_clusters(c1, c2)
        
        return aliases
    
    def resolve(self, text: str) -> tuple[str | None, str | None, float]:
        """Resolve a text mention to a cluster.
        
        Args:
            text: The entity text to resolve
            
        Returns:
            Tuple of (cluster_id, canonical_name, confidence)
        """
        text_lower = text.lower().strip()
        
        # Exact match
        if text_lower in self._name_to_cluster:
            cluster = self.clusters[self._name_to_cluster[text_lower]]
            return cluster.id, cluster.canonical_name, 1.0
        
        # Without articles
        for prefix in ["the ", "a ", "an "]:
            if text_lower.startswith(prefix):
                stripped = text_lower[len(prefix):]
                if stripped in self._name_to_cluster:
                    cluster = self.clusters[self._name_to_cluster[stripped]]
                    return cluster.id, cluster.canonical_name, 0.95
        
        # Fuzzy match
        best_match = None
        best_score = 0
        for name, cluster_id in self._name_to_cluster.items():
            score = fuzz.ratio(text_lower, name)
            if score > best_score and score >= 85:
                best_score = score
                best_match = cluster_id
        
        if best_match:
            cluster = self.clusters[best_match]
            return cluster.id, cluster.canonical_name, best_score / 100
        
        return None, None, 0.0
    
    def get_entities_by_type(self, entity_type: str) -> list[EntityCluster]:
        """Get all clusters of a specific type."""
        return [c for c in self.clusters.values() if c.entity_type == entity_type]
    
    def get_top_entities(self, n: int = 20) -> list[EntityCluster]:
        """Get the most frequently mentioned entities."""
        sorted_clusters = sorted(
            self.clusters.values(),
            key=lambda c: c.mention_count,
            reverse=True,
        )
        return sorted_clusters[:n]
    
    @property
    def stats(self) -> dict:
        """Get statistics about the resolved entities."""
        by_type = defaultdict(int)
        for cluster in self.clusters.values():
            by_type[cluster.entity_type] += 1
        
        return {
            "total_clusters": len(self.clusters),
            "total_mentions": sum(c.mention_count for c in self.clusters.values()),
            "by_type": dict(by_type),
            "total_aliases": sum(len(c.aliases) for c in self.clusters.values()),
        }
    
    def export_to_seed_format(self) -> dict:
        """Export clusters to seed file format for future use."""
        characters = []
        places = []
        objects = []
        
        for cluster in self.clusters.values():
            entry = {
                "id": cluster.id,
                "canonical_name": cluster.canonical_name,
                "aliases": list(cluster.aliases),
                "description": f"Mentioned {cluster.mention_count} times",
            }
            
            if cluster.entity_type == "character":
                characters.append(entry)
            elif cluster.entity_type == "place":
                places.append(entry)
            elif cluster.entity_type == "object":
                objects.append(entry)
        
        return {
            "characters": characters,
            "places": places,
            "objects": objects,
        }
