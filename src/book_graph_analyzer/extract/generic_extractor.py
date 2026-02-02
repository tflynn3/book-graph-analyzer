"""Generic entity extraction that works on any novel without pre-seeding.

Two-pass extraction:
1. Pass 1: Extract all entities, build dynamic entity database
2. Pass 2: Consolidate aliases, extract relationships
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ..config import get_settings
from ..ingest.loader import load_book
from ..ingest.splitter import split_into_passages, Passage
from .ner import NERPipeline
from .dynamic_resolver import DynamicEntityResolver, EntityCluster
from .relationships import RelationshipExtractor, RelationshipExtractionResult
from ..models.relationships import ExtractedRelationship


@dataclass
class GenericExtractionResult:
    """Result of generic extraction from a passage."""
    
    passage: Passage
    entities: list[EntityCluster]
    relationships: list[ExtractedRelationship] = field(default_factory=list)


@dataclass 
class BookAnalysis:
    """Complete analysis of a book."""
    
    title: str
    total_passages: int
    
    # Entities
    entity_clusters: dict[str, EntityCluster]
    total_mentions: int
    
    # Relationships  
    relationships: list[ExtractedRelationship]
    relationship_counts: dict[str, int]
    
    # Per-passage results
    passage_results: list[GenericExtractionResult] = field(default_factory=list)
    
    def get_character_graph(self) -> dict:
        """Get character interaction graph."""
        characters = {
            c.id: c for c in self.entity_clusters.values() 
            if c.entity_type == "character"
        }
        
        edges = defaultdict(lambda: defaultdict(int))
        for rel in self.relationships:
            if rel.subject_id in characters and rel.object_id in characters:
                edges[rel.subject_id][rel.object_id] += 1
        
        return {
            "nodes": [
                {"id": c.id, "name": c.canonical_name, "mentions": c.mention_count}
                for c in characters.values()
            ],
            "edges": [
                {"source": src, "target": tgt, "weight": count}
                for src, targets in edges.items()
                for tgt, count in targets.items()
            ],
        }


class GenericExtractor:
    """Extract entities and relationships from any novel without pre-seeding."""
    
    def __init__(self, use_llm: bool = True):
        """Initialize the generic extractor.
        
        Args:
            use_llm: Whether to use LLM for enhanced extraction and alias detection
        """
        self.settings = get_settings()
        self.use_llm = use_llm
        self.ner = NERPipeline(use_llm=use_llm)
    
    def analyze_book(
        self,
        file_path: Path,
        title: str | None = None,
        progress_callback=None,
    ) -> BookAnalysis:
        """Perform complete analysis of a book.
        
        Args:
            file_path: Path to the book file
            title: Optional title override
            progress_callback: Optional callback(phase, current, total, message)
            
        Returns:
            BookAnalysis with entities and relationships
        """
        book_title = title or file_path.stem.replace("_", " ").title()
        
        # Load and split text
        text = load_book(file_path)
        passages = split_into_passages(text, book_title)
        total_passages = len(passages)
        
        # Phase 1: Entity extraction with dynamic resolution
        if progress_callback:
            progress_callback("entities", 0, total_passages, "Extracting entities...")
        
        resolver = DynamicEntityResolver(use_llm=self.use_llm)
        passage_entities: dict[str, list[EntityCluster]] = {}
        
        for i, passage in enumerate(passages):
            # Extract entities from passage
            entities = self.ner.extract_entities(passage.text)
            
            # Process each mention with dynamic resolver
            clusters = []
            for entity in entities:
                cluster = resolver.process_mention(
                    entity=entity,
                    passage_id=passage.id,
                    passage_text=passage.text,
                )
                clusters.append(cluster)
            
            # Check for explicit alias statements
            resolver.detect_aliases_from_text(passage.text, passage.id)
            
            passage_entities[passage.id] = clusters
            
            if progress_callback and (i + 1) % 100 == 0:
                progress_callback("entities", i + 1, total_passages, 
                    f"Extracted {len(resolver.clusters)} unique entities...")
        
        # Consolidate aliases
        if progress_callback:
            progress_callback("consolidate", 0, 1, "Consolidating entity aliases...")
        
        merges = resolver.consolidate_clusters(min_cooccurrence=3)
        
        if progress_callback:
            progress_callback("consolidate", 1, 1, f"Merged {merges} entity pairs")
        
        # Phase 2: Relationship extraction
        if progress_callback:
            progress_callback("relationships", 0, total_passages, "Extracting relationships...")
        
        all_relationships = []
        relationship_counts: dict[str, int] = defaultdict(int)
        passage_results = []
        
        # Create a relationship extractor that uses our dynamic resolver
        rel_extractor = RelationshipExtractor(
            resolver=None,  # We'll handle resolution ourselves
            use_llm=self.use_llm,
        )
        
        for i, passage in enumerate(passages):
            clusters = passage_entities.get(passage.id, [])
            
            # Skip passages with <2 entities
            if len(clusters) < 2:
                passage_results.append(GenericExtractionResult(
                    passage=passage,
                    entities=clusters,
                    relationships=[],
                ))
                continue
            
            # Build fake ResolvedEntity objects for the relationship extractor
            from .resolver import ResolvedEntity
            from .ner import ExtractedEntity
            
            resolved_entities = []
            for cluster in clusters:
                if cluster.mentions:
                    mention = cluster.mentions[-1]  # Most recent mention in this passage
                    resolved_entities.append(ResolvedEntity(
                        extracted=ExtractedEntity(
                            text=mention.text,
                            label=mention.label,
                            start_char=mention.char_offset,
                            end_char=mention.char_offset + len(mention.text),
                        ),
                        canonical_id=cluster.id,
                        canonical_name=cluster.canonical_name,
                        entity_type=cluster.entity_type,
                        confidence=1.0,
                        is_new=False,
                    ))
            
            # Extract relationships
            rel_result = rel_extractor.extract_relationships(
                text=passage.text,
                passage_id=passage.id,
                entities=resolved_entities,
            )
            
            for rel in rel_result.relationships:
                all_relationships.append(rel)
                relationship_counts[rel.predicate.value] += 1
            
            passage_results.append(GenericExtractionResult(
                passage=passage,
                entities=clusters,
                relationships=rel_result.relationships,
            ))
            
            if progress_callback and (i + 1) % 100 == 0:
                progress_callback("relationships", i + 1, total_passages,
                    f"Found {len(all_relationships)} relationships...")
        
        return BookAnalysis(
            title=book_title,
            total_passages=total_passages,
            entity_clusters=resolver.clusters,
            total_mentions=sum(c.mention_count for c in resolver.clusters.values()),
            relationships=all_relationships,
            relationship_counts=dict(relationship_counts),
            passage_results=passage_results,
        )
    
    def analyze_text(
        self,
        text: str,
        title: str = "Unknown",
    ) -> BookAnalysis:
        """Analyze a text string (for testing).
        
        Args:
            text: The text to analyze
            title: Title for the text
            
        Returns:
            BookAnalysis results
        """
        # Split into passages
        passages = split_into_passages(text, title)
        
        # Use same logic as analyze_book
        resolver = DynamicEntityResolver(use_llm=self.use_llm)
        
        for passage in passages:
            entities = self.ner.extract_entities(passage.text)
            for entity in entities:
                resolver.process_mention(entity, passage.id, passage.text)
            resolver.detect_aliases_from_text(passage.text, passage.id)
        
        resolver.consolidate_clusters()
        
        return BookAnalysis(
            title=title,
            total_passages=len(passages),
            entity_clusters=resolver.clusters,
            total_mentions=sum(c.mention_count for c in resolver.clusters.values()),
            relationships=[],
            relationship_counts={},
        )
