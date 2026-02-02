"""Write entities and relationships to Neo4j."""

from typing import Iterator
from collections import defaultdict

from neo4j import Driver

from .connection import get_driver, init_schema
from ..extract.resolver import ResolvedEntity
from ..extract.relationships import RelationshipExtractionResult
from ..models.relationships import ExtractedRelationship, RelationshipTriple


class GraphWriter:
    """Writes extracted data to Neo4j graph database."""

    def __init__(self, driver: Driver | None = None):
        """Initialize the graph writer.

        Args:
            driver: Optional Neo4j driver (created if not provided)
        """
        self._driver = driver
        self._initialized = False

    @property
    def driver(self) -> Driver:
        """Get the Neo4j driver, creating if needed."""
        if self._driver is None:
            self._driver = get_driver()
            if self._driver is None:
                raise ConnectionError("Cannot connect to Neo4j")
        return self._driver

    def initialize(self) -> None:
        """Initialize the graph schema."""
        if not self._initialized:
            init_schema()
            self._initialized = True

    def write_entity(self, entity: ResolvedEntity, book: str) -> None:
        """Write a single entity to the graph.

        Args:
            entity: The resolved entity to write
            book: The book this entity was found in
        """
        if not entity.canonical_id:
            return  # Skip unresolved entities

        # Map entity type to Neo4j label
        label_map = {
            "character": "Character",
            "place": "Place",
            "object": "Object",
            "event": "Event",
        }
        label = label_map.get(entity.entity_type, "Entity")

        query = f"""
        MERGE (e:{label} {{id: $id}})
        ON CREATE SET
            e.canonical_name = $name,
            e.first_seen_book = $book,
            e.mention_count = 1
        ON MATCH SET
            e.mention_count = e.mention_count + 1
        """

        with self.driver.session() as session:
            session.run(
                query,
                id=entity.canonical_id,
                name=entity.canonical_name or entity.extracted.text,
                book=book,
            )

    def write_entities_batch(
        self,
        entities: list[ResolvedEntity],
        book: str,
    ) -> int:
        """Write multiple entities in a batch.

        Args:
            entities: List of resolved entities
            book: The book these entities were found in

        Returns:
            Number of entities written
        """
        # Group by type
        by_type: dict[str, list[ResolvedEntity]] = defaultdict(list)
        for entity in entities:
            if entity.canonical_id:
                by_type[entity.entity_type].append(entity)

        count = 0

        for entity_type, type_entities in by_type.items():
            label_map = {
                "character": "Character",
                "place": "Place",
                "object": "Object",
                "event": "Event",
            }
            label = label_map.get(entity_type, "Entity")

            # Prepare batch data
            batch_data = [
                {
                    "id": e.canonical_id,
                    "name": e.canonical_name or e.extracted.text,
                }
                for e in type_entities
            ]

            query = f"""
            UNWIND $batch AS item
            MERGE (e:{label} {{id: item.id}})
            ON CREATE SET
                e.canonical_name = item.name,
                e.first_seen_book = $book,
                e.mention_count = 1
            ON MATCH SET
                e.mention_count = e.mention_count + 1
            """

            with self.driver.session() as session:
                session.run(query, batch=batch_data, book=book)
                count += len(batch_data)

        return count

    def write_relationship(self, rel: ExtractedRelationship) -> None:
        """Write a single relationship to the graph.

        Args:
            rel: The extracted relationship to write
        """
        if not rel.subject_id or not rel.object_id:
            return  # Need both entities resolved

        # Create relationship with passage reference
        query = """
        MATCH (s {id: $subject_id})
        MATCH (o {id: $object_id})
        MERGE (s)-[r:""" + rel.predicate.value + """]->(o)
        ON CREATE SET
            r.first_passage = $passage_id,
            r.mention_count = 1,
            r.passages = [$passage_id]
        ON MATCH SET
            r.mention_count = r.mention_count + 1,
            r.passages = CASE 
                WHEN NOT $passage_id IN r.passages 
                THEN r.passages + $passage_id 
                ELSE r.passages 
            END
        """

        with self.driver.session() as session:
            session.run(
                query,
                subject_id=rel.subject_id,
                object_id=rel.object_id,
                passage_id=rel.passage_id,
            )

    def write_relationships_batch(
        self,
        relationships: list[ExtractedRelationship],
    ) -> int:
        """Write multiple relationships in a batch.

        Args:
            relationships: List of extracted relationships

        Returns:
            Number of relationships written
        """
        # Group by relationship type (Neo4j needs separate queries per type)
        by_type: dict[str, list[ExtractedRelationship]] = defaultdict(list)
        for rel in relationships:
            if rel.subject_id and rel.object_id:
                by_type[rel.predicate.value].append(rel)

        count = 0

        for rel_type, type_rels in by_type.items():
            batch_data = [
                {
                    "subject_id": r.subject_id,
                    "object_id": r.object_id,
                    "passage_id": r.passage_id,
                }
                for r in type_rels
            ]

            query = f"""
            UNWIND $batch AS item
            MATCH (s {{id: item.subject_id}})
            MATCH (o {{id: item.object_id}})
            MERGE (s)-[r:{rel_type}]->(o)
            ON CREATE SET
                r.first_passage = item.passage_id,
                r.mention_count = 1,
                r.passages = [item.passage_id]
            ON MATCH SET
                r.mention_count = r.mention_count + 1,
                r.passages = CASE 
                    WHEN NOT item.passage_id IN r.passages 
                    THEN r.passages + item.passage_id 
                    ELSE r.passages 
                END
            """

            with self.driver.session() as session:
                session.run(query, batch=batch_data)
                count += len(batch_data)

        return count

    def write_passage(
        self,
        passage_id: str,
        text: str,
        book: str,
        chapter_num: int,
        paragraph_num: int,
        sentence_num: int,
    ) -> None:
        """Write a passage node to the graph.

        Args:
            passage_id: Unique passage identifier
            text: The passage text
            book: Book title
            chapter_num: Chapter number
            paragraph_num: Paragraph number
            sentence_num: Sentence number
        """
        query = """
        MERGE (p:Passage {id: $id})
        ON CREATE SET
            p.text = $text,
            p.book = $book,
            p.chapter_num = $chapter_num,
            p.paragraph_num = $paragraph_num,
            p.sentence_num = $sentence_num
        """

        with self.driver.session() as session:
            session.run(
                query,
                id=passage_id,
                text=text[:500],  # Truncate for storage
                book=book,
                chapter_num=chapter_num,
                paragraph_num=paragraph_num,
                sentence_num=sentence_num,
            )

    def link_entity_to_passage(
        self,
        entity_id: str,
        passage_id: str,
    ) -> None:
        """Create a MENTIONED_IN relationship between entity and passage.

        Args:
            entity_id: The entity's canonical ID
            passage_id: The passage ID
        """
        query = """
        MATCH (e {id: $entity_id})
        MATCH (p:Passage {id: $passage_id})
        MERGE (e)-[r:MENTIONED_IN]->(p)
        ON CREATE SET r.count = 1
        ON MATCH SET r.count = r.count + 1
        """

        with self.driver.session() as session:
            session.run(query, entity_id=entity_id, passage_id=passage_id)

    def write_extraction_results(
        self,
        entity_results: list,  # List of ExtractionResult
        relationship_results: list[RelationshipExtractionResult],
        book: str,
        progress_callback=None,
    ) -> dict:
        """Write complete extraction results to the graph.

        Args:
            entity_results: Results from entity extraction
            relationship_results: Results from relationship extraction
            book: Book title
            progress_callback: Optional callback(step, total_steps, message)

        Returns:
            Stats dict with counts
        """
        self.initialize()

        stats = {
            "entities_written": 0,
            "relationships_written": 0,
            "passages_written": 0,
        }

        total_steps = 3
        current_step = 0

        # Step 1: Write entities
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, "Writing entities...")

        all_entities = []
        for result in entity_results:
            for entity in result.entities:
                if entity.canonical_id:
                    all_entities.append(entity)

        # Deduplicate by canonical_id
        unique_entities = {e.canonical_id: e for e in all_entities}
        stats["entities_written"] = self.write_entities_batch(
            list(unique_entities.values()),
            book,
        )

        # Step 2: Write relationships
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, "Writing relationships...")

        all_relationships = []
        for result in relationship_results:
            all_relationships.extend(result.relationships)

        stats["relationships_written"] = self.write_relationships_batch(all_relationships)

        # Step 3: Write passages with entity links
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, "Writing passages...")

        # Only write passages that have relationships (to save space)
        passage_ids_with_rels = {r.passage_id for r in relationship_results if r.relationships}

        for result in entity_results:
            if result.passage.id in passage_ids_with_rels:
                self.write_passage(
                    passage_id=result.passage.id,
                    text=result.passage.text,
                    book=result.passage.book,
                    chapter_num=result.passage.chapter_num,
                    paragraph_num=result.passage.paragraph_num,
                    sentence_num=result.passage.sentence_num,
                )
                stats["passages_written"] += 1

        return stats

    def close(self) -> None:
        """Close the driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
