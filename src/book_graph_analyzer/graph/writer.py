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

    # =========================================================================
    # Style Analysis Integration (Phase 4)
    # =========================================================================

    def write_book_style(
        self,
        book_id: str,
        title: str,
        author: str,
        fingerprint,  # AuthorStyleFingerprint
    ) -> None:
        """Write book node with style fingerprint data.

        Args:
            book_id: Unique book identifier
            title: Book title
            author: Author name
            fingerprint: AuthorStyleFingerprint object
        """
        query = """
        MERGE (b:Book {id: $id})
        SET b.title = $title,
            b.author = $author,
            b.total_words = $total_words,
            b.total_sentences = $total_sentences,
            b.avg_sentence_length = $avg_sentence_length,
            b.sentence_length_std = $sentence_length_std,
            b.flesch_reading_ease = $flesch_reading_ease,
            b.flesch_kincaid_grade = $flesch_kincaid_grade,
            b.gunning_fog = $gunning_fog,
            b.dialogue_ratio = $dialogue_ratio,
            b.passive_voice_ratio = $passive_voice_ratio,
            b.question_ratio = $question_ratio,
            b.exclamation_ratio = $exclamation_ratio,
            b.type_token_ratio = $type_token_ratio,
            b.archaism_density = $archaism_density
        """

        params = {
            "id": book_id,
            "title": title,
            "author": author,
            "total_words": fingerprint.total_word_count,
            "total_sentences": fingerprint.total_sentence_count,
            "avg_sentence_length": fingerprint.sentence_length_dist.mean if fingerprint.sentence_length_dist else 0,
            "sentence_length_std": fingerprint.sentence_length_dist.std if fingerprint.sentence_length_dist else 0,
            "flesch_reading_ease": fingerprint.flesch_reading_ease,
            "flesch_kincaid_grade": fingerprint.flesch_kincaid_grade,
            "gunning_fog": fingerprint.gunning_fog,
            "dialogue_ratio": fingerprint.dialogue_ratio,
            "passive_voice_ratio": fingerprint.passive_voice_ratio,
            "question_ratio": fingerprint.question_ratio,
            "exclamation_ratio": fingerprint.exclamation_ratio,
            "type_token_ratio": fingerprint.vocabulary_profile.type_token_ratio if fingerprint.vocabulary_profile else 0,
            "archaism_density": fingerprint.archaism_density,
        }

        with self.driver.session() as session:
            session.run(query, **params)

    def write_passage_style(
        self,
        passage_id: str,
        passage_type: str,
        word_count: int,
        has_dialogue: bool,
        book_id: str,
    ) -> None:
        """Update passage with style classification.

        Args:
            passage_id: Passage ID
            passage_type: Classification (dialogue, action, etc.)
            word_count: Word count
            has_dialogue: Whether passage contains dialogue
            book_id: Book ID for linking
        """
        query = """
        MERGE (p:Passage {id: $passage_id})
        SET p.passage_type = $passage_type,
            p.word_count = $word_count,
            p.has_dialogue = $has_dialogue
        WITH p
        MATCH (b:Book {id: $book_id})
        MERGE (b)-[:CONTAINS]->(p)
        """

        with self.driver.session() as session:
            session.run(
                query,
                passage_id=passage_id,
                passage_type=passage_type,
                word_count=word_count,
                has_dialogue=has_dialogue,
                book_id=book_id,
            )

    # =========================================================================
    # Voice Profile Integration (Phase 5)
    # =========================================================================

    def write_character_voice(
        self,
        character_id: str,
        profile,  # CharacterVoiceProfile
    ) -> None:
        """Update character node with voice profile data.

        Args:
            character_id: Character's canonical ID
            profile: CharacterVoiceProfile object
        """
        query = """
        MATCH (c:Character {id: $id})
        SET c.total_lines = $total_lines,
            c.total_dialogue_words = $total_words,
            c.avg_utterance_length = $avg_utterance_length,
            c.utterance_length_std = $utterance_length_std,
            c.question_ratio = $question_ratio,
            c.exclamation_ratio = $exclamation_ratio,
            c.vocabulary_richness = $vocabulary_richness,
            c.contraction_ratio = $contraction_ratio,
            c.distinctive_words = $distinctive_words,
            c.sample_quotes = $sample_quotes,
            c.archaisms_used = $archaisms_used
        """

        with self.driver.session() as session:
            session.run(
                query,
                id=character_id,
                total_lines=profile.total_lines,
                total_words=profile.total_words,
                avg_utterance_length=profile.avg_utterance_length,
                utterance_length_std=profile.utterance_length_std,
                question_ratio=profile.question_ratio,
                exclamation_ratio=profile.exclamation_ratio,
                vocabulary_richness=profile.type_token_ratio,
                contraction_ratio=profile.contraction_ratio,
                distinctive_words=profile.distinctive_words[:10],
                sample_quotes=profile.sample_quotes[:5],
                archaisms_used=profile.archaisms_used,
            )

    def write_dialogue_line(
        self,
        line_id: str,
        text: str,
        speaker_id: str,
        passage_id: str,
        is_question: bool,
        is_exclamation: bool,
    ) -> None:
        """Write a dialogue line and link to speaker and passage.

        Args:
            line_id: Unique ID for the dialogue line
            text: The dialogue text
            speaker_id: Character ID of the speaker
            passage_id: Passage ID where this appears
            is_question: Whether it's a question
            is_exclamation: Whether it's an exclamation
        """
        query = """
        MERGE (d:DialogueLine {id: $line_id})
        SET d.text = $text,
            d.is_question = $is_question,
            d.is_exclamation = $is_exclamation,
            d.word_count = $word_count
        WITH d
        MATCH (c:Character {id: $speaker_id})
        MERGE (c)-[:SPEAKS]->(d)
        WITH d
        MATCH (p:Passage {id: $passage_id})
        MERGE (d)-[:IN_PASSAGE]->(p)
        """

        with self.driver.session() as session:
            session.run(
                query,
                line_id=line_id,
                text=text[:500],  # Truncate
                is_question=is_question,
                is_exclamation=is_exclamation,
                word_count=len(text.split()),
                speaker_id=speaker_id,
                passage_id=passage_id,
            )

    def write_voice_analysis_results(
        self,
        voice_result,  # VoiceAnalysisResult
        book_id: str,
        entity_id_map: dict[str, str],  # speaker_name -> canonical_id
        progress_callback=None,
    ) -> dict:
        """Write complete voice analysis results to the graph.

        Args:
            voice_result: VoiceAnalysisResult from voice analyzer
            book_id: Book ID for linking
            entity_id_map: Map from speaker names to canonical entity IDs
            progress_callback: Optional callback(step, total, message)

        Returns:
            Stats dict
        """
        stats = {
            "profiles_written": 0,
            "dialogue_lines_written": 0,
        }

        total_profiles = len(voice_result.profiles)

        # Write voice profiles to character nodes
        for i, (name, profile) in enumerate(voice_result.profiles.items()):
            if progress_callback:
                progress_callback(i + 1, total_profiles, f"Writing {name} profile...")

            # Try to find canonical ID
            char_id = entity_id_map.get(name) or entity_id_map.get(name.lower())
            if not char_id:
                # Create a simple ID if not mapped
                char_id = f"char_{name.lower().replace(' ', '_')}"
                # Ensure character node exists
                self._ensure_character_exists(char_id, name)

            self.write_character_voice(char_id, profile)
            stats["profiles_written"] += 1

        return stats

    def _ensure_character_exists(self, char_id: str, name: str) -> None:
        """Ensure a character node exists."""
        query = """
        MERGE (c:Character {id: $id})
        ON CREATE SET c.canonical_name = $name
        """
        with self.driver.session() as session:
            session.run(query, id=char_id, name=name)

    # =========================================================================
    # Event Graph Integration (Phase 6+)
    # =========================================================================

    def write_event(
        self,
        event,  # Event from lore.events
        book: str,
    ) -> None:
        """Write a single event to the graph.

        Args:
            event: Event object from lore.events module
            book: The book this event was found in
        """
        query = """
        MERGE (e:Event {id: $id})
        SET e.description = $description,
            e.agent = $agent,
            e.action = $action,
            e.patient = $patient,
            e.era = $era,
            e.year = $year,
            e.source_book = $book,
            e.confidence = $confidence
        """

        with self.driver.session() as session:
            session.run(
                query,
                id=event.id,
                description=event.description,
                agent=event.agent,
                action=event.action,
                patient=event.patient,
                era=event.era.value if event.era else None,
                year=event.year,
                book=book,
                confidence=event.confidence,
            )

    def write_events_batch(
        self,
        events: list,  # List of Event objects
        book: str,
    ) -> int:
        """Write multiple events in a batch.

        Args:
            events: List of Event objects
            book: The book these events were found in

        Returns:
            Number of events written
        """
        if not events:
            return 0

        batch_data = [
            {
                "id": e.id,
                "description": e.description,
                "agent": e.agent,
                "action": e.action,
                "patient": e.patient,
                "era": e.era.value if e.era else None,
                "year": e.year,
                "confidence": e.confidence,
            }
            for e in events
        ]

        query = """
        UNWIND $batch AS item
        MERGE (e:Event {id: item.id})
        SET e.description = item.description,
            e.agent = item.agent,
            e.action = item.action,
            e.patient = item.patient,
            e.era = item.era,
            e.year = item.year,
            e.source_book = $book,
            e.confidence = item.confidence
        """

        with self.driver.session() as session:
            session.run(query, batch=batch_data, book=book)

        return len(batch_data)

    def write_event_relations_batch(
        self,
        relations: list,  # List of EventRelation objects
    ) -> int:
        """Write event temporal relations.

        Args:
            relations: List of EventRelation objects

        Returns:
            Number of relations written
        """
        if not relations:
            return 0

        # Group by relation type
        by_type: dict[str, list] = defaultdict(list)
        for rel in relations:
            # Sanitize relation type for Neo4j (no spaces, only alphanumeric + underscore)
            rel_type = rel.relation.upper().replace(" ", "_")
            rel_type = "".join(c if c.isalnum() or c == "_" else "_" for c in rel_type)
            by_type[rel_type].append(rel)

        count = 0

        for rel_type, type_rels in by_type.items():
            batch_data = [
                {
                    "event1_id": r.event1_id,
                    "event2_id": r.event2_id,
                    "confidence": r.confidence,
                }
                for r in type_rels
            ]

            query = f"""
            UNWIND $batch AS item
            MATCH (e1:Event {{id: item.event1_id}})
            MATCH (e2:Event {{id: item.event2_id}})
            MERGE (e1)-[r:{rel_type}]->(e2)
            SET r.confidence = item.confidence
            """

            with self.driver.session() as session:
                session.run(query, batch=batch_data)
                count += len(batch_data)

        return count

    def link_event_to_entities(
        self,
        event,  # Event object
    ) -> int:
        """Link an event to its participant entities.

        Creates PARTICIPATED_IN relationships between characters and events.

        Args:
            event: Event object with agent/patient

        Returns:
            Number of links created
        """
        links = 0

        if event.agent:
            # Try to find matching character
            query = """
            MATCH (c:Character)
            WHERE toLower(c.canonical_name) CONTAINS toLower($name)
               OR toLower(c.name) CONTAINS toLower($name)
            WITH c LIMIT 1
            MATCH (e:Event {id: $event_id})
            MERGE (c)-[r:PARTICIPATED_IN]->(e)
            SET r.role = 'agent'
            RETURN count(r) as cnt
            """
            with self.driver.session() as session:
                result = session.run(query, name=event.agent, event_id=event.id)
                record = result.single()
                if record:
                    links += record["cnt"]

        if event.patient:
            # Try to find matching entity (could be character, place, or object)
            for label in ["Character", "Place", "Object"]:
                query = f"""
                MATCH (n:{label})
                WHERE toLower(n.canonical_name) CONTAINS toLower($name)
                   OR toLower(n.name) CONTAINS toLower($name)
                WITH n LIMIT 1
                MATCH (e:Event {{id: $event_id}})
                MERGE (n)-[r:INVOLVED_IN]->(e)
                SET r.role = 'patient'
                RETURN count(r) as cnt
                """
                with self.driver.session() as session:
                    result = session.run(query, name=event.patient, event_id=event.id)
                    record = result.single()
                    if record and record["cnt"] > 0:
                        links += record["cnt"]
                        break  # Found a match, stop looking

        return links

    def write_event_graph(
        self,
        event_graph,  # EventGraph object
        book: str,
        link_entities: bool = True,
        progress_callback=None,
    ) -> dict:
        """Write a complete event graph to Neo4j.

        Args:
            event_graph: EventGraph from lore.events
            book: Book title
            link_entities: Whether to link events to existing entities
            progress_callback: Optional callback(step, total, message)

        Returns:
            Stats dict
        """
        self.initialize()

        stats = {
            "events_written": 0,
            "relations_written": 0,
            "entity_links": 0,
        }

        total_steps = 3 if link_entities else 2
        current_step = 0

        # Step 1: Write events
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, "Writing events...")

        events = list(event_graph.events.values())
        stats["events_written"] = self.write_events_batch(events, book)

        # Step 2: Write relations
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, "Writing temporal relations...")

        stats["relations_written"] = self.write_event_relations_batch(event_graph.relations)

        # Step 3: Link to entities
        if link_entities:
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, "Linking to entities...")

            for event in events:
                stats["entity_links"] += self.link_event_to_entities(event)

        return stats

    def query_events(
        self,
        agent: str | None = None,
        action: str | None = None,
        patient: str | None = None,
        era: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query events from Neo4j.

        Args:
            agent: Filter by agent name (fuzzy match)
            action: Filter by action verb (fuzzy match)
            patient: Filter by patient/object (fuzzy match)
            era: Filter by era (exact match)
            limit: Maximum results

        Returns:
            List of event dicts
        """
        conditions = []
        params = {"limit": limit}

        if agent:
            conditions.append("toLower(e.agent) CONTAINS toLower($agent)")
            params["agent"] = agent

        if action:
            conditions.append("toLower(e.action) CONTAINS toLower($action)")
            params["action"] = action

        if patient:
            conditions.append("toLower(e.patient) CONTAINS toLower($patient)")
            params["patient"] = patient

        if era:
            conditions.append("e.era = $era")
            params["era"] = era

        where_clause = " AND ".join(conditions) if conditions else "true"

        query = f"""
        MATCH (e:Event)
        WHERE {where_clause}
        RETURN e.id as id, e.description as description, e.agent as agent,
               e.action as action, e.patient as patient, e.era as era,
               e.year as year, e.source_book as source_book,
               e.confidence as confidence
        ORDER BY e.era, e.year
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def query_event_ordering(
        self,
        event1_desc: str,
        event2_desc: str,
    ) -> dict | None:
        """Query the ordering relationship between two events.

        Args:
            event1_desc: Description or agent+action of first event
            event2_desc: Description or agent+action of second event

        Returns:
            Dict with ordering info, or None if not found
        """
        query = """
        MATCH (e1:Event), (e2:Event)
        WHERE toLower(e1.description) CONTAINS toLower($desc1)
           OR (toLower(e1.agent) CONTAINS toLower($desc1) AND e1.agent IS NOT NULL)
        WITH e1
        MATCH (e2:Event)
        WHERE toLower(e2.description) CONTAINS toLower($desc2)
           OR (toLower(e2.agent) CONTAINS toLower($desc2) AND e2.agent IS NOT NULL)
        OPTIONAL MATCH (e1)-[r]->(e2)
        WHERE type(r) IN ['BEFORE', 'AFTER', 'DURING', 'CAUSES']
        RETURN e1.id as event1_id, e1.description as event1,
               e2.id as event2_id, e2.description as event2,
               type(r) as relation,
               e1.era as era1, e1.year as year1,
               e2.era as era2, e2.year as year2
        LIMIT 1
        """

        with self.driver.session() as session:
            result = session.run(query, desc1=event1_desc, desc2=event2_desc)
            record = result.single()

            if not record:
                return None

            ordering = record["relation"]

            # If no direct relation, try to infer from era/year
            if not ordering:
                era1 = record["era1"]
                era2 = record["era2"]
                year1 = record["year1"]
                year2 = record["year2"]

                if era1 and era2:
                    era_order = {
                        "first_age": 1,
                        "second_age": 2,
                        "third_age": 3,
                        "fourth_age": 4,
                    }
                    if era_order.get(era1, 0) < era_order.get(era2, 0):
                        ordering = "BEFORE"
                    elif era_order.get(era1, 0) > era_order.get(era2, 0):
                        ordering = "AFTER"
                    elif year1 and year2:
                        if year1 < year2:
                            ordering = "BEFORE"
                        elif year1 > year2:
                            ordering = "AFTER"

            return {
                "event1": record["event1"],
                "event2": record["event2"],
                "relation": ordering,
                "era1": record["era1"],
                "year1": record["year1"],
                "era2": record["era2"],
                "year2": record["year2"],
            }
