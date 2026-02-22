"""Write entities and relationships to Neo4j."""

from typing import Iterator
from collections import defaultdict
import re

from neo4j import Driver

from .connection import get_driver, init_schema
from .temporal import TemporalValidity, ERA_ORDER, canonicalize_era
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

    def init_era_chain(self) -> None:
        """Create Era nodes and FOLLOWED_BY chain in the correct historical order.

        This is idempotent — safe to call multiple times.
        Era nodes enable temporal ordering queries in Cypher using
        the era_order property instead of a hardcoded lookup map.
        """
        eras = [
            ("Before Time",        0),
            ("Years of the Lamps", 1),
            ("Years of the Trees", 2),
            ("First Age",          3),
            ("Second Age",         4),
            ("Third Age",          5),
            ("Fourth Age",         6),
        ]

        with self.driver.session() as session:
            # Create / update each Era node
            for name, order in eras:
                session.run(
                    "MERGE (e:Era {name: $name}) SET e.era_order = $order",
                    name=name, order=order,
                )

            # Create FOLLOWED_BY chain
            for i in range(len(eras) - 1):
                session.run(
                    """
                    MATCH (a:Era {name: $a})
                    MATCH (b:Era {name: $b})
                    MERGE (a)-[:FOLLOWED_BY]->(b)
                    """,
                    a=eras[i][0], b=eras[i + 1][0],
                )

    def query_at_time(
        self,
        character_name: str,
        era: str,
        year: int | None = None,
    ) -> dict:
        """Return a snapshot of everything known about a character at a given point in time.

        Queries all relationships that are valid at era/year and returns:
            - people they know (KNOWS / MET / FRIEND_OF / ENEMY_OF etc.)
            - places they are / have been
            - objects they possess(ed)
            - events they participated in
            - emotional state (if available)

        Args:
            character_name: Canonical name or alias
            era: Era string e.g. 'Third Age'
            year: Optional specific year within the era

        Returns:
            Dict with keys: character, knows, places, objects, events, emotional_state
        """
        from .temporal import point_in_time_cypher_where

        era_filter = point_in_time_cypher_where("r", "$era", "$year")

        with self.driver.session() as session:

            # Find the character node
            char_row = session.run(
                "MATCH (c) WHERE c.canonical_name = $name OR $name IN coalesce(c.aliases, []) "
                "RETURN c LIMIT 1",
                name=character_name,
            ).single()

            if not char_row:
                return {"error": f"Character '{character_name}' not found"}

            char = dict(char_row["c"])

            # Relationships to other entities
            knows_rows = session.run(
                f"""
                MATCH (c {{canonical_name: $name}})-[r]->(other)
                WHERE {era_filter}
                  AND NOT other:Event AND NOT other:Passage
                RETURN type(r) as rel, other.canonical_name as name,
                       other.aliases as aliases, labels(other)[0] as type,
                       r.era_start as era_start, r.era_end as era_end,
                       r.year_start as year_start, r.year_end as year_end
                LIMIT 50
                """,
                name=character_name, era=era, year=year,
            )
            relationships = [dict(r) for r in knows_rows]

            # Events participated in during this era
            events_rows = session.run(
                """
                MATCH (c {canonical_name: $name})-[:PARTICIPATED_IN]->(e:Event)
                WHERE (e.era = $era OR e.era IS NULL)
                RETURN e.description as description, e.era as era,
                       e.year as year, e.agent as agent
                LIMIT 20
                """,
                name=character_name, era=era,
            )
            events = [dict(r) for r in events_rows]

        return {
            "character": char,
            "at": {"era": era, "year": year},
            "relationships": relationships,
            "events": events,
        }

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

        # Build temporal props from the relationship model
        temporal = TemporalValidity(
            era_start=canonicalize_era(getattr(rel, "era_start", None)),
            era_end=canonicalize_era(getattr(rel, "era_end", None)),
            year_start=getattr(rel, "year_start", None),
            year_end=getattr(rel, "year_end", None),
            source_passage_id=rel.passage_id,
        )
        temporal_props = temporal.to_dict()

        # Create relationship with passage reference + temporal validity
        set_clauses = ["r.first_passage = $passage_id", "r.mention_count = 1",
                       "r.passages = [$passage_id]"]
        for k in temporal_props:
            set_clauses.append(f"r.{k} = ${k}")

        query = f"""
        MATCH (s {{id: $subject_id}})
        MATCH (o {{id: $object_id}})
        MERGE (s)-[r:{rel.predicate.value}]->(o)
        ON CREATE SET {", ".join(set_clauses)}
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
                **temporal_props,
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
                    # Temporal validity fields
                    "era_start": canonicalize_era(getattr(r, "era_start", None)),
                    "era_end":   canonicalize_era(getattr(r, "era_end", None)),
                    "year_start": getattr(r, "year_start", None),
                    "year_end":   getattr(r, "year_end", None),
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
                r.passages = [item.passage_id],
                r.era_start  = item.era_start,
                r.era_end    = item.era_end,
                r.year_start = item.year_start,
                r.year_end   = item.year_end
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
        source_id: str | None = None,
        source_title: str | None = None,
        source_stratum: str | None = None,
        source_authority_weight: float | None = None,
        provenance_tags: list[str] | None = None,
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
        SET p.source_id = coalesce($source_id, p.source_id),
            p.source_title = coalesce($source_title, p.source_title),
            p.source_stratum = coalesce($source_stratum, p.source_stratum),
            p.source_authority_weight = coalesce($source_authority_weight, p.source_authority_weight),
            p.provenance_tags = coalesce($provenance_tags, p.provenance_tags)
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
                source_id=source_id,
                source_title=source_title,
                source_stratum=source_stratum,
                source_authority_weight=source_authority_weight,
                provenance_tags=provenance_tags,
            )

    def write_passage_provenance(
        self,
        passage_id: str,
        source_id: str,
        source_title: str,
        source_stratum: str = "core_text",
        authority_weight: float = 1.0,
        confidence: float = 1.0,
    ) -> None:
        """Attach a Passage to a Source and stratum metadata for layer-aware queries."""
        query = """
        MATCH (p:Passage {id: $passage_id})
        MERGE (s:Source {id: $source_id})
        SET s.source_title = $source_title,
            s.authority_weight = $authority_weight
        MERGE (p)-[r:ATTESTED_IN]->(s)
        SET r.source_stratum = $source_stratum,
            r.confidence = $confidence
        """
        with self.driver.session() as session:
            session.run(
                query,
                passage_id=passage_id,
                source_id=source_id,
                source_title=source_title,
                source_stratum=source_stratum,
                authority_weight=authority_weight,
                confidence=max(0.0, min(1.0, float(confidence))),
            )

    def query_layer_report(self, source_id: str | None = None, limit: int = 200) -> list[dict]:
        """Summarize passages by source and stratum for reporting."""
        query = """
        MATCH (p:Passage)
        WHERE $source_id IS NULL OR p.source_id = $source_id
        RETURN coalesce(p.source_id, p.book) AS source,
               coalesce(p.source_stratum, 'core_text') AS stratum,
               count(*) AS passage_count,
               avg(coalesce(p.source_authority_weight, 1.0)) AS avg_authority
        ORDER BY source, stratum
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, source_id=source_id, limit=limit)
            return [dict(r) for r in result]

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
                    source_id=getattr(result.passage, "source_id", None),
                    source_title=getattr(result.passage, "source_title", None),
                    source_stratum=getattr(result.passage, "source_stratum", None),
                    source_authority_weight=getattr(result.passage, "source_authority_weight", None),
                    provenance_tags=getattr(result.passage, "provenance_tags", None),
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

    @staticmethod
    def _resolve_event_book(event, fallback_book: str) -> str:
        """Resolve source book for an event write operation."""
        event_book = getattr(event, "source_book", None)
        return (event_book or fallback_book or "").strip()

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
        MERGE (e:Event {id: $id, source_book: $book})
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
                book=self._resolve_event_book(event, book),
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
                "source_text": e.source_text,
                "era": e.era.value if e.era else None,
                "year": e.year,
                "confidence": e.confidence,
                "source_book": self._resolve_event_book(e, book),
            }
            for e in events
        ]

        query = """
        UNWIND $batch AS item
        MERGE (e:Event {id: item.id, source_book: item.source_book})
        SET e.description = item.description,
            e.agent = item.agent,
            e.action = item.action,
            e.patient = item.patient,
            e.source_text = item.source_text,
            e.era = item.era,
            e.year = item.year,
            e.confidence = item.confidence
        """

        with self.driver.session() as session:
            session.run(query, batch=batch_data)

        return len(batch_data)

    def write_event_relations_batch(
        self,
        relations: list,  # List of EventRelation objects
        event_book_by_id: dict[str, str] | None = None,
        default_book: str | None = None,
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
                    "event1_book": (event_book_by_id or {}).get(r.event1_id, default_book or ""),
                    "event2_book": (event_book_by_id or {}).get(r.event2_id, default_book or ""),
                    "confidence": r.confidence,
                }
                for r in type_rels
            ]

            query = f"""
            UNWIND $batch AS item
            MATCH (e1:Event {{id: item.event1_id, source_book: item.event1_book}})
            MATCH (e2:Event {{id: item.event2_id, source_book: item.event2_book}})
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
        book: str,
    ) -> int:
        """Link an event to its participant entities.

        Creates PARTICIPATED_IN relationships between characters and events.

        Args:
            event: Event object with agent/patient

        Returns:
            Number of links created
        """
        links = 0

        links += self._link_event_role(
            event_id=event.id,
            source_book=self._resolve_event_book(event, book),
            raw_value=event.agent,
            labels=["Character"],
            rel_type="PARTICIPATED_IN",
            role="agent",
        )

        # Patient may be Character / Place / Object. Choose one best hit only.
        links += self._link_event_role(
            event_id=event.id,
            source_book=self._resolve_event_book(event, book),
            raw_value=event.patient,
            labels=["Character", "Place", "Object"],
            rel_type="INVOLVED_IN",
            role="patient",
        )

        # Conservative place cue fallback from event prose if no explicit patient match.
        if not event.patient and (event.description or event.source_text):
            cue_text = f"{event.description or ''} {event.source_text or ''}".strip()
            links += self._link_event_role(
                event_id=event.id,
                source_book=self._resolve_event_book(event, book),
                raw_value=cue_text,
                labels=["Place"],
                rel_type="TOOK_PLACE_AT",
                role="location_cue",
            )

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

        event_book_by_id = {e.id: self._resolve_event_book(e, book) for e in events}
        stats["relations_written"] = self.write_event_relations_batch(
            event_graph.relations,
            event_book_by_id=event_book_by_id,
            default_book=book,
        )

        # Step 3: Link to entities
        if link_entities:
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, "Linking to entities...")

            for event in events:
                stats["entity_links"] += self.link_event_to_entities(event, book=book)

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

    # =========================================================================
    # World-Building Layer Stubs (Issue #45 — Tolkien World-Building Kickoff)
    # =========================================================================

    def write_linguistic_lineage(
        self,
        lineage,  # LinguisticLineage
    ) -> int:
        """Write a linguistic lineage (etymology chain) to the graph.

        Creates LanguageForm nodes and DERIVED_FROM relationships.
        Idempotent: uses MERGE so repeated calls are safe.

        Optionally links each LanguageForm to an existing entity node
        (Character/Place/Object/Entity) via HAS_NAME if entity_id is set.

        Args:
            lineage: LinguisticLineage object with forms and derivations

        Returns:
            Number of forms written
        """
        if not lineage or not lineage.forms:
            return 0

        count = 0

        with self.driver.session() as session:
            # Step 1: MERGE LanguageForm nodes
            for form in lineage.forms:
                session.run(
                    """
                    MERGE (lf:LanguageForm {id: $id})
                    SET lf.form = $form,
                        lf.language = $language,
                        lf.entity_id = $entity_id,
                        lf.gloss = $gloss,
                        lf.phonetic = $phonetic,
                        lf.source_passage_id = $source_passage_id
                    """,
                    id=form.id,
                    form=form.form,
                    language=form.language.value if hasattr(form.language, "value") else str(form.language),
                    entity_id=form.entity_id,
                    gloss=form.gloss,
                    phonetic=form.phonetic,
                    source_passage_id=form.source_passage_id,
                )
                count += 1

                # Optional: link to existing entity node via HAS_NAME
                if form.entity_id:
                    session.run(
                        """
                        MATCH (e {id: $entity_id})
                        MATCH (lf:LanguageForm {id: $form_id})
                        MERGE (e)-[:HAS_NAME]->(lf)
                        """,
                        entity_id=form.entity_id,
                        form_id=form.id,
                    )

            # Step 2: MERGE DERIVED_FROM edges
            for deriv in lineage.derivations:
                session.run(
                    """
                    MATCH (src:LanguageForm {id: $source_id})
                    MATCH (tgt:LanguageForm {id: $target_id})
                    MERGE (src)-[r:DERIVED_FROM]->(tgt)
                    SET r.derivation_type = $dtype,
                        r.notes = $notes
                    """,
                    source_id=deriv.source_form_id,
                    target_id=deriv.target_form_id,
                    dtype=deriv.derivation_type.value if hasattr(deriv.derivation_type, "value") else str(deriv.derivation_type),
                    notes=deriv.notes,
                )

        return count

    def write_linguistic_lineage_batch(
        self,
        lineages: list,  # list[LinguisticLineage]
    ) -> int:
        """Write multiple linguistic lineages in a batch.

        Args:
            lineages: List of LinguisticLineage objects

        Returns:
            Total number of forms written
        """
        total = 0
        for lineage in lineages:
            total += self.write_linguistic_lineage(lineage)
        return total

    def query_linguistic_lineage(
        self,
        entity_id: str | None = None,
        language: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Query linguistic forms and their derivation chains.

        Args:
            entity_id: Filter by entity ID
            language: Filter by language name
            limit: Maximum results

        Returns:
            List of dicts with form info and derivation chains
        """
        conditions = []
        params: dict = {"limit": limit}

        if entity_id:
            conditions.append("lf.entity_id = $entity_id")
            params["entity_id"] = entity_id

        if language:
            conditions.append("lf.language = $language")
            params["language"] = language

        where = " AND ".join(conditions) if conditions else "true"

        query = f"""
        MATCH (lf:LanguageForm)
        WHERE {where}
        OPTIONAL MATCH (lf)-[r:DERIVED_FROM]->(parent:LanguageForm)
        RETURN lf.id as id, lf.form as form, lf.language as language,
               lf.entity_id as entity_id, lf.gloss as gloss,
               lf.phonetic as phonetic,
               parent.id as derived_from_id, parent.form as derived_from_form,
               parent.language as derived_from_language,
               r.derivation_type as derivation_type
        ORDER BY lf.entity_id, lf.language
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def write_genealogy_batch(
        self,
        relations: list,  # list[GenealogyRelation]
        book: str = "",
    ) -> int:
        """Write genealogy relationships with generational metadata.

        Creates typed family edges (PARENT_OF, ANCESTOR_OF, etc.) with
        generation_depth, house, and inheritance_traits properties.

        TODO(#47): Implement Cypher for genealogy edges with depth/house metadata
        TODO(#47): Add inverse-relation auto-creation (PARENT_OF ↔ CHILD_OF)

        Args:
            relations: List of GenealogyRelation objects
            book: Source book for provenance

        Returns:
            Number of relations written
        """
        if not relations:
            return 0

        query = """
        MERGE (a:Character {canonical_id: $source_id})
          ON CREATE SET a.canonical_name = coalesce($source_name, $source_id)
          ON MATCH SET a.canonical_name = coalesce(a.canonical_name, $source_name, $source_id)
        MERGE (b:Character {canonical_id: $target_id})
          ON CREATE SET b.canonical_name = coalesce($target_name, $target_id)
          ON MATCH SET b.canonical_name = coalesce(b.canonical_name, $target_name, $target_id)
        MERGE (a)-[r:GENEALOGY {relation_type: $relation_type}]->(b)
        SET r.generation_depth = $generation_depth,
            r.house = $house,
            r.inheritance_traits = $inheritance_traits,
            r.era = $era,
            r.passage_ids = $passage_ids,
            r.confidence = $confidence,
            r.book = $book,
            r.updated_at = datetime()
        """

        count = 0
        with self.driver.session() as session:
            for rel in relations:
                session.run(
                    query,
                    source_id=rel.source_id,
                    source_name=getattr(rel, "source_name", None),
                    target_id=rel.target_id,
                    target_name=getattr(rel, "target_name", None),
                    relation_type=rel.relation_type.value if hasattr(rel.relation_type, "value") else str(rel.relation_type),
                    generation_depth=getattr(rel, "generation_depth", None),
                    house=getattr(rel, "house", None),
                    inheritance_traits=list(getattr(rel, "inheritance_traits", []) or []),
                    era=getattr(rel, "era", None),
                    passage_ids=list(getattr(rel, "passage_ids", []) or []),
                    confidence=float(getattr(rel, "confidence", 1.0) or 1.0),
                    book=book,
                )
                count += 1

        return count

    def query_genealogy(
        self,
        character_name: str | None = None,
        house: str | None = None,
        depth: int = 3,
        limit: int = 200,
    ) -> list[dict]:
        """Query genealogy edges by character and/or house.

        Returns flattened relationship rows for CLI rendering.
        """
        conditions = []
        params: dict = {"depth": depth, "limit": limit}

        if character_name:
            conditions.append(
                "(toLower(a.canonical_name) CONTAINS toLower($character_name) OR toLower(b.canonical_name) CONTAINS toLower($character_name))"
            )
            params["character_name"] = character_name

        if house:
            conditions.append("toLower(coalesce(r.house,'')) CONTAINS toLower($house)")
            params["house"] = house

        where = " AND ".join(conditions) if conditions else "true"

        query = f"""
        MATCH (a:Character)-[r:GENEALOGY]->(b:Character)
        WHERE {where}
          AND ($depth IS NULL OR r.generation_depth IS NULL OR r.generation_depth <= $depth)
        RETURN a.canonical_name AS source,
               r.relation_type AS rel,
               b.canonical_name AS target,
               r.house AS house,
               r.generation_depth AS generation_depth,
               r.inheritance_traits AS inheritance_traits,
               r.confidence AS confidence,
               r.book AS book
        ORDER BY source, rel, target
        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def write_editorial_provenance(
        self,
        entity_id: str,
        source,  # EditorialLayer
        confidence: float = 1.0,
        page_ref: str | None = None,
    ) -> None:
        """Link an entity to its editorial source via ATTESTED_IN relationship.

        Creates a (:Source) node if needed and an ATTESTED_IN edge.
        """
        if not entity_id or source is None:
            return

        source_props = {
            "source_id": getattr(source, "source_id", None),
            "source_title": getattr(source, "source_title", None),
            "editorial_status": getattr(getattr(source, "editorial_status", None), "value", None)
            or str(getattr(source, "editorial_status", ""))
            or None,
            "author_period": getattr(getattr(source, "author_period", None), "value", None)
            or str(getattr(source, "author_period", ""))
            or None,
            "publication_year": getattr(source, "publication_year", None),
            "editor": getattr(source, "editor", None),
            "volume": getattr(source, "volume", None),
            "authority_weight": float(getattr(source, "authority_weight", 1.0) or 1.0),
            "notes": getattr(source, "notes", None),
        }

        source_id = source_props["source_id"] or source_props["source_title"]
        if not source_id:
            return

        query = """
        MATCH (e {id: $entity_id})
        MERGE (s:Source {id: $source_id})
        SET s.source_title = $source_title,
            s.editorial_status = $editorial_status,
            s.author_period = $author_period,
            s.publication_year = $publication_year,
            s.editor = $editor,
            s.volume = $volume,
            s.authority_weight = $authority_weight,
            s.notes = $notes
        MERGE (e)-[r:ATTESTED_IN]->(s)
        SET r.confidence = $confidence,
            r.page_ref = $page_ref
        """

        with self.driver.session() as session:
            session.run(
                query,
                entity_id=entity_id,
                source_id=source_id,
                source_title=source_props["source_title"],
                editorial_status=source_props["editorial_status"],
                author_period=source_props["author_period"],
                publication_year=source_props["publication_year"],
                editor=source_props["editor"],
                volume=source_props["volume"],
                authority_weight=source_props["authority_weight"],
                notes=source_props["notes"],
                confidence=max(0.0, min(1.0, float(confidence))),
                page_ref=page_ref,
            )

    # =========================================================================
    # Lore Depth Engine (Issue #50 slice 1)
    # =========================================================================

    def write_lore_artifacts_batch(self, artifacts: list, book: str = "") -> int:
        """Persist lore artifacts (songs/poems/artifacts) as first-class nodes."""
        if not artifacts:
            return 0

        with self.driver.session() as session:
            for art in artifacts:
                session.run(
                    """
                    MERGE (a:LoreArtifact {id: $id})
                    SET a.name = $name,
                        a.artifact_type = $artifact_type,
                        a.description = $description,
                        a.source_book = $source_book,
                        a.confidence = $confidence,
                        a.updated_at = datetime()
                    """,
                    id=art.id,
                    name=art.name,
                    artifact_type=getattr(getattr(art, "artifact_type", None), "value", None)
                    or str(getattr(art, "artifact_type", "artifact")),
                    description=getattr(art, "description", None),
                    source_book=getattr(art, "source_book", None) or book,
                    confidence=float(getattr(art, "confidence", 0.7) or 0.7),
                )
                if getattr(art, "passage_id", None):
                    session.run(
                        """
                        MATCH (a:LoreArtifact {id: $artifact_id})
                        MATCH (p:Passage {id: $passage_id})
                        MERGE (a)-[:ATTESTED_IN]->(p)
                        """,
                        artifact_id=art.id,
                        passage_id=art.passage_id,
                    )
        return len(artifacts)

    def write_broken_references_batch(self, refs: list) -> int:
        """Persist unresolved references for later curation."""
        if not refs:
            return 0

        with self.driver.session() as session:
            for ref in refs:
                session.run(
                    """
                    MERGE (u:UnresolvedReference {id: $id})
                    SET u.mention_text = $mention_text,
                        u.context_text = $context_text,
                        u.context_before = $context_before,
                        u.context_after = $context_after,
                        u.expected_type = $expected_type,
                        u.source_book = $source_book,
                        u.passage_id = $passage_id,
                        u.resolved_entity_id = $resolved_entity_id,
                        u.confidence = $confidence,
                        u.candidates = $candidates,
                        u.provenance_notes = $provenance_notes,
                        u.conflict_weight = $conflict_weight,
                        u.updated_at = datetime()
                    """,
                    id=ref.id,
                    mention_text=getattr(ref, "mention_text", ""),
                    context_text=getattr(ref, "context_text", None),
                    context_before=getattr(ref, "context_before", None),
                    context_after=getattr(ref, "context_after", None),
                    expected_type=getattr(ref, "expected_type", None),
                    source_book=getattr(ref, "source_book", None),
                    passage_id=getattr(ref, "passage_id", None),
                    resolved_entity_id=getattr(ref, "resolved_entity_id", None),
                    confidence=float(getattr(ref, "confidence", 0.6) or 0.6),
                    candidates=[c.model_dump() if hasattr(c, "model_dump") else c for c in (getattr(ref, "candidates", None) or [])],
                    provenance_notes=list(getattr(ref, "provenance_notes", None) or []),
                    conflict_weight=float(getattr(ref, "conflict_weight", 0.0) or 0.0),
                )
        return len(refs)

    def query_lore_artifacts(self, artifact_type: str | None = None, limit: int = 100) -> list[dict]:
        """Query lore artifacts by optional type."""
        where = "true"
        params: dict = {"limit": limit}
        if artifact_type:
            where = "a.artifact_type = $artifact_type"
            params["artifact_type"] = artifact_type

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (a:LoreArtifact)
                WHERE {where}
                RETURN a.id AS id, a.name AS name, a.artifact_type AS artifact_type,
                       a.description AS description, a.source_book AS source_book,
                       a.confidence AS confidence
                ORDER BY a.name
                LIMIT $limit
                """,
                **params,
            )
            return [dict(r) for r in result]

    def query_unresolved_references(self, source_book: str | None = None, limit: int = 100) -> list[dict]:
        """Query unresolved references, filtered by source book if provided."""
        where = "u.resolved_entity_id IS NULL"
        params: dict = {"limit": limit}
        if source_book:
            where += " AND toLower(coalesce(u.source_book,'')) CONTAINS toLower($source_book)"
            params["source_book"] = source_book

        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (u:UnresolvedReference)
                WHERE {where}
                RETURN u.id AS id, u.mention_text AS mention_text,
                       u.expected_type AS expected_type,
                       u.source_book AS source_book,
                       u.passage_id AS passage_id,
                       u.confidence AS confidence,
                       coalesce(u.conflict_weight, 0.0) AS conflict_weight,
                       coalesce(u.candidates, []) AS candidates,
                       coalesce(u.provenance_notes, []) AS provenance_notes
                ORDER BY (coalesce(u.conflict_weight, 0.0) + coalesce(u.confidence, 0.0)) DESC
                LIMIT $limit
                """,
                **params,
            )
            return [dict(r) for r in result]

    def query_unresolved_reference_queue(self, source_book: str | None = None, limit: int = 100) -> list[dict]:
        """Alias query optimized for downstream generation/review queue."""
        return self.query_unresolved_references(source_book=source_book, limit=limit)

    # =========================================================================
    # Sociolinguistic Registers (Issue #47 slice 1)
    # =========================================================================

    def write_register_profile(
        self,
        entity_id: str,
        profile,
        source_passage_id: str | None = None,
    ) -> None:
        """Persist the current sociolinguistic register profile for an entity."""
        from ..lore.sociolinguistic_registers import ground_character_entity_id

        entity_id = ground_character_entity_id(entity_id)
        if not entity_id:
            raise ValueError("Register profiles require a canonical character entity id")

        with self.driver.session() as session:
            session.run(
                """
                MATCH (e {id: $entity_id})
                MERGE (rp:RegisterProfile {entity_id: $entity_id})
                SET rp.dominant_register = $dominant_register,
                    rp.confidence = $confidence,
                    rp.formality_score = $formality_score,
                    rp.archaism_rate = $archaism_rate,
                    rp.contraction_rate = $contraction_rate,
                    rp.avg_sentence_length = $avg_sentence_length,
                    rp.token_count = $token_count,
                    rp.source_passage_id = $source_passage_id,
                    rp.updated_at = datetime()
                MERGE (e)-[:HAS_REGISTER_PROFILE]->(rp)
                """,
                entity_id=entity_id,
                dominant_register=profile.dominant_register,
                confidence=profile.confidence,
                formality_score=profile.formality_score,
                archaism_rate=profile.archaism_rate,
                contraction_rate=profile.contraction_rate,
                avg_sentence_length=profile.avg_sentence_length,
                token_count=profile.token_count,
                source_passage_id=source_passage_id,
            )

    def write_register_observation(
        self,
        entity_id: str,
        profile,
        observed_at: str,
        source_passage_id: str | None = None,
    ) -> None:
        """Write a time-stamped register observation for later drift analysis."""
        from ..lore.sociolinguistic_registers import ground_character_entity_id

        entity_id = ground_character_entity_id(entity_id)
        if not entity_id:
            raise ValueError("Register observations require a canonical character entity id")

        with self.driver.session() as session:
            session.run(
                """
                MATCH (e {id: $entity_id})
                CREATE (obs:RegisterObservation {
                    id: randomUUID(),
                    entity_id: $entity_id,
                    observed_at: $observed_at,
                    dominant_register: $dominant_register,
                    confidence: $confidence,
                    formality_score: $formality_score,
                    archaism_rate: $archaism_rate,
                    contraction_rate: $contraction_rate,
                    avg_sentence_length: $avg_sentence_length,
                    token_count: $token_count,
                    source_passage_id: $source_passage_id,
                    created_at: datetime()
                })
                MERGE (e)-[:HAS_REGISTER_OBSERVATION]->(obs)
                """,
                entity_id=entity_id,
                observed_at=observed_at,
                dominant_register=profile.dominant_register,
                confidence=profile.confidence,
                formality_score=profile.formality_score,
                archaism_rate=profile.archaism_rate,
                contraction_rate=profile.contraction_rate,
                avg_sentence_length=profile.avg_sentence_length,
                token_count=profile.token_count,
                source_passage_id=source_passage_id,
            )

    def query_register_drift(
        self,
        entity_id: str,
        min_delta: float = 0.2,
        limit: int = 20,
    ) -> list[dict]:
        """Query consecutive register observations and calculate drift deltas."""
        from ..lore.sociolinguistic_registers import ground_character_entity_id

        entity_id = ground_character_entity_id(entity_id)
        if not entity_id:
            return []

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e {id: $entity_id})-[:HAS_REGISTER_OBSERVATION]->(obs:RegisterObservation)
                RETURN obs.observed_at AS observed_at,
                       obs.dominant_register AS dominant_register,
                       obs.formality_score AS formality_score,
                       obs.archaism_rate AS archaism_rate,
                       obs.confidence AS confidence
                ORDER BY obs.observed_at ASC
                LIMIT $limit
                """,
                entity_id=entity_id,
                limit=limit,
            )
            rows = [dict(r) for r in result]

        drifts: list[dict] = []
        for i in range(1, len(rows)):
            prev = rows[i - 1]
            cur = rows[i]
            delta = {
                "from": prev.get("observed_at"),
                "to": cur.get("observed_at"),
                "from_register": prev.get("dominant_register"),
                "to_register": cur.get("dominant_register"),
                "formality_shift": (cur.get("formality_score") or 0.0) - (prev.get("formality_score") or 0.0),
                "archaism_shift": (cur.get("archaism_rate") or 0.0) - (prev.get("archaism_rate") or 0.0),
            }
            magnitude = max(abs(delta["formality_shift"]), abs(delta["archaism_shift"]))
            if delta["from_register"] != delta["to_register"]:
                magnitude = max(magnitude, 0.3)
            if magnitude >= min_delta:
                delta["magnitude"] = round(magnitude, 4)
                drifts.append(delta)
        return drifts

    # =========================================================================
    # Sociolinguistic Registers (Issue #47 slice 1)
    # =========================================================================

    def write_register_profile(
        self,
        entity_id: str,
        profile,
        source_passage_id: str | None = None,
    ) -> None:
        """Persist the current sociolinguistic register profile for an entity."""
        from ..lore.sociolinguistic_registers import ground_character_entity_id

        entity_id = ground_character_entity_id(entity_id)
        if not entity_id:
            raise ValueError("Register profiles require a canonical character entity id")

        with self.driver.session() as session:
            session.run(
                """
                MATCH (e {id: $entity_id})
                MERGE (rp:RegisterProfile {entity_id: $entity_id})
                SET rp.dominant_register = $dominant_register,
                    rp.confidence = $confidence,
                    rp.formality_score = $formality_score,
                    rp.archaism_rate = $archaism_rate,
                    rp.contraction_rate = $contraction_rate,
                    rp.avg_sentence_length = $avg_sentence_length,
                    rp.token_count = $token_count,
                    rp.source_passage_id = $source_passage_id,
                    rp.updated_at = datetime()
                MERGE (e)-[:HAS_REGISTER_PROFILE]->(rp)
                """,
                entity_id=entity_id,
                dominant_register=profile.dominant_register,
                confidence=profile.confidence,
                formality_score=profile.formality_score,
                archaism_rate=profile.archaism_rate,
                contraction_rate=profile.contraction_rate,
                avg_sentence_length=profile.avg_sentence_length,
                token_count=profile.token_count,
                source_passage_id=source_passage_id,
            )

    def write_register_observation(
        self,
        entity_id: str,
        profile,
        observed_at: str,
        source_passage_id: str | None = None,
    ) -> None:
        """Write a time-stamped register observation for later drift analysis."""
        from ..lore.sociolinguistic_registers import ground_character_entity_id

        entity_id = ground_character_entity_id(entity_id)
        if not entity_id:
            raise ValueError("Register observations require a canonical character entity id")

        with self.driver.session() as session:
            session.run(
                """
                MATCH (e {id: $entity_id})
                CREATE (obs:RegisterObservation {
                    id: randomUUID(),
                    entity_id: $entity_id,
                    observed_at: $observed_at,
                    dominant_register: $dominant_register,
                    confidence: $confidence,
                    formality_score: $formality_score,
                    archaism_rate: $archaism_rate,
                    contraction_rate: $contraction_rate,
                    avg_sentence_length: $avg_sentence_length,
                    token_count: $token_count,
                    source_passage_id: $source_passage_id,
                    created_at: datetime()
                })
                MERGE (e)-[:HAS_REGISTER_OBSERVATION]->(obs)
                """,
                entity_id=entity_id,
                observed_at=observed_at,
                dominant_register=profile.dominant_register,
                confidence=profile.confidence,
                formality_score=profile.formality_score,
                archaism_rate=profile.archaism_rate,
                contraction_rate=profile.contraction_rate,
                avg_sentence_length=profile.avg_sentence_length,
                token_count=profile.token_count,
                source_passage_id=source_passage_id,
            )

    def query_register_drift(
        self,
        entity_id: str,
        min_delta: float = 0.2,
        limit: int = 20,
    ) -> list[dict]:
        """Query consecutive register observations and calculate drift deltas."""
        from ..lore.sociolinguistic_registers import ground_character_entity_id

        entity_id = ground_character_entity_id(entity_id)
        if not entity_id:
            return []

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e {id: $entity_id})-[:HAS_REGISTER_OBSERVATION]->(obs:RegisterObservation)
                RETURN obs.observed_at AS observed_at,
                       obs.dominant_register AS dominant_register,
                       obs.formality_score AS formality_score,
                       obs.archaism_rate AS archaism_rate,
                       obs.confidence AS confidence
                ORDER BY obs.observed_at ASC
                LIMIT $limit
                """,
                entity_id=entity_id,
                limit=limit,
            )
            rows = [dict(r) for r in result]

        drifts: list[dict] = []
        for i in range(1, len(rows)):
            prev = rows[i - 1]
            cur = rows[i]
            delta = {
                "from": prev.get("observed_at"),
                "to": cur.get("observed_at"),
                "from_register": prev.get("dominant_register"),
                "to_register": cur.get("dominant_register"),
                "formality_shift": (cur.get("formality_score") or 0.0) - (prev.get("formality_score") or 0.0),
                "archaism_shift": (cur.get("archaism_rate") or 0.0) - (prev.get("archaism_rate") or 0.0),
            }
            magnitude = max(abs(delta["formality_shift"]), abs(delta["archaism_shift"]))
            if delta["from_register"] != delta["to_register"]:
                magnitude = max(magnitude, 0.3)
            if magnitude >= min_delta:
                delta["magnitude"] = round(magnitude, 4)
                drifts.append(delta)
        return drifts

    def query_register_observations(
        self,
        entity_id: str,
        limit: int = 25,
    ) -> list[dict]:
        """Return recent register observations for an entity."""
        from ..lore.sociolinguistic_registers import ground_character_entity_id

        entity_id = ground_character_entity_id(entity_id)
        if not entity_id:
            return []

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e {id: $entity_id})-[:HAS_REGISTER_OBSERVATION]->(obs:RegisterObservation)
                RETURN obs.observed_at AS observed_at,
                       obs.dominant_register AS dominant_register,
                       obs.confidence AS confidence,
                       obs.formality_score AS formality_score,
                       obs.archaism_rate AS archaism_rate,
                       obs.contraction_rate AS contraction_rate,
                       obs.source_passage_id AS source_passage_id
                ORDER BY obs.observed_at DESC
                LIMIT $limit
                """,
                entity_id=entity_id,
                limit=limit,
            )
            return [dict(r) for r in result]

    def query_register_drift_summary(
        self,
        entity_id: str,
        min_delta: float = 0.2,
        limit: int = 100,
    ) -> dict:
        """Summarize drift counts and strongest transition for reporting."""
        from ..lore.sociolinguistic_registers import ground_character_entity_id

        entity_id = ground_character_entity_id(entity_id)
        if not entity_id:
            return {
                "entity_id": None,
                "drift_count": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "strongest": None,
            }

        drifts = self.query_register_drift(entity_id=entity_id, min_delta=min_delta, limit=limit)
        if not drifts:
            return {
                "entity_id": entity_id,
                "drift_count": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "strongest": None,
            }

        def _severity(d: dict) -> str:
            mag = float(d.get("magnitude") or 0.0)
            if mag >= 0.45:
                return "high"
            if mag >= 0.25:
                return "medium"
            return "low"

        counts = {"high": 0, "medium": 0, "low": 0}
        for d in drifts:
            counts[_severity(d)] += 1

        strongest = max(drifts, key=lambda d: float(d.get("magnitude") or 0.0))
        return {
            "entity_id": entity_id,
            "drift_count": len(drifts),
            **counts,
            "strongest": strongest,
        }

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

    # =========================================================================
    # Spatiotemporal Engine (Issue #48)
    # =========================================================================

    def write_spatiotemporal_event(self, event) -> None:
        """Persist a SpatiotemporalEvent node with normalized time fields.

        Creates :SpatiotemporalEvent node linked to entity and location.
        """
        query = """
        MERGE (e:SpatiotemporalEvent {id: $id})
        SET e.entity_id = $entity_id, e.entity_name = $entity_name,
            e.location_id = $location_id, e.location_name = $location_name,
            e.description = $description, e.event_type = $event_type,
            e.source_book = $source_book, e.source_passage_id = $source_passage_id,
            e.source_id = $source_id,
            e.structural_stratum = $structural_stratum,
            e.editorial_status = $editorial_status,
            e.source_authority_weight = $source_authority_weight,
            e.time_era = $time_era, e.time_year_start = $time_year_start,
            e.time_year_end = $time_year_end, e.time_confidence = $time_confidence,
            e.time_raw_text = $time_raw_text
        """
        params = {
            "id": event.id, "entity_id": event.entity_id,
            "entity_name": event.entity_name, "location_id": event.location_id,
            "location_name": event.location_name, "description": event.description,
            "event_type": event.event_type, "source_book": event.source_book,
            "source_passage_id": event.source_passage_id,
            "source_id": getattr(event, "source_id", None),
            "structural_stratum": getattr(event, "structural_stratum", None),
            "editorial_status": getattr(event, "editorial_status", None),
            "source_authority_weight": getattr(event, "source_authority_weight", None),
            "time_era": event.time.era, "time_year_start": event.time.year_start,
            "time_year_end": event.time.year_end, "time_confidence": event.time.confidence,
            "time_raw_text": event.time.raw_text,
        }
        with self.driver.session() as session:
            session.run(query, **params)
            if event.entity_id:
                session.run("""
                    MATCH (se:SpatiotemporalEvent {id: $se_id})
                    MATCH (ent) WHERE ent.canonical_name = $ename OR ent.id = $eid
                    MERGE (ent)-[:PARTICIPATED_IN]->(se)
                """, se_id=event.id, ename=event.entity_name, eid=event.entity_id)
            if event.location_id:
                session.run("""
                    MATCH (se:SpatiotemporalEvent {id: $se_id})
                    MERGE (loc:Location {id: $loc_id})
                    ON CREATE SET loc.name = $loc_name
                    MERGE (se)-[:LOCATED_AT]->(loc)
                """, se_id=event.id, loc_id=event.location_id, loc_name=event.location_name)

    def write_spatiotemporal_events_batch(self, events: list) -> int:
        """Write a batch of SpatiotemporalEvent objects. Returns count written."""
        for event in events:
            self.write_spatiotemporal_event(event)
        return len(events)

    def write_location_graph(self, locations: list, edges: list) -> dict:
        """Persist location nodes and travel edges."""
        with self.driver.session() as session:
            for loc in locations:
                session.run("""
                    MERGE (l:Location {id: $id})
                    SET l.name = $name, l.region = $region, l.x = $x, l.y = $y, l.aliases = $aliases
                """, id=loc.id, name=loc.name, region=loc.region, x=loc.x, y=loc.y, aliases=loc.aliases)
            for edge in edges:
                session.run("""
                    MATCH (a:Location {id: $src}), (b:Location {id: $tgt})
                    MERGE (a)-[r:TRAVEL_ROUTE]->(b)
                    SET r.travel_days = $days, r.mode = $mode, r.difficulty = $difficulty
                """, src=edge.source_id, tgt=edge.target_id,
                    days=edge.travel_days, mode=edge.mode, difficulty=edge.difficulty)
                if edge.bidirectional:
                    session.run("""
                        MATCH (a:Location {id: $tgt}), (b:Location {id: $src})
                        MERGE (a)-[r:TRAVEL_ROUTE]->(b)
                        SET r.travel_days = $days, r.mode = $mode, r.difficulty = $difficulty
                    """, src=edge.source_id, tgt=edge.target_id,
                        days=edge.travel_days, mode=edge.mode, difficulty=edge.difficulty)
        return {"locations_written": len(locations), "edges_written": len(edges)}

    def query_conflicting_overlaps(self, entity_id: str) -> list[dict]:
        """Find SpatiotemporalEvents where entity is at two places at once."""
        query = """
        MATCH (e1:SpatiotemporalEvent {entity_id: $eid}),
              (e2:SpatiotemporalEvent {entity_id: $eid})
        WHERE e1.id < e2.id AND e1.location_id <> e2.location_id
          AND e1.time_era = e2.time_era
          AND (e1.time_year_start <= e2.time_year_end AND e2.time_year_start <= e1.time_year_end
               OR e1.time_year_start IS NULL OR e2.time_year_start IS NULL)
        RETURN e1.id AS event1_id, e1.description AS desc1, e1.location_name AS loc1,
               e2.id AS event2_id, e2.description AS desc2, e2.location_name AS loc2
        LIMIT 50
        """
        results = []
        with self.driver.session() as session:
            for record in session.run(query, eid=entity_id):
                results.append(dict(record))
        return results

    def write_timeline_conflict(self, conflict) -> None:
        """Persist a TimelineConflict node to Neo4j.

        Idempotent: uses MERGE on conflict id.
        Links to involved SpatiotemporalEvent nodes if they exist.
        """
        query = """
        MERGE (c:TimelineConflict {id: $id})
        SET c.conflict_type = $conflict_type, c.severity = $severity,
            c.description = $description, c.event_a_id = $event_a_id,
            c.event_b_id = $event_b_id, c.entity_id = $entity_id,
            c.suggestion = $suggestion, c.confidence = $confidence,
            c.event_a_source_book = $event_a_source_book,
            c.event_b_source_book = $event_b_source_book,
            c.event_a_source_authority_weight = $event_a_source_authority_weight,
            c.event_b_source_authority_weight = $event_b_source_authority_weight,
            c.updated_at = datetime()
        """
        params = {
            "id": conflict.id,
            "conflict_type": conflict.conflict_type.value if hasattr(conflict.conflict_type, "value") else str(conflict.conflict_type),
            "severity": conflict.severity,
            "description": conflict.description,
            "event_a_id": conflict.event_a_id,
            "event_b_id": conflict.event_b_id,
            "entity_id": conflict.entity_id,
            "suggestion": conflict.suggestion,
            "confidence": conflict.confidence,
            "event_a_source_book": getattr(conflict, "event_a_source_book", None),
            "event_b_source_book": getattr(conflict, "event_b_source_book", None),
            "event_a_source_authority_weight": getattr(conflict, "event_a_source_authority_weight", None),
            "event_b_source_authority_weight": getattr(conflict, "event_b_source_authority_weight", None),
        }
        with self.driver.session() as session:
            session.run(query, **params)
            # Link to event nodes if they exist
            if conflict.event_a_id:
                session.run("""
                    MATCH (c:TimelineConflict {id: $cid})
                    MATCH (e:SpatiotemporalEvent {id: $eid})
                    MERGE (c)-[:INVOLVES]->(e)
                """, cid=conflict.id, eid=conflict.event_a_id)
            if conflict.event_b_id:
                session.run("""
                    MATCH (c:TimelineConflict {id: $cid})
                    MATCH (e:SpatiotemporalEvent {id: $eid})
                    MERGE (c)-[:INVOLVES]->(e)
                """, cid=conflict.id, eid=conflict.event_b_id)

    def write_timeline_conflicts_batch(self, conflicts: list) -> int:
        """Write a batch of TimelineConflict objects. Returns count written."""
        for conflict in conflicts:
            self.write_timeline_conflict(conflict)
        return len(conflicts)

    def query_timeline_conflicts(
        self,
        conflict_type: str | None = None,
        severity: str | None = None,
        entity_id: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 50,
    ) -> list[dict]:
        """Query persisted timeline conflicts from Neo4j.

        Args:
            conflict_type: Filter by type (e.g. 'causal_paradox', 'temporal_overlap')
            severity: Filter by severity ('error' or 'warning')
            entity_id: Filter by involved entity
            min_confidence: Minimum confidence threshold
            limit: Max results

        Returns:
            List of conflict dicts sorted by confidence descending
        """
        conditions = ["c.confidence >= $min_conf"]
        params: dict = {"min_conf": min_confidence, "limit": limit}

        if conflict_type:
            conditions.append("c.conflict_type = $ctype")
            params["ctype"] = conflict_type
        if severity:
            conditions.append("c.severity = $sev")
            params["sev"] = severity
        if entity_id:
            conditions.append("c.entity_id = $eid")
            params["eid"] = entity_id

        where = " AND ".join(conditions)
        query = f"""
        MATCH (c:TimelineConflict)
        WHERE {where}
        RETURN c.id AS id, c.conflict_type AS conflict_type,
               c.severity AS severity, c.description AS description,
               c.event_a_id AS event_a_id, c.event_b_id AS event_b_id,
               c.entity_id AS entity_id, c.suggestion AS suggestion,
               c.confidence AS confidence,
               c.event_a_source_book AS event_a_source_book,
               c.event_b_source_book AS event_b_source_book,
               c.event_a_source_authority_weight AS event_a_source_authority_weight,
               c.event_b_source_authority_weight AS event_b_source_authority_weight
        ORDER BY c.confidence DESC
        LIMIT $limit
        """
        results = []
        with self.driver.session() as session:
            for record in session.run(query, **params):
                results.append(dict(record))
        return results

    def query_recent_critical_conflicts(self, limit: int = 20) -> list[dict]:
        """Query recent high-severity timeline conflicts.

        Returns error-level conflicts ordered by most recently updated.
        """
        query = """
        MATCH (c:TimelineConflict)
        WHERE c.severity = 'error'
        RETURN c.id AS id, c.conflict_type AS conflict_type,
               c.description AS description, c.confidence AS confidence,
               c.entity_id AS entity_id, c.updated_at AS updated_at
        ORDER BY c.updated_at DESC, c.confidence DESC
        LIMIT $limit
        """
        results = []
        with self.driver.session() as session:
            for record in session.run(query, limit=limit):
                results.append(dict(record))
        return results

    def query_travel_infeasibility(self, entity_id: str, max_speed_per_year: float = 365.0) -> list[dict]:
        """Find consecutive events where travel time exceeds available time."""
        query = """
        MATCH (e:SpatiotemporalEvent {entity_id: $eid})
        WHERE e.time_year_start IS NOT NULL AND e.location_id IS NOT NULL
        WITH e ORDER BY e.time_era, e.time_year_start
        WITH collect(e) AS events
        UNWIND range(0, size(events)-2) AS i
        WITH events[i] AS e1, events[i+1] AS e2
        WHERE e1.location_id <> e2.location_id
        OPTIONAL MATCH (l1:Location {id: e1.location_id})
        OPTIONAL MATCH (l2:Location {id: e2.location_id})
        WITH e1, e2, l1, l2,
             CASE WHEN l1 IS NOT NULL AND l2 IS NOT NULL
                  THEN sqrt((l1.x - l2.x)^2 + (l1.y - l2.y)^2)
                  ELSE null END AS distance,
             CASE WHEN e1.time_era = e2.time_era
                  THEN abs(e2.time_year_start - e1.time_year_start)
                  ELSE null END AS year_gap
        WHERE distance IS NOT NULL AND year_gap IS NOT NULL
          AND distance > year_gap * $speed
        RETURN e1.id AS from_id, e1.location_name AS from_loc,
               e2.id AS to_id, e2.location_name AS to_loc, distance, year_gap
        """
        results = []
        with self.driver.session() as session:
            for record in session.run(query, eid=entity_id, speed=max_speed_per_year):
                results.append(dict(record))
        return results

    # ------------------------------------------------------------------
    # CausalLink persistence (Slice 4 — Issue #48)
    # ------------------------------------------------------------------

    def write_causal_link(self, link) -> None:
        """Persist a CausalLink as an edge between SpatiotemporalEvent nodes.

        Creates a :CAUSES relationship with description and confidence.
        Idempotent: uses MERGE on the (cause, effect) pair.
        Also creates a :CausalLink node for queryability.
        """
        # Node for queryability
        node_q = """
        MERGE (cl:CausalLink {cause_event_id: $cause_id, effect_event_id: $effect_id})
        SET cl.description = $desc, cl.confidence = $conf, cl.updated_at = datetime()
        """
        # Edge between events
        edge_q = """
        MATCH (a:SpatiotemporalEvent {id: $cause_id})
        MATCH (b:SpatiotemporalEvent {id: $effect_id})
        MERGE (a)-[r:CAUSES]->(b)
        SET r.description = $desc, r.confidence = $conf
        """
        params = {
            "cause_id": link.cause_event_id,
            "effect_id": link.effect_event_id,
            "desc": link.description,
            "conf": link.confidence,
        }
        with self.driver.session() as session:
            session.run(node_q, **params)
            session.run(edge_q, **params)

    def write_causal_links_batch(self, links: list) -> int:
        """Write a batch of CausalLink objects. Returns count written."""
        for link in links:
            self.write_causal_link(link)
        return len(links)

    def query_causal_chain(self, event_id: str, direction: str = "forward", max_depth: int = 10) -> list[dict]:
        """Query causal chain from an event.

        Args:
            event_id: Starting event ID
            direction: 'forward' (effects) or 'backward' (causes)
            max_depth: Maximum chain depth

        Returns:
            List of dicts with event info and chain depth
        """
        if direction == "forward":
            query = f"""
            MATCH path = (start:SpatiotemporalEvent {{id: $eid}})-[:CAUSES*1..{max_depth}]->(e:SpatiotemporalEvent)
            RETURN e.id AS id, e.description AS description, e.entity_name AS entity_name,
                   e.time_era AS era, e.time_year_start AS year, length(path) AS depth
            ORDER BY depth
            """
        else:
            query = f"""
            MATCH path = (e:SpatiotemporalEvent)-[:CAUSES*1..{max_depth}]->(target:SpatiotemporalEvent {{id: $eid}})
            RETURN e.id AS id, e.description AS description, e.entity_name AS entity_name,
                   e.time_era AS era, e.time_year_start AS year, length(path) AS depth
            ORDER BY depth
            """
        results = []
        with self.driver.session() as session:
            for record in session.run(query, eid=event_id):
                results.append(dict(record))
        return results

    def query_causal_violations(self, min_confidence: float = 0.0, limit: int = 50) -> list[dict]:
        """Find causal links where effect occurs before cause (paradoxes).

        Uses the graph to find CAUSES edges where the effect event's time
        is earlier than the cause event's time.
        """
        query = """
        MATCH (cause:SpatiotemporalEvent)-[r:CAUSES]->(effect:SpatiotemporalEvent)
        WHERE r.confidence >= $min_conf
          AND cause.time_era = effect.time_era
          AND effect.time_year_end IS NOT NULL AND cause.time_year_start IS NOT NULL
          AND effect.time_year_end < cause.time_year_start
        RETURN cause.id AS cause_id, cause.description AS cause_desc,
               effect.id AS effect_id, effect.description AS effect_desc,
               r.confidence AS confidence, r.description AS link_desc,
               cause.time_era AS era
        ORDER BY r.confidence DESC
        LIMIT $limit
        """
        results = []
        with self.driver.session() as session:
            for record in session.run(query, min_conf=min_confidence, limit=limit):
                results.append(dict(record))
        return results

    def query_divergence_hotspots(self, min_sources: int = 2, limit: int = 25) -> list[dict]:
        """Find entity/type conflict clusters spanning multiple source books."""
        query = """
        MATCH (c:TimelineConflict)-[:INVOLVES]->(e:SpatiotemporalEvent)
        WITH coalesce(c.entity_id, e.entity_id) AS entity_id,
             c.conflict_type AS conflict_type,
             collect(DISTINCT c.id) AS conflict_ids,
             collect(DISTINCT coalesce(e.source_book, 'unknown')) AS source_books,
             avg(coalesce(e.source_authority_weight, 1.0)) AS avg_authority
        WHERE size(source_books) >= $min_sources
        RETURN entity_id, conflict_type, conflict_ids, source_books,
               size(conflict_ids) AS conflict_count,
               size(source_books) AS source_count,
               avg_authority
        ORDER BY conflict_count DESC, avg_authority DESC
        LIMIT $limit
        """
        results = []
        with self.driver.session() as session:
            for record in session.run(query, min_sources=min_sources, limit=limit):
                results.append(dict(record))
        return results

    def query_source_divergence(self, source_a: str, source_b: str, limit: int = 50) -> list[dict]:
        """Return conflicts evidenced by events from both requested sources."""
        query = """
        MATCH (c:TimelineConflict)-[:INVOLVES]->(e:SpatiotemporalEvent)
        WITH c, collect(DISTINCT coalesce(e.source_book, 'unknown')) AS sources,
             collect(DISTINCT e.id) AS events
        WHERE $source_a IN sources AND $source_b IN sources
        RETURN c.id AS id, c.conflict_type AS conflict_type,
               c.severity AS severity, c.description AS description,
               c.confidence AS confidence, c.entity_id AS entity_id,
               sources, events
        ORDER BY c.confidence DESC
        LIMIT $limit
        """
        results = []
        with self.driver.session() as session:
            for record in session.run(query, source_a=source_a, source_b=source_b, limit=limit):
                results.append(dict(record))
        return results
    _STOPWORDS = {
        "the", "a", "an", "of", "and", "or", "to", "for", "in", "on", "at", "by", "with",
        "from", "into", "onto", "upon", "his", "her", "their", "its", "him", "them",
    }

    @staticmethod
    def _normalize_entity_text(value: str | None) -> str:
        if not value:
            return ""
        text = value.lower().strip()
        text = re.sub(r"[\"'`]+", "", text)
        text = re.sub(r"[^a-z0-9\s-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _meaningful_tokens(cls, value: str | None) -> list[str]:
        text = cls._normalize_entity_text(value)
        if not text:
            return []
        return [t for t in text.split(" ") if t and t not in cls._STOPWORDS and len(t) > 2]

    @classmethod
    def _expand_candidate_strings(cls, value: str | None) -> list[str]:
        """Generate conservative candidate strings for entity matching."""
        text = cls._normalize_entity_text(value)
        if not text:
            return []

        variants = {text}
        # Split simple conjunction/apposition forms: "Bilbo and dwarves", "Bilbo, Gandalf"
        for part in re.split(r"\b(?:and|or)\b|,|/", text):
            p = part.strip()
            if p:
                variants.add(p)

        # Add compact token-only variants for noisy spans
        tokens = cls._meaningful_tokens(text)
        if len(tokens) >= 2:
            variants.add(" ".join(tokens))
        for t in tokens:
            variants.add(t)

        # Prefer longer strings first for better precision
        return sorted((v for v in variants if v), key=len, reverse=True)

    def _link_event_role(
        self,
        *,
        event_id: str,
        source_book: str,
        raw_value: str | None,
        labels: list[str],
        rel_type: str,
        role: str,
    ) -> int:
        """Link one event role to a best-matching canonical entity.

        Returns 1 when a link exists after MERGE, else 0.
        """
        candidates = self._expand_candidate_strings(raw_value)
        candidate_ids = [c for c in candidates if re.match(r"^(char|place|obj)_[a-z0-9_\-]+$", c)]
        if not candidates:
            return 0

        query = f"""
        MATCH (e:Event {{id: $event_id, source_book: $source_book}})
        WITH e, [c IN $candidates WHERE c IS NOT NULL AND trim(c) <> ''] AS candidates
        UNWIND candidates AS cand
        MATCH (n)
        WHERE any(lbl IN $labels WHERE lbl IN labels(n))
        WITH e, n, cand,
             toLower(coalesce(n.canonical_name, n.name, '')) AS cname,
             toLower(coalesce(n.canonical_id, n.id, '')) AS cid,
             [a IN coalesce(n.aliases, []) | toLower(a)] AS aliases
        WHERE (cname <> '' OR cid <> '')
          AND (
            cid IN $candidate_ids
            OR
            cname = cand
            OR cand IN aliases
            OR cname CONTAINS cand
            OR cand CONTAINS cname
            OR any(a IN aliases WHERE a CONTAINS cand OR cand CONTAINS a)
          )
        WITH e, n, cand,
             CASE
               WHEN cid IN $candidate_ids THEN 110
               WHEN cname = cand OR cand IN aliases THEN 100
               WHEN any(a IN aliases WHERE a = cand) THEN 95
               WHEN cname CONTAINS cand OR cand CONTAINS cname THEN 70
               ELSE 50
             END AS score,
             size(cname) AS name_len
        ORDER BY score DESC, name_len DESC
        LIMIT 1
        MERGE (n)-[r:{rel_type}]->(e)
        SET r.role = $role
        RETURN count(r) AS cnt
        """

        with self.driver.session() as session:
            row = session.run(
                query,
                event_id=event_id,
                source_book=source_book,
                candidates=candidates,
                candidate_ids=candidate_ids,
                labels=labels,
                role=role,
            ).single()
            return int(row["cnt"]) if row else 0
