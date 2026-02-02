"""Relationship extraction from text.

Extracts relationships between entities using:
1. LLM-based extraction for complex sentences
2. Dependency parsing for simple subject-verb-object patterns
3. Pattern matching for common relationship types
"""

import json
import re
from dataclasses import dataclass
from typing import Iterator

import httpx
import spacy

from ..config import get_settings
from ..models.relationships import RelationshipType, ExtractedRelationship
from .resolver import EntityResolver, ResolvedEntity


# Mapping from verb lemmas to relationship types
# Using lemmas (base forms) for consistent matching
VERB_TO_RELATIONSHIP: dict[str, RelationshipType] = {
    # Speech
    "say": RelationshipType.SPOKE_TO,
    "ask": RelationshipType.SPOKE_TO,
    "tell": RelationshipType.SPOKE_TO,
    "answer": RelationshipType.SPOKE_TO,
    "reply": RelationshipType.SPOKE_TO,
    "speak": RelationshipType.SPOKE_WITH,
    "talk": RelationshipType.SPOKE_WITH,
    "call": RelationshipType.SPOKE_TO,
    "cry": RelationshipType.SPOKE_TO,
    "shout": RelationshipType.SPOKE_TO,
    "whisper": RelationshipType.SPOKE_TO,

    # Movement
    "go": RelationshipType.TRAVELED_TO,
    "travel": RelationshipType.TRAVELED_TO,
    "come": RelationshipType.TRAVELED_TO,
    "arrive": RelationshipType.TRAVELED_TO,
    "reach": RelationshipType.TRAVELED_TO,
    "enter": RelationshipType.ENTERED,
    "leave": RelationshipType.LEFT,
    "flee": RelationshipType.LEFT,
    "escape": RelationshipType.LEFT,
    "follow": RelationshipType.TRAVELED_WITH,
    "accompany": RelationshipType.TRAVELED_WITH,

    # Combat
    "fight": RelationshipType.FOUGHT,
    "attack": RelationshipType.FOUGHT_AGAINST,
    "kill": RelationshipType.KILLED,
    "slay": RelationshipType.KILLED,
    "defeat": RelationshipType.FOUGHT_AGAINST,
    "capture": RelationshipType.CAPTURED,
    "free": RelationshipType.FREED,

    # Objects
    "give": RelationshipType.GAVE,
    "receive": RelationshipType.RECEIVED,
    "take": RelationshipType.POSSESSES,
    "find": RelationshipType.FOUND,
    "lose": RelationshipType.LOST,
    "steal": RelationshipType.STOLE,
    "use": RelationshipType.USED,
    "carry": RelationshipType.POSSESSES,
    "wear": RelationshipType.POSSESSES,
    "wield": RelationshipType.POSSESSES,

    # Social
    "meet": RelationshipType.MET,
    "join": RelationshipType.ALLIED_WITH,
    "help": RelationshipType.HELPED,
    "serve": RelationshipType.SERVES,
    "lead": RelationshipType.LEADS,
    "betray": RelationshipType.BETRAYED,

    # Location
    "live": RelationshipType.LIVES_IN,
    "dwell": RelationshipType.LIVES_IN,
    "visit": RelationshipType.VISITED,
    "rule": RelationshipType.RULES,
    "guard": RelationshipType.GUARDS,
}


@dataclass
class RelationshipExtractionResult:
    """Result of extracting relationships from a passage."""

    passage_id: str
    passage_text: str
    relationships: list[ExtractedRelationship]
    entities_involved: list[ResolvedEntity]


class RelationshipExtractor:
    """Extracts relationships between entities in text."""

    def __init__(self, resolver: EntityResolver | None = None, use_llm: bool = True):
        """Initialize the relationship extractor.

        Args:
            resolver: Entity resolver for mapping names to canonical IDs
            use_llm: Whether to use LLM for complex extraction
        """
        self.settings = get_settings()
        self.resolver = resolver or EntityResolver()
        self.use_llm = use_llm
        self._nlp = None

    @property
    def nlp(self) -> spacy.Language:
        """Lazy-load spaCy model."""
        if self._nlp is None:
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def extract_relationships(
        self,
        text: str,
        passage_id: str,
        entities: list[ResolvedEntity],
    ) -> RelationshipExtractionResult:
        """Extract relationships from a passage.

        Args:
            text: The passage text
            passage_id: Unique ID for this passage
            entities: Pre-extracted entities in this passage

        Returns:
            RelationshipExtractionResult with extracted relationships
        """
        relationships = []

        # Build entity lookup by text
        entity_lookup = {e.extracted.text.lower(): e for e in entities}

        # Method 1: Dependency parsing for simple patterns
        dep_relationships = self._extract_dependency(text, passage_id, entity_lookup)
        relationships.extend(dep_relationships)

        # Method 2: LLM extraction for complex patterns
        if self.use_llm and len(entities) >= 2:
            llm_relationships = self._extract_llm(text, passage_id, entities)
            relationships.extend(llm_relationships)

        # Deduplicate relationships
        relationships = self._deduplicate(relationships)

        return RelationshipExtractionResult(
            passage_id=passage_id,
            passage_text=text,
            relationships=relationships,
            entities_involved=entities,
        )

    def _extract_dependency(
        self,
        text: str,
        passage_id: str,
        entity_lookup: dict[str, ResolvedEntity],
    ) -> list[ExtractedRelationship]:
        """Extract relationships using dependency parsing."""
        relationships = []
        doc = self.nlp(text)

        for token in doc:
            # Look for verbs
            if token.pos_ != "VERB":
                continue

            # Get the lemma for mapping
            verb_lemma = token.lemma_.lower()
            if verb_lemma not in VERB_TO_RELATIONSHIP:
                continue

            rel_type = VERB_TO_RELATIONSHIP[verb_lemma]

            # Find subject (nsubj)
            subject = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    # Get the full noun phrase
                    subject = self._get_span_text(child)
                    break

            # Find object (dobj, pobj via prep)
            obj = None
            prep_type = None
            for child in token.children:
                if child.dep_ in ("dobj", "attr"):
                    obj = self._get_span_text(child)
                    break
                # Check prepositional phrases (e.g., "traveled TO X", "spoke WITH Y")
                if child.dep_ == "prep":
                    prep_type = child.text.lower()
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            obj = self._get_span_text(pobj)
                            break
                    if obj:
                        break

            if not subject or not obj:
                continue

            # Try to resolve to known entities
            subject_entity = self._find_entity(subject, entity_lookup)
            object_entity = self._find_entity(obj, entity_lookup)

            # Only create relationship if at least one is a known entity
            if subject_entity or object_entity:
                relationships.append(
                    ExtractedRelationship(
                        subject_text=subject,
                        subject_id=subject_entity.canonical_id if subject_entity else None,
                        subject_type=subject_entity.entity_type if subject_entity else None,
                        predicate=rel_type,
                        predicate_raw=token.text,
                        object_text=obj,
                        object_id=object_entity.canonical_id if object_entity else None,
                        object_type=object_entity.entity_type if object_entity else None,
                        passage_id=passage_id,
                        passage_text=text[:200],
                        confidence=0.7,
                        extraction_method="dependency",
                    )
                )

        return relationships

    def _get_span_text(self, token) -> str:
        """Get the full text of a noun phrase from a token."""
        # Try to get the noun chunk containing this token
        doc = token.doc
        for chunk in doc.noun_chunks:
            if token in chunk:
                return chunk.text

        # Fall back to just the token and its compounds
        text_parts = []
        for child in token.lefts:
            if child.dep_ in ("compound", "amod", "det"):
                text_parts.append(child.text)
        text_parts.append(token.text)
        return " ".join(text_parts)

    def _find_entity(
        self,
        text: str,
        entity_lookup: dict[str, ResolvedEntity],
    ) -> ResolvedEntity | None:
        """Find a resolved entity matching the text."""
        text_lower = text.lower().strip()

        # Exact match
        if text_lower in entity_lookup:
            return entity_lookup[text_lower]

        # Try without articles
        for prefix in ["the ", "a ", "an "]:
            if text_lower.startswith(prefix):
                stripped = text_lower[len(prefix):]
                if stripped in entity_lookup:
                    return entity_lookup[stripped]

        # Try partial match (entity text contains our text or vice versa)
        for key, entity in entity_lookup.items():
            if key in text_lower or text_lower in key:
                return entity

        return None

    def _extract_llm(
        self,
        text: str,
        passage_id: str,
        entities: list[ResolvedEntity],
    ) -> list[ExtractedRelationship]:
        """Extract relationships using LLM."""
        # Build entity list for context
        entity_names = [e.extracted.text for e in entities]

        prompt = f"""Extract relationships between entities in this sentence from Tolkien's works.

Entities present: {', '.join(entity_names)}

Sentence: "{text}"

Return a JSON array of relationships. Each relationship should have:
- "subject": the entity performing the action
- "predicate": the relationship type (use: SPOKE_WITH, SPOKE_TO, TRAVELED_TO, TRAVELED_WITH, FOUGHT, KILLED, GAVE, RECEIVED, FOUND, POSSESSES, MET, HELPED, CAPTURED, ENTERED, LEFT, LIVES_IN)
- "object": the entity receiving the action
- "indirect_object": (optional) third party (e.g., for "X gave Y to Z", Z is indirect)

Only include relationships you can clearly identify. Return empty array [] if none found.

JSON array:"""

        try:
            response = httpx.post(
                f"{self.settings.ollama_base_url}/api/generate",
                json={
                    "model": self.settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
                timeout=30.0,
            )

            if response.status_code != 200:
                return []

            result = response.json()
            response_text = result.get("response", "").strip()

            # Parse JSON
            if "```" in response_text:
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
                if json_match:
                    response_text = json_match.group(1)

            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                array_match = re.search(r"\[[\s\S]*\]", response_text)
                if array_match:
                    data = json.loads(array_match.group(0))
                else:
                    return []

            # Convert to ExtractedRelationship objects
            relationships = []
            entity_lookup = {e.extracted.text.lower(): e for e in entities}

            for item in data:
                if not isinstance(item, dict):
                    continue

                subject_text = item.get("subject", "")
                predicate_str = item.get("predicate", "")
                object_text = item.get("object", "")

                if not subject_text or not predicate_str or not object_text:
                    continue

                # Map predicate to enum
                try:
                    predicate = RelationshipType(predicate_str.upper())
                except ValueError:
                    predicate = RelationshipType.RELATED_TO

                subject_entity = self._find_entity(subject_text, entity_lookup)
                object_entity = self._find_entity(object_text, entity_lookup)

                relationships.append(
                    ExtractedRelationship(
                        subject_text=subject_text,
                        subject_id=subject_entity.canonical_id if subject_entity else None,
                        subject_type=subject_entity.entity_type if subject_entity else None,
                        predicate=predicate,
                        predicate_raw=predicate_str,
                        object_text=object_text,
                        object_id=object_entity.canonical_id if object_entity else None,
                        object_type=object_entity.entity_type if object_entity else None,
                        indirect_object_text=item.get("indirect_object"),
                        passage_id=passage_id,
                        passage_text=text[:200],
                        confidence=0.85,
                        extraction_method="llm",
                    )
                )

            return relationships

        except (httpx.RequestError, httpx.TimeoutException):
            return []

    def _deduplicate(
        self,
        relationships: list[ExtractedRelationship],
    ) -> list[ExtractedRelationship]:
        """Remove duplicate relationships, keeping highest confidence."""
        seen: dict[str, ExtractedRelationship] = {}

        for rel in relationships:
            # Create a key from the triple
            key = f"{rel.subject_text}|{rel.predicate.value}|{rel.object_text}".lower()

            if key not in seen or rel.confidence > seen[key].confidence:
                seen[key] = rel

        return list(seen.values())

    def extract_from_results(
        self,
        extraction_results: list,  # List of ExtractionResult from entity extraction
        progress_callback=None,
    ) -> Iterator[RelationshipExtractionResult]:
        """Extract relationships from entity extraction results.

        Args:
            extraction_results: Results from EntityExtractor
            progress_callback: Optional callback(current, total)

        Yields:
            RelationshipExtractionResult for each passage
        """
        total = len(extraction_results)

        for i, result in enumerate(extraction_results):
            # Only process passages with 2+ entities
            if len(result.entities) >= 2:
                rel_result = self.extract_relationships(
                    text=result.passage.text,
                    passage_id=result.passage.id,
                    entities=result.entities,
                )
                yield rel_result

            if progress_callback:
                progress_callback(i + 1, total)
