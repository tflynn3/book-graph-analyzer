"""Named Entity Recognition pipeline.

Combines spaCy NER with LLM-based extraction for Tolkien-specific entities.
"""

import json
import re
from dataclasses import dataclass
from typing import Literal

import httpx
import spacy
from spacy.tokens import Doc

from ..config import get_settings


@dataclass
class ExtractedEntity:
    """A raw extracted entity before resolution."""

    text: str
    label: Literal["PERSON", "PLACE", "OBJECT", "ORG", "EVENT", "UNKNOWN"]
    start_char: int
    end_char: int
    confidence: float = 1.0
    source: Literal["spacy", "llm", "pattern"] = "spacy"


class NERPipeline:
    """Named Entity Recognition pipeline combining multiple extraction methods."""

    # Patterns for Tolkien-specific entity detection
    TITLE_PATTERNS = [
        r"\b(King|Queen|Lord|Lady|Prince|Princess|Chief|Captain|Steward)\s+(?:of\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"\b(the\s+)?(Grey|White|Dark|Black|High|Great)\s+(Wizard|King|Lord|Lady|Enemy|Rider)",
    ]

    # Common epithets and titles in Tolkien
    EPITHET_PATTERNS = [
        r"\b(the\s+)?(Ring-?bearer|Halfling|Perian|Dwarf-friend|Elf-friend)",
        r"\b(the\s+)?(Necromancer|Enemy|Dark Lord|Shadow)",
        r"\b(Gandalf|Saruman|Radagast)\s+(the\s+)?(Grey|White|Brown)",
    ]

    def __init__(self, use_llm: bool = True):
        """Initialize the NER pipeline.

        Args:
            use_llm: Whether to use LLM for enhanced extraction
        """
        self.settings = get_settings()
        self.use_llm = use_llm
        self._nlp = None
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> list[re.Pattern]:
        """Compile regex patterns for pattern-based extraction."""
        patterns = []
        for p in self.TITLE_PATTERNS + self.EPITHET_PATTERNS:
            patterns.append(re.compile(p, re.IGNORECASE))
        return patterns

    @property
    def nlp(self) -> spacy.Language:
        """Lazy-load spaCy model."""
        if self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Model not installed, download it
                import subprocess

                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def extract_entities(self, text: str) -> list[ExtractedEntity]:
        """Extract entities from text using all available methods.

        Args:
            text: Input text to extract entities from

        Returns:
            List of extracted entities (may contain duplicates)
        """
        entities = []

        # 1. spaCy NER
        entities.extend(self._extract_spacy(text))

        # 2. Pattern-based extraction
        entities.extend(self._extract_patterns(text))

        # 3. LLM-based extraction (if enabled)
        if self.use_llm:
            llm_entities = self._extract_llm(text)
            if llm_entities:
                entities.extend(llm_entities)

        # Deduplicate overlapping entities
        entities = self._deduplicate(entities)

        return entities

    def _extract_spacy(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using spaCy NER."""
        doc = self.nlp(text)
        entities = []

        # Map spaCy labels to our simplified labels
        label_map = {
            "PERSON": "PERSON",
            "GPE": "PLACE",
            "LOC": "PLACE",
            "FAC": "PLACE",
            "ORG": "ORG",
            "EVENT": "EVENT",
            "WORK_OF_ART": "OBJECT",
            "PRODUCT": "OBJECT",
        }

        for ent in doc.ents:
            label = label_map.get(ent.label_, "UNKNOWN")
            if label != "UNKNOWN":
                entities.append(
                    ExtractedEntity(
                        text=ent.text,
                        label=label,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        source="spacy",
                    )
                )

        # Also extract proper nouns that might be missed
        for token in doc:
            if token.pos_ == "PROPN" and not any(
                e.start_char <= token.idx < e.end_char for e in entities
            ):
                # Check if it's part of a noun chunk
                for chunk in doc.noun_chunks:
                    if token in chunk:
                        chunk_text = chunk.text.strip()
                        if chunk_text[0].isupper():
                            entities.append(
                                ExtractedEntity(
                                    text=chunk_text,
                                    label="UNKNOWN",
                                    start_char=chunk.start_char,
                                    end_char=chunk.end_char,
                                    confidence=0.7,
                                    source="spacy",
                                )
                            )
                        break

        return entities

    def _extract_patterns(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using regex patterns."""
        entities = []

        for pattern in self._patterns:
            for match in pattern.finditer(text):
                entities.append(
                    ExtractedEntity(
                        text=match.group(0),
                        label="PERSON",  # Most patterns capture character references
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=0.9,
                        source="pattern",
                    )
                )

        return entities

    def _extract_llm(self, text: str) -> list[ExtractedEntity]:
        """Extract entities using local LLM via Ollama."""
        prompt = f"""Extract all named entities from this text from Tolkien's works.
Return a JSON array of objects with keys: "text" (the entity), "type" (PERSON, PLACE, OBJECT, or EVENT).

Only include:
- PERSON: Characters, peoples, races (e.g., "Frodo", "the Dwarves", "Gandalf the Grey")
- PLACE: Locations, regions, buildings (e.g., "Rivendell", "the Shire", "Mount Doom")
- OBJECT: Significant items (e.g., "the Ring", "Sting", "the Arkenstone")
- EVENT: Named events (e.g., "the Battle of Five Armies", "the Council of Elrond")

Text: "{text}"

JSON array (no explanation, just the array):"""

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

            # Try to extract JSON from response
            # Sometimes LLM wraps it in markdown code blocks
            if "```" in response_text:
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
                if json_match:
                    response_text = json_match.group(1)

            # Parse JSON
            try:
                entities_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to find array in response
                array_match = re.search(r"\[[\s\S]*\]", response_text)
                if array_match:
                    entities_data = json.loads(array_match.group(0))
                else:
                    return []

            # Convert to ExtractedEntity objects
            entities = []
            for item in entities_data:
                if isinstance(item, dict) and "text" in item:
                    entity_text = item["text"]
                    # Find position in original text
                    idx = text.find(entity_text)
                    if idx == -1:
                        # Try case-insensitive search
                        idx = text.lower().find(entity_text.lower())

                    label = item.get("type", "UNKNOWN").upper()
                    if label not in ["PERSON", "PLACE", "OBJECT", "EVENT"]:
                        label = "UNKNOWN"

                    entities.append(
                        ExtractedEntity(
                            text=entity_text,
                            label=label,
                            start_char=idx if idx >= 0 else 0,
                            end_char=(idx + len(entity_text)) if idx >= 0 else len(entity_text),
                            confidence=0.85,
                            source="llm",
                        )
                    )

            return entities

        except (httpx.RequestError, httpx.TimeoutException):
            # LLM not available, continue without
            return []

    def _deduplicate(self, entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        """Remove duplicate and overlapping entities, preferring higher confidence."""
        if not entities:
            return []

        # Sort by start position, then by confidence (descending)
        sorted_entities = sorted(entities, key=lambda e: (e.start_char, -e.confidence))

        result = []
        for entity in sorted_entities:
            # Check if this entity overlaps with any already accepted entity
            overlaps = False
            for accepted in result:
                # Check for overlap
                if not (entity.end_char <= accepted.start_char or entity.start_char >= accepted.end_char):
                    overlaps = True
                    # If this one has higher confidence and is more specific, replace
                    if (
                        entity.confidence > accepted.confidence
                        and len(entity.text) >= len(accepted.text)
                    ):
                        result.remove(accepted)
                        result.append(entity)
                    break

            if not overlaps:
                result.append(entity)

        return sorted(result, key=lambda e: e.start_char)
