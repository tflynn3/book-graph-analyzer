"""Main entity extraction coordinator.

Orchestrates NER extraction, resolution, and storage of entities from text.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ..config import get_settings
from ..ingest.splitter import SentenceSplitter
from ..models.passage import Passage
from .ner import NERPipeline, ExtractedEntity
from .resolver import EntityResolver, ResolvedEntity


@dataclass
class ExtractionResult:
    """Result of extracting entities from a passage."""

    passage: Passage
    entities: list[ResolvedEntity]
    raw_extractions: list[ExtractedEntity]


@dataclass
class BookExtractionStats:
    """Statistics from processing a book."""

    total_passages: int = 0
    total_entities_extracted: int = 0
    total_entities_resolved: int = 0
    new_entities_found: int = 0
    entities_by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    top_characters: list[tuple[str, int]] = field(default_factory=list)
    top_places: list[tuple[str, int]] = field(default_factory=list)


class EntityExtractor:
    """Coordinates entity extraction from books."""

    def __init__(
        self,
        use_llm: bool = True,
        seed_dir: Path | None = None,
    ):
        """Initialize the extractor.

        Args:
            use_llm: Whether to use LLM for enhanced extraction
            seed_dir: Directory containing seed entity files
        """
        self.settings = get_settings()
        self.ner = NERPipeline(use_llm=use_llm)
        self.resolver = EntityResolver(seed_dir=seed_dir)
        self.splitter = SentenceSplitter()

    def extract_from_passage(self, passage: Passage) -> ExtractionResult:
        """Extract and resolve entities from a single passage.

        Args:
            passage: The passage to extract from

        Returns:
            ExtractionResult with resolved entities
        """
        # Extract raw entities
        raw_entities = self.ner.extract_entities(passage.text)

        # Resolve to canonical entities
        resolved = self.resolver.resolve_all(raw_entities)

        return ExtractionResult(
            passage=passage,
            entities=resolved,
            raw_extractions=raw_entities,
        )

    def extract_from_sentences(
        self,
        sentences: list[Passage],
        progress_callback=None,
    ) -> Iterator[ExtractionResult]:
        """Extract entities from a list of sentences/passages.

        Args:
            sentences: List of Passage objects
            progress_callback: Optional callback(current, total) for progress

        Yields:
            ExtractionResult for each passage
        """
        total = len(sentences)
        for i, passage in enumerate(sentences):
            yield self.extract_from_passage(passage)

            if progress_callback:
                progress_callback(i + 1, total)

    def extract_from_text(
        self,
        text: str,
        book_title: str = "Unknown",
        chapter: str | None = None,
    ) -> list[ExtractionResult]:
        """Extract entities from raw text.

        Args:
            text: Raw text to process
            book_title: Title of the book
            chapter: Chapter identifier

        Returns:
            List of ExtractionResult objects
        """
        # Split into sentences
        sentences = self.splitter.split_to_passages(
            text,
            book_title=book_title,
            chapter=chapter,
        )

        # Extract from each
        results = []
        for sentence in sentences:
            results.append(self.extract_from_passage(sentence))

        return results

    def extract_from_file(
        self,
        file_path: Path,
        book_title: str | None = None,
        progress_callback=None,
    ) -> tuple[list[ExtractionResult], BookExtractionStats]:
        """Extract entities from a text file.

        Args:
            file_path: Path to text file
            book_title: Optional title override
            progress_callback: Optional callback(current, total) for progress

        Returns:
            Tuple of (results list, extraction statistics)
        """
        from ..ingest.loader import TextLoader

        loader = TextLoader()
        text = loader.load(file_path)

        title = book_title or file_path.stem.replace("_", " ").title()

        # Split into sentences
        sentences = self.splitter.split_to_passages(text, book_title=title)

        # Track statistics
        stats = BookExtractionStats()
        entity_counts: dict[str, int] = defaultdict(int)
        character_counts: dict[str, int] = defaultdict(int)
        place_counts: dict[str, int] = defaultdict(int)

        results = []
        total = len(sentences)

        for i, passage in enumerate(sentences):
            result = self.extract_from_passage(passage)
            results.append(result)

            # Update stats
            stats.total_passages += 1
            stats.total_entities_extracted += len(result.raw_extractions)

            for entity in result.entities:
                if entity.canonical_id:
                    stats.total_entities_resolved += 1
                    entity_counts[entity.canonical_id] += 1

                    if entity.entity_type == "character":
                        character_counts[entity.canonical_name or entity.canonical_id] += 1
                    elif entity.entity_type == "place":
                        place_counts[entity.canonical_name or entity.canonical_id] += 1
                else:
                    stats.new_entities_found += 1

                stats.entities_by_type[entity.entity_type] += 1

            if progress_callback:
                progress_callback(i + 1, total)

        # Compute top entities
        stats.top_characters = sorted(
            character_counts.items(), key=lambda x: x[1], reverse=True
        )[:20]
        stats.top_places = sorted(
            place_counts.items(), key=lambda x: x[1], reverse=True
        )[:20]

        return results, stats

    def get_unique_entities(
        self,
        results: list[ExtractionResult],
        only_resolved: bool = False,
    ) -> dict[str, list[ResolvedEntity]]:
        """Get unique entities from extraction results, grouped by type.

        Args:
            results: List of extraction results
            only_resolved: If True, only return entities resolved to canonical IDs

        Returns:
            Dict mapping entity type to list of unique resolved entities
        """
        seen_ids: set[str] = set()
        seen_texts: set[str] = set()
        by_type: dict[str, list[ResolvedEntity]] = defaultdict(list)

        for result in results:
            for entity in result.entities:
                # Skip unresolved if requested
                if only_resolved and not entity.canonical_id:
                    continue

                # Use canonical ID if available, else extracted text
                key = entity.canonical_id or entity.extracted.text.lower()

                if key not in seen_ids:
                    seen_ids.add(key)
                    by_type[entity.entity_type].append(entity)

        return dict(by_type)

    def get_new_entity_suggestions(
        self,
        results: list[ExtractionResult],
        min_occurrences: int = 2,
    ) -> list[dict]:
        """Get suggestions for new entities not in the seed database.

        Args:
            results: List of extraction results
            min_occurrences: Minimum times an entity must appear to be suggested

        Returns:
            List of dicts with entity suggestions
        """
        # Count occurrences of unresolved entities
        new_counts: dict[str, dict] = {}

        for result in results:
            for entity in result.entities:
                if entity.is_new:
                    text = entity.extracted.text
                    key = text.lower()

                    if key not in new_counts:
                        new_counts[key] = {
                            "text": text,
                            "type": entity.entity_type,
                            "label": entity.extracted.label,
                            "count": 0,
                            "examples": [],
                        }

                    new_counts[key]["count"] += 1
                    if len(new_counts[key]["examples"]) < 3:
                        # Store context
                        new_counts[key]["examples"].append(
                            result.passage.text[:200] + "..."
                            if len(result.passage.text) > 200
                            else result.passage.text
                        )

        # Filter by minimum occurrences
        suggestions = [
            info for info in new_counts.values() if info["count"] >= min_occurrences
        ]

        # Sort by count
        suggestions.sort(key=lambda x: x["count"], reverse=True)

        return suggestions
