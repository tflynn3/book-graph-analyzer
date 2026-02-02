"""Tests for the entity extraction pipeline."""

import pytest

from book_graph_analyzer.extract.ner import NERPipeline, ExtractedEntity
from book_graph_analyzer.extract.resolver import EntityResolver, EntityDatabase
from book_graph_analyzer.extract.extractor import EntityExtractor
from book_graph_analyzer.models.entities import Character, Place, Object


class TestNERPipeline:
    """Tests for NER extraction."""

    def test_spacy_extraction(self):
        """Test basic spaCy entity extraction."""
        ner = NERPipeline(use_llm=False)
        text = "Gandalf the Grey arrived at Bag End to visit Bilbo."

        entities = ner.extract_entities(text)

        # Should find some entities (exact results depend on model)
        assert len(entities) > 0
        texts = [e.text for e in entities]
        # Gandalf and Bilbo are likely to be detected as PERSON
        assert any("Gandalf" in t for t in texts) or any("Bilbo" in t for t in texts)

    def test_pattern_extraction(self):
        """Test pattern-based extraction for titles."""
        ner = NERPipeline(use_llm=False)
        text = "The King of Gondor spoke to the Grey Wizard."

        entities = ner._extract_patterns(text)

        # Should capture the title patterns
        texts = [e.text.lower() for e in entities]
        assert any("king of gondor" in t for t in texts)

    def test_deduplication(self):
        """Test that overlapping entities are deduplicated."""
        ner = NERPipeline(use_llm=False)

        entities = [
            ExtractedEntity("Gandalf", "PERSON", 0, 7, 0.8, "spacy"),
            ExtractedEntity("Gandalf the Grey", "PERSON", 0, 16, 0.9, "pattern"),
        ]

        deduped = ner._deduplicate(entities)

        # Should keep the higher confidence, more specific one
        assert len(deduped) == 1
        assert deduped[0].text == "Gandalf the Grey"


class TestEntityResolver:
    """Tests for entity resolution."""

    @pytest.fixture
    def resolver(self, tmp_path):
        """Create a resolver with test seed data."""
        # Create test seed files
        import json

        seeds_dir = tmp_path / "seeds"
        seeds_dir.mkdir()

        # Characters
        chars = [
            {
                "id": "gandalf",
                "canonical_name": "Gandalf",
                "aliases": ["Gandalf the Grey", "Mithrandir", "the Grey Pilgrim"],
                "race": "Maia",
            },
            {
                "id": "bilbo_baggins",
                "canonical_name": "Bilbo Baggins",
                "aliases": ["Bilbo", "Mr. Baggins"],
                "race": "Hobbit",
            },
        ]
        with open(seeds_dir / "characters.json", "w") as f:
            json.dump(chars, f)

        # Places
        places = [
            {
                "id": "the_shire",
                "canonical_name": "The Shire",
                "aliases": ["Shire"],
                "type": "region",
            },
        ]
        with open(seeds_dir / "places.json", "w") as f:
            json.dump(places, f)

        return EntityResolver(seed_dir=seeds_dir)

    def test_exact_match(self, resolver):
        """Test exact name matching."""
        entity = ExtractedEntity("Gandalf", "PERSON", 0, 7, 1.0, "spacy")
        resolved = resolver.resolve(entity)

        assert resolved.canonical_id == "gandalf"
        assert resolved.canonical_name == "Gandalf"
        assert resolved.entity_type == "character"
        assert not resolved.is_new

    def test_alias_match(self, resolver):
        """Test alias matching."""
        entity = ExtractedEntity("Mithrandir", "PERSON", 0, 10, 1.0, "spacy")
        resolved = resolver.resolve(entity)

        assert resolved.canonical_id == "gandalf"
        assert resolved.canonical_name == "Gandalf"

    def test_article_stripping(self, resolver):
        """Test that articles are stripped for matching."""
        entity = ExtractedEntity("the Shire", "PLACE", 0, 9, 1.0, "spacy")
        resolved = resolver.resolve(entity)

        assert resolved.canonical_id == "the_shire"

    def test_fuzzy_match(self, resolver):
        """Test fuzzy matching for slight misspellings."""
        entity = ExtractedEntity("Gandaf", "PERSON", 0, 6, 1.0, "spacy")
        resolved = resolver.resolve(entity)

        # Should fuzzy match to Gandalf
        assert resolved.canonical_id == "gandalf"
        assert resolved.confidence < 1.0  # Lower confidence for fuzzy

    def test_new_entity(self, resolver):
        """Test handling of unknown entities."""
        entity = ExtractedEntity("Tom Bombadil", "PERSON", 0, 12, 1.0, "spacy")
        resolved = resolver.resolve(entity)

        assert resolved.canonical_id is None
        assert resolved.is_new

    def test_type_inference(self, resolver):
        """Test type inference for new entities."""
        # Should infer place from text
        entity = ExtractedEntity("the Great Forest", "UNKNOWN", 0, 16, 1.0, "spacy")
        resolved = resolver.resolve(entity)

        assert resolved.entity_type == "place"


class TestEntityExtractor:
    """Tests for the main extractor."""

    def test_extract_from_text(self, tmp_path):
        """Test end-to-end extraction from text."""
        # Create minimal seed data
        import json

        seeds_dir = tmp_path / "seeds"
        seeds_dir.mkdir()

        chars = [
            {
                "id": "gandalf",
                "canonical_name": "Gandalf",
                "aliases": ["Gandalf the Grey"],
            },
        ]
        with open(seeds_dir / "characters.json", "w") as f:
            json.dump(chars, f)

        # Create empty files for places/objects
        (seeds_dir / "places.json").write_text("[]")
        (seeds_dir / "objects.json").write_text("[]")

        extractor = EntityExtractor(use_llm=False, seed_dir=seeds_dir)

        text = "Gandalf arrived at the door."
        results = extractor.extract_from_text(text, book_title="Test")

        assert len(results) > 0
        # Should have extracted at least one entity
        all_entities = [e for r in results for e in r.entities]
        assert len(all_entities) >= 1

    def test_get_unique_entities(self, tmp_path):
        """Test deduplication of entities across passages."""
        import json

        seeds_dir = tmp_path / "seeds"
        seeds_dir.mkdir()

        chars = [
            {
                "id": "gandalf",
                "canonical_name": "Gandalf",
                "aliases": ["Gandalf", "the wizard"],
            },
        ]
        with open(seeds_dir / "characters.json", "w") as f:
            json.dump(chars, f)

        (seeds_dir / "places.json").write_text("[]")
        (seeds_dir / "objects.json").write_text("[]")

        extractor = EntityExtractor(use_llm=False, seed_dir=seeds_dir)

        text = "Gandalf spoke. Gandalf laughed. Gandalf left."
        results = extractor.extract_from_text(text, book_title="Test")

        unique = extractor.get_unique_entities(results, only_resolved=True)

        # Should only have one Gandalf despite multiple mentions
        character_count = len(unique.get("character", []))
        assert character_count <= 1  # May be 0 or 1 depending on extraction


class TestEntityDatabase:
    """Tests for the entity database."""

    def test_add_and_lookup(self):
        """Test adding and looking up entities."""
        db = EntityDatabase()

        char = Character(
            id="gandalf",
            canonical_name="Gandalf",
            aliases=["Mithrandir", "the Grey"],
        )
        db.add_character(char)

        # Lookup by canonical name
        entity_id, entity_type, conf = db.lookup("Gandalf")
        assert entity_id == "gandalf"
        assert entity_type == "character"

        # Lookup by alias
        entity_id, entity_type, conf = db.lookup("Mithrandir")
        assert entity_id == "gandalf"

    def test_lookup_not_found(self):
        """Test lookup for non-existent entity."""
        db = EntityDatabase()

        entity_id, entity_type, conf = db.lookup("Unknown Entity")
        assert entity_id is None
        assert conf == 0.0
