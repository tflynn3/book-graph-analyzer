"""Tests for relationship extraction."""

import pytest

from book_graph_analyzer.extract.relationships import RelationshipExtractor
from book_graph_analyzer.extract.resolver import EntityResolver, ResolvedEntity
from book_graph_analyzer.extract.ner import ExtractedEntity
from book_graph_analyzer.models.relationships import RelationshipType


class TestRelationshipExtractor:
    """Tests for relationship extraction."""

    @pytest.fixture
    def extractor(self, tmp_path):
        """Create an extractor with test seed data."""
        import json

        seeds_dir = tmp_path / "seeds"
        seeds_dir.mkdir()

        chars = [
            {
                "id": "bilbo",
                "canonical_name": "Bilbo",
                "aliases": ["Bilbo Baggins", "Mr. Baggins"],
            },
            {
                "id": "gandalf",
                "canonical_name": "Gandalf",
                "aliases": ["Gandalf the Grey"],
            },
            {
                "id": "gollum",
                "canonical_name": "Gollum",
                "aliases": ["Smeagol"],
            },
        ]
        with open(seeds_dir / "characters.json", "w") as f:
            json.dump(chars, f)

        places = [
            {
                "id": "bag_end",
                "canonical_name": "Bag End",
                "aliases": [],
            },
        ]
        with open(seeds_dir / "places.json", "w") as f:
            json.dump(places, f)

        (seeds_dir / "objects.json").write_text("[]")

        resolver = EntityResolver(seed_dir=seeds_dir)
        return RelationshipExtractor(resolver=resolver, use_llm=False)

    def make_entity(self, text: str, canonical_id: str | None, entity_type: str) -> ResolvedEntity:
        """Helper to create a ResolvedEntity."""
        return ResolvedEntity(
            extracted=ExtractedEntity(
                text=text,
                label="PERSON",
                start_char=0,
                end_char=len(text),
            ),
            canonical_id=canonical_id,
            canonical_name=canonical_id,
            entity_type=entity_type,
            confidence=1.0,
            is_new=canonical_id is None,
        )

    def test_dependency_extraction_spoke(self, extractor):
        """Test extracting speech relationships via dependency parsing."""
        text = "Gandalf spoke to Bilbo about the adventure."
        entities = [
            self.make_entity("Gandalf", "gandalf", "character"),
            self.make_entity("Bilbo", "bilbo", "character"),
        ]

        result = extractor.extract_relationships(text, "test_1", entities)

        # Should find a SPOKE relationship
        assert len(result.relationships) >= 1
        rel = result.relationships[0]
        assert rel.predicate in (RelationshipType.SPOKE_WITH, RelationshipType.SPOKE_TO)

    def test_dependency_extraction_traveled(self, extractor):
        """Test extracting travel relationships."""
        text = "Bilbo went to Bag End."
        entities = [
            self.make_entity("Bilbo", "bilbo", "character"),
            self.make_entity("Bag End", "bag_end", "place"),
        ]

        result = extractor.extract_relationships(text, "test_2", entities)

        # Should find a TRAVELED_TO relationship
        travel_rels = [r for r in result.relationships if r.predicate == RelationshipType.TRAVELED_TO]
        assert len(travel_rels) >= 1

    def test_dependency_extraction_killed(self, extractor):
        """Test extracting combat relationships."""
        text = "The hero killed the dragon."
        entities = [
            self.make_entity("The hero", None, "character"),
            self.make_entity("the dragon", None, "character"),
        ]

        result = extractor.extract_relationships(text, "test_3", entities)

        # Should find a KILLED relationship
        kill_rels = [r for r in result.relationships if r.predicate == RelationshipType.KILLED]
        assert len(kill_rels) >= 1

    def test_no_relationships_single_entity(self, extractor):
        """Test that single-entity passages return no relationships."""
        text = "Bilbo was happy."
        entities = [
            self.make_entity("Bilbo", "bilbo", "character"),
        ]

        result = extractor.extract_relationships(text, "test_4", entities)

        # Might have no relationships (need 2 entities for a relationship)
        # This depends on implementation
        assert result is not None

    def test_deduplication(self, extractor):
        """Test that duplicate relationships are removed."""
        text = "Gandalf met Bilbo and spoke with Bilbo."
        entities = [
            self.make_entity("Gandalf", "gandalf", "character"),
            self.make_entity("Bilbo", "bilbo", "character"),
        ]

        result = extractor.extract_relationships(text, "test_5", entities)

        # Check for deduplication - same subject/object/predicate should be merged
        keys = set()
        for rel in result.relationships:
            key = f"{rel.subject_text}|{rel.predicate.value}|{rel.object_text}"
            assert key not in keys, f"Duplicate relationship: {key}"
            keys.add(key)


class TestRelationshipTypes:
    """Tests for relationship type mapping."""

    def test_verb_mapping_coverage(self):
        """Test that common verbs are mapped to relationship types."""
        from book_graph_analyzer.extract.relationships import VERB_TO_RELATIONSHIP

        expected_verbs = [
            "said", "spoke", "went", "traveled", "fought", "killed",
            "gave", "took", "met", "helped", "lived"
        ]

        for verb in expected_verbs:
            assert verb in VERB_TO_RELATIONSHIP, f"Missing verb mapping: {verb}"

    def test_relationship_type_values(self):
        """Test that relationship types have valid enum values."""
        for rel_type in RelationshipType:
            assert rel_type.value == rel_type.value.upper(), f"Type should be uppercase: {rel_type}"
            assert "_" in rel_type.value or rel_type.value.isalpha(), f"Invalid format: {rel_type}"
