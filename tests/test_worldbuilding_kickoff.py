"""Tests for the Tolkien World-Building kickoff slice (Issue #45).

Validates:
1. New model types import and instantiate correctly
2. Models are exported from the models package
3. GraphWriter has world-building extension method stubs
4. CLI placeholder commands are registered
5. Existing pipeline behaviour is not broken
"""

import pytest


# ============================================================================
# 1. Model imports and instantiation
# ============================================================================


class TestWorldBuildingModels:
    """Test that new model stubs are importable and functional."""

    def test_language_form_creation(self):
        from book_graph_analyzer.models.worldbuilding import (
            LanguageForm,
            TolkienLanguage,
        )

        form = LanguageForm(
            id="lang_imladris",
            form="Imladris",
            language=TolkienLanguage.SINDARIN,
            entity_id="place_rivendell",
            gloss="Deep dale of the cleft",
        )
        assert form.form == "Imladris"
        assert form.language == TolkienLanguage.SINDARIN
        assert form.entity_id == "place_rivendell"

    def test_linguistic_lineage_creation(self):
        from book_graph_analyzer.models.worldbuilding import (
            LanguageForm,
            LinguisticLineage,
            TolkienLanguage,
        )

        lineage = LinguisticLineage(
            entity_id="place_rivendell",
            forms=[
                LanguageForm(
                    id="lang_imladris",
                    form="Imladris",
                    language=TolkienLanguage.SINDARIN,
                ),
                LanguageForm(
                    id="lang_rivendell",
                    form="Rivendell",
                    language=TolkienLanguage.COMMON_SPEECH,
                ),
            ],
        )
        assert len(lineage.forms) == 2
        assert lineage.primary_form(TolkienLanguage.SINDARIN).form == "Imladris"
        assert lineage.primary_form(TolkienLanguage.KHUZDUL) is None

    def test_genealogy_relation_creation(self):
        from book_graph_analyzer.models.worldbuilding import (
            GenealogyRelation,
            GenealogyRelationType,
        )

        rel = GenealogyRelation(
            source_id="char_aragorn",
            target_id="char_elendil",
            relation_type=GenealogyRelationType.DESCENDANT_OF,
            generation_depth=39,
            house="House of Elendil",
            inheritance_traits=["longevity", "foresight"],
        )
        assert rel.generation_depth == 39
        assert rel.house == "House of Elendil"
        assert "longevity" in rel.inheritance_traits

    def test_editorial_layer_creation(self):
        from book_graph_analyzer.models.worldbuilding import (
            AuthorPeriod,
            EditorialLayer,
            EditorialStatus,
        )

        layer = EditorialLayer(
            source_id="src_silmarillion_1977",
            source_title="The Silmarillion",
            editorial_status=EditorialStatus.PUBLISHED,
            author_period=AuthorPeriod.EDITORIAL,
            publication_year=1977,
            authority_weight=0.85,
        )
        assert not layer.is_primary_canon  # EDITORIAL period
        assert layer.authority_weight == 0.85

    def test_editorial_layer_primary_canon(self):
        from book_graph_analyzer.models.worldbuilding import (
            AuthorPeriod,
            EditorialLayer,
            EditorialStatus,
        )

        lotr = EditorialLayer(
            source_id="src_fellowship",
            source_title="The Fellowship of the Ring",
            editorial_status=EditorialStatus.PUBLISHED,
            author_period=AuthorPeriod.MIDDLE,
            publication_year=1954,
        )
        assert lotr.is_primary_canon

    def test_tolkien_sources_registry(self):
        from book_graph_analyzer.models.worldbuilding import TOLKIEN_SOURCES

        assert len(TOLKIEN_SOURCES) >= 7
        titles = {s.source_title for s in TOLKIEN_SOURCES}
        assert "The Hobbit" in titles
        assert "The Silmarillion" in titles

    def test_tolkien_language_enum_completeness(self):
        from book_graph_analyzer.models.worldbuilding import TolkienLanguage

        assert len(TolkienLanguage) >= 10
        assert TolkienLanguage.QUENYA.value == "Quenya"
        assert TolkienLanguage.SINDARIN.value == "Sindarin"

    def test_derivation_type_enum(self):
        from book_graph_analyzer.models.worldbuilding import DerivationType

        assert DerivationType.TRANSLATION.value == "translation"
        assert DerivationType.COGNATE.value == "cognate"


# ============================================================================
# 2. Package-level exports
# ============================================================================


class TestPackageExports:
    """Test that new models are accessible from the models package."""

    def test_models_package_exports_linguistic_lineage(self):
        from book_graph_analyzer.models import LinguisticLineage

        assert LinguisticLineage is not None

    def test_models_package_exports_genealogy_relation(self):
        from book_graph_analyzer.models import GenealogyRelation

        assert GenealogyRelation is not None

    def test_models_package_exports_editorial_layer(self):
        from book_graph_analyzer.models import EditorialLayer

        assert EditorialLayer is not None

    def test_models_package_exports_language_form(self):
        from book_graph_analyzer.models import LanguageForm

        assert LanguageForm is not None

    def test_existing_exports_still_work(self):
        """Verify backward compatibility of models package."""
        from book_graph_analyzer.models import (
            Character,
            Concept,
            Event,
            Object,
            Passage,
            Place,
        )

        assert Character is not None
        assert Place is not None
        assert Object is not None
        assert Event is not None
        assert Concept is not None
        assert Passage is not None


# ============================================================================
# 3. GraphWriter extension stubs
# ============================================================================


class TestGraphWriterStubs:
    """Test that GraphWriter has the new world-building method stubs."""

    def test_has_write_linguistic_lineage(self):
        from book_graph_analyzer.graph.writer import GraphWriter

        writer = GraphWriter.__new__(GraphWriter)
        assert hasattr(writer, "write_linguistic_lineage")
        assert callable(writer.write_linguistic_lineage)

    def test_has_write_genealogy_batch(self):
        from book_graph_analyzer.graph.writer import GraphWriter

        writer = GraphWriter.__new__(GraphWriter)
        assert hasattr(writer, "write_genealogy_batch")
        assert callable(writer.write_genealogy_batch)

    def test_has_write_editorial_provenance(self):
        from book_graph_analyzer.graph.writer import GraphWriter

        writer = GraphWriter.__new__(GraphWriter)
        assert hasattr(writer, "write_editorial_provenance")
        assert callable(writer.write_editorial_provenance)

    def test_linguistic_lineage_implemented(self):
        """write_linguistic_lineage is now implemented (Issue #46)."""
        from book_graph_analyzer.graph.writer import GraphWriter
        from unittest.mock import MagicMock

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        writer = GraphWriter(driver=mock_driver)

        # Should return 0 for None/empty, NOT raise NotImplementedError
        assert writer.write_linguistic_lineage(None) == 0

    def test_genealogy_batch_handles_empty_list(self):
        from book_graph_analyzer.graph.writer import GraphWriter

        from unittest.mock import MagicMock

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        writer = GraphWriter(driver=mock_driver)
        assert writer.write_genealogy_batch([]) == 0

    def test_editorial_provenance_raises_not_implemented(self):
        from book_graph_analyzer.graph.writer import GraphWriter

        writer = GraphWriter.__new__(GraphWriter)
        with pytest.raises(NotImplementedError, match="Issue #48"):
            writer.write_editorial_provenance("entity_1", None)

    def test_existing_methods_still_exist(self):
        """Verify existing GraphWriter methods are not broken."""
        from book_graph_analyzer.graph.writer import GraphWriter

        for method_name in [
            "write_entity",
            "write_entities_batch",
            "write_relationship",
            "write_relationships_batch",
            "write_passage",
            "write_extraction_results",
            "write_book_style",
            "write_character_voice",
            "write_event",
            "init_era_chain",
            "query_at_time",
        ]:
            assert hasattr(GraphWriter, method_name), f"Missing method: {method_name}"


# ============================================================================
# 4. CLI command registration
# ============================================================================


class TestCLICommands:
    """Test that placeholder CLI commands are registered under correct groups."""

    def test_worldbible_languages_registered(self):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["worldbible", "--help"])
        assert "languages" in result.output

    def test_lore_genealogy_registered(self):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["lore", "--help"])
        assert "genealogy" in result.output

    def test_corpus_sources_registered(self):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["corpus", "--help"])
        assert "sources" in result.output

    def test_pipeline_worldbuilding_registered(self):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["pipeline", "--help"])
        assert "worldbuilding" in result.output

    def test_existing_commands_still_registered(self):
        """Verify existing CLI commands are not broken."""
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        runner = CliRunner()

        # Top-level commands
        result = runner.invoke(main, ["--help"])
        for cmd in ["ingest", "extract", "graph", "style", "voice", "pipeline",
                     "corpus", "worldbible", "lore", "status", "analyze"]:
            assert cmd in result.output, f"Missing top-level command: {cmd}"


# ============================================================================
# 5. Backward compatibility
# ============================================================================


class TestBackwardCompatibility:
    """Ensure existing pipeline components still work correctly."""

    def test_entity_models_unchanged(self):
        from book_graph_analyzer.models.entities import (
            Character,
            Concept,
            EntityBase,
            Event,
            Object,
            Place,
        )

        char = Character(id="test", canonical_name="Test")
        assert char.id == "test"
        assert char.aliases == []

    def test_relationship_models_unchanged(self):
        from book_graph_analyzer.models.relationships import (
            ExtractedRelationship,
            RelationshipTriple,
            RelationshipType,
        )

        assert RelationshipType.PARENT_OF.value == "PARENT_OF"
        assert RelationshipType.SPOKE_WITH.value == "SPOKE_WITH"

    def test_passage_model_unchanged(self):
        from book_graph_analyzer.models.passage import Passage

        p = Passage(
            id="test",
            text="Test passage",
            book="Test Book",
            chapter="1",
            chapter_num=1,
            paragraph_num=1,
            sentence_num=1,
            char_offset=0,
        )
        assert p.text == "Test passage"
