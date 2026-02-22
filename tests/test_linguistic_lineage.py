"""Tests for Issue #46: Linguistic Engine v1.

Validates:
1. Model parsing and serialization (lineage parser)
2. GraphWriter.write_linguistic_lineage contract (mock driver)
3. CLI worldbible languages command behavior
4. Backward compatibility — existing tests still pass
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest


# ============================================================================
# 1. Lineage parser / model tests
# ============================================================================


class TestLineageParser:
    """Test the lineage JSON parser."""

    def test_parse_language_form(self):
        from book_graph_analyzer.worldbible.lineage import parse_language_form
        from book_graph_analyzer.models.worldbuilding import TolkienLanguage

        form = parse_language_form({
            "id": "lf_test",
            "form": "Imladris",
            "language": "Sindarin",
            "gloss": "Deep dale",
        })
        assert form.id == "lf_test"
        assert form.form == "Imladris"
        assert form.language == TolkienLanguage.SINDARIN
        assert form.gloss == "Deep dale"

    def test_parse_language_form_unknown_language(self):
        from book_graph_analyzer.worldbible.lineage import parse_language_form
        from book_graph_analyzer.models.worldbuilding import TolkienLanguage

        form = parse_language_form({
            "id": "lf_x",
            "form": "test",
            "language": "Totally Made Up",
        })
        assert form.language == TolkienLanguage.OTHER

    def test_parse_derivation(self):
        from book_graph_analyzer.worldbible.lineage import parse_derivation
        from book_graph_analyzer.models.worldbuilding import DerivationType

        d = parse_derivation({
            "source_form_id": "a",
            "target_form_id": "b",
            "derivation_type": "cognate",
            "notes": "test note",
        })
        assert d.source_form_id == "a"
        assert d.target_form_id == "b"
        assert d.derivation_type == DerivationType.COGNATE
        assert d.notes == "test note"

    def test_parse_lineage(self):
        from book_graph_analyzer.worldbible.lineage import parse_lineage

        lin = parse_lineage({
            "entity_id": "place_rivendell",
            "forms": [
                {"id": "lf_a", "form": "Imladris", "language": "Sindarin"},
                {"id": "lf_b", "form": "Rivendell", "language": "Common Speech"},
            ],
            "derivations": [
                {"source_form_id": "lf_b", "target_form_id": "lf_a",
                 "derivation_type": "translation"},
            ],
        })
        assert lin.entity_id == "place_rivendell"
        assert len(lin.forms) == 2
        assert len(lin.derivations) == 1
        # entity_id auto-populated on forms
        assert lin.forms[0].entity_id == "place_rivendell"
        # ids normalized to canonical namespace
        assert lin.forms[0].id.startswith("lf:place_rivendell:")
        assert lin.derivations[0].source_form_id.startswith("lf:place_rivendell:")

    def test_load_lineages_from_file(self):
        from book_graph_analyzer.worldbible.lineage import load_lineages_from_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "lineages": [
                    {
                        "entity_id": "test",
                        "forms": [
                            {"id": "lf_1", "form": "Test", "language": "Quenya"},
                        ],
                    }
                ]
            }, f)
            f.flush()
            path = f.name

        try:
            lineages = load_lineages_from_file(path)
            assert len(lineages) == 1
            assert lineages[0].entity_id == "test"
        finally:
            os.unlink(path)

    def test_load_empty_file(self):
        from book_graph_analyzer.worldbible.lineage import load_lineages_from_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"lineages": []}, f)
            f.flush()
            path = f.name

        try:
            lineages = load_lineages_from_file(path)
            assert lineages == []
        finally:
            os.unlink(path)

    def test_roundtrip_serialization(self):
        from book_graph_analyzer.worldbible.lineage import (
            parse_lineage,
            lineages_to_json,
            load_lineages_from_file,
        )

        original = {
            "entity_id": "char_gandalf",
            "forms": [
                {"id": "lf_g1", "form": "Gandalf", "language": "Common Speech",
                 "gloss": "Elf of the wand"},
                {"id": "lf_g2", "form": "Mithrandir", "language": "Sindarin",
                 "gloss": "Grey Pilgrim"},
            ],
            "derivations": [
                {"source_form_id": "lf_g1", "target_form_id": "lf_g2",
                 "derivation_type": "translation"},
            ],
        }
        lineage = parse_lineage(original)
        exported = lineages_to_json([lineage])

        # Re-parse and verify
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(exported, f)
            f.flush()
            path = f.name

        try:
            reloaded = load_lineages_from_file(path)
            assert len(reloaded) == 1
            assert reloaded[0].entity_id == "char_gandalf"
            assert len(reloaded[0].forms) == 2
            assert len(reloaded[0].derivations) == 1
        finally:
            os.unlink(path)


# ============================================================================
# 2. GraphWriter linguistic lineage contract (mock)
# ============================================================================


class TestGraphWriterLinguistic:
    """Test GraphWriter.write_linguistic_lineage with a mock driver."""

    def _make_writer(self):
        from book_graph_analyzer.graph.writer import GraphWriter

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        writer = GraphWriter(driver=mock_driver)
        return writer, mock_session

    def test_write_empty_lineage(self):
        writer, session = self._make_writer()
        from book_graph_analyzer.models.worldbuilding import LinguisticLineage

        result = writer.write_linguistic_lineage(LinguisticLineage(entity_id="x"))
        assert result == 0
        session.run.assert_not_called()

    def test_write_none_lineage(self):
        writer, session = self._make_writer()
        result = writer.write_linguistic_lineage(None)
        assert result == 0

    def test_write_forms_and_derivations(self):
        writer, session = self._make_writer()
        from book_graph_analyzer.worldbible.lineage import parse_lineage

        lineage = parse_lineage({
            "entity_id": "place_rivendell",
            "forms": [
                {"id": "lf_a", "form": "Imladris", "language": "Sindarin",
                 "entity_id": "place_rivendell"},
                {"id": "lf_b", "form": "Rivendell", "language": "Common Speech",
                 "entity_id": "place_rivendell"},
            ],
            "derivations": [
                {"source_form_id": "lf_b", "target_form_id": "lf_a",
                 "derivation_type": "translation"},
            ],
        })

        result = writer.write_linguistic_lineage(lineage)
        assert result == 2

        # Should have: 2 MERGE form calls + 2 HAS_NAME link calls + 1 DERIVED_FROM call
        assert session.run.call_count == 5

    def test_write_batch(self):
        writer, session = self._make_writer()
        from book_graph_analyzer.worldbible.lineage import parse_lineage

        lin1 = parse_lineage({
            "entity_id": "a",
            "forms": [{"id": "f1", "form": "X", "language": "Quenya"}],
        })
        lin2 = parse_lineage({
            "entity_id": "b",
            "forms": [{"id": "f2", "form": "Y", "language": "Sindarin"}],
        })

        result = writer.write_linguistic_lineage_batch([lin1, lin2])
        assert result == 2

    def test_no_longer_raises_not_implemented(self):
        """Verify the NotImplementedError is gone."""
        writer, _ = self._make_writer()
        from book_graph_analyzer.models.worldbuilding import LinguisticLineage

        # Should NOT raise
        writer.write_linguistic_lineage(LinguisticLineage(entity_id="x"))


# ============================================================================
# 3. CLI command tests
# ============================================================================


class TestCLILanguages:
    """Test the worldbible languages CLI command."""

    def test_languages_shows_lineages(self, tmp_path):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        lineage_file = tmp_path / "lineages.json"
        lineage_file.write_text(json.dumps({
            "lineages": [{
                "entity_id": "place_rivendell",
                "forms": [
                    {"id": "lf_a", "form": "Imladris", "language": "Sindarin",
                     "gloss": "Deep dale"},
                    {"id": "lf_b", "form": "Rivendell", "language": "Common Speech"},
                ],
                "derivations": [],
            }]
        }))

        runner = CliRunner()
        result = runner.invoke(main, ["worldbible", "languages", str(lineage_file)])
        assert result.exit_code == 0
        assert "Imladris" in result.output
        assert "Rivendell" in result.output
        assert "place_rivendell" in result.output

    def test_languages_filter_entity(self, tmp_path):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        lineage_file = tmp_path / "lineages.json"
        lineage_file.write_text(json.dumps({
            "lineages": [
                {
                    "entity_id": "place_rivendell",
                    "forms": [{"id": "lf_a", "form": "Imladris", "language": "Sindarin"}],
                },
                {
                    "entity_id": "place_moria",
                    "forms": [{"id": "lf_b", "form": "Moria", "language": "Sindarin"}],
                },
            ]
        }))

        runner = CliRunner()
        result = runner.invoke(main, ["worldbible", "languages", str(lineage_file), "-e", "place_moria"])
        assert result.exit_code == 0
        assert "Moria" in result.output
        assert "Imladris" not in result.output

    def test_languages_empty_file(self, tmp_path):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        lineage_file = tmp_path / "lineages.json"
        lineage_file.write_text(json.dumps({"lineages": []}))

        runner = CliRunner()
        result = runner.invoke(main, ["worldbible", "languages", str(lineage_file)])
        assert "No lineages found" in result.output

    def test_languages_invalid_json(self, tmp_path):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        lineage_file = tmp_path / "bad.json"
        lineage_file.write_text("not json{{{")

        runner = CliRunner()
        result = runner.invoke(main, ["worldbible", "languages", str(lineage_file)])
        assert result.exit_code != 0

    def test_languages_entity_not_found(self, tmp_path):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        lineage_file = tmp_path / "lineages.json"
        lineage_file.write_text(json.dumps({
            "lineages": [{
                "entity_id": "place_rivendell",
                "forms": [{"id": "lf_a", "form": "Test", "language": "Sindarin"}],
            }]
        }))

        runner = CliRunner()
        result = runner.invoke(main, ["worldbible", "languages", str(lineage_file), "-e", "nonexistent"])
        assert "No lineages found for entity" in result.output

    def test_languages_command_still_registered(self):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["worldbible", "--help"])
        assert "languages" in result.output

    def test_languages_join_check_command_registered(self):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["worldbible", "--help"])
        assert "languages-join-check" in result.output

    def test_languages_join_check_strict_namespace_passes(self, tmp_path):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        lineage_file = tmp_path / "lineages.json"
        lineage_file.write_text(json.dumps({
            "lineages": [{
                "entity_id": "place_rivendell",
                "forms": [{"id": "lf_old", "form": "Imladris", "language": "Sindarin"}],
            }]
        }))

        runner = CliRunner()
        result = runner.invoke(main, ["worldbible", "languages-join-check", str(lineage_file), "--strict-namespace"])
        assert result.exit_code == 0
        assert "join success" in result.output.lower()


# ============================================================================
# 4. Backward compatibility
# ============================================================================


class TestBackwardCompat:
    """Ensure existing worldbible and graph writer functionality is intact."""

    def test_worldbible_model_imports(self):
        from book_graph_analyzer.worldbible import (
            WorldBible, WorldRule, WorldBibleCategory,
            CulturalProfile, MagicSystem, GeographyEntry,
            WorldBibleExtractor, ExtractionConfig, PatternMatcher,
        )
        assert WorldBible is not None

    def test_graph_writer_existing_methods(self):
        from book_graph_analyzer.graph.writer import GraphWriter

        for method in [
            "write_entity", "write_entities_batch",
            "write_relationship", "write_relationships_batch",
            "write_passage", "write_extraction_results",
            "write_linguistic_lineage", "write_genealogy_batch",
            "write_editorial_provenance",
        ]:
            assert hasattr(GraphWriter, method), f"Missing: {method}"

    def test_other_stubs_state(self):
        """genealogy empty batch returns 0; editorial provenance no-ops for None source."""
        from book_graph_analyzer.graph.writer import GraphWriter

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        writer = GraphWriter(driver=mock_driver)

        assert writer.write_genealogy_batch([]) == 0
        writer.write_editorial_provenance("x", None)

    def test_existing_kickoff_test_not_implemented_updated(self):
        """The old test expected NotImplementedError for linguistic lineage.
        Verify write_linguistic_lineage no longer raises it."""
        from book_graph_analyzer.graph.writer import GraphWriter

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        writer = GraphWriter(driver=mock_driver)

        # Should NOT raise NotImplementedError
        from book_graph_analyzer.models.worldbuilding import LinguisticLineage
        result = writer.write_linguistic_lineage(LinguisticLineage(entity_id="test"))
        assert result == 0
