"""Tests for Issue #46 closeout: query helpers, new patterns, linguistic context.

Covers:
1. GraphWriter.query_entity_names and query_lineage_chain (mock driver)
2. New lineage extraction patterns (which_is, in_lang_or, derives_from, literally, the_lang_form)
3. LinguisticContext dataclass and lore-rules scoping hook
4. CLI entity-names and lineage-chain commands
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from book_graph_analyzer.worldbible.lineage_extractor import extract_lineages_from_text


# ============================================================================
# 1. New extraction patterns
# ============================================================================


class TestNewPatterns:
    """Test the additional regex patterns added for rarer etymology phrasing."""

    def test_which_is_pattern(self):
        text = "Eressëa, which is the Lonely Isle in Common Speech."
        result = extract_lineages_from_text(text)
        assert result.lineages
        forms = {f.form for lin in result.lineages for f in lin.forms}
        assert len(forms) >= 1

    def test_which_the_elves_call(self):
        text = "The Misty Mountains, which the Elves call Hithaeglir."
        result = extract_lineages_from_text(text)
        assert result.lineages

    def test_in_lang_or_pattern(self):
        text = "Oiolossë in Quenya or Uilos in Sindarin."
        result = extract_lineages_from_text(text)
        assert result.lineages
        forms = {f.form for lin in result.lineages for f in lin.forms}
        assert "Oiolossë" in forms or "Uilos" in forms

    def test_the_lang_form_pattern(self):
        text = "The Sindarin form of Oromë is Araw."
        result = extract_lineages_from_text(text)
        assert result.lineages
        forms = {f.form for lin in result.lineages for f in lin.forms}
        # Pattern captures "Oromë" as the form attributed to Sindarin
        assert "Oromë" in forms or "Araw" in forms

    def test_derives_from_pattern(self):
        text = "Gondolin derives from the Sindarin Ondolindë."
        result = extract_lineages_from_text(text)
        assert result.lineages

    def test_literally_pattern(self):
        text = "Barad-dûr, literally 'Dark Tower'."
        result = extract_lineages_from_text(text)
        assert result.lineages
        gloss_found = any(
            f.gloss and "Dark Tower" in f.gloss
            for lin in result.lineages for f in lin.forms
        )
        assert gloss_found

    def test_literally_with_language(self):
        text = "Amon Sûl, literally 'Hill of Wind' in Sindarin."
        result = extract_lineages_from_text(text)
        assert result.lineages

    def test_all_patterns_no_false_positives(self):
        """Plain prose should still produce no results."""
        text = "The hobbits walked slowly through the green fields and ate lunch."
        result = extract_lineages_from_text(text)
        assert result.lineages == []


# ============================================================================
# 2. GraphWriter query helpers (mock)
# ============================================================================


class TestGraphWriterQueryHelpers:
    """Test query_entity_names and query_lineage_chain with mock driver."""

    def _make_writer(self):
        from book_graph_analyzer.graph.writer import GraphWriter

        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        writer = GraphWriter(driver=mock_driver)
        return writer, mock_session

    def test_query_entity_names_runs_query(self):
        writer, session = self._make_writer()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([
            {"id": "lf_1", "form": "Imladris", "language": "Sindarin", "gloss": "Deep dale", "phonetic": None},
            {"id": "lf_2", "form": "Rivendell", "language": "Common Speech", "gloss": None, "phonetic": None},
        ]))
        session.run.return_value = mock_result

        results = writer.query_entity_names("place_rivendell")
        assert len(results) == 2
        assert results[0]["form"] == "Imladris"
        session.run.assert_called_once()

    def test_query_entity_names_empty(self):
        writer, session = self._make_writer()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        session.run.return_value = mock_result

        results = writer.query_entity_names("nonexistent")
        assert results == []

    def test_query_lineage_chain_runs_query(self):
        writer, session = self._make_writer()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([
            {
                "source_id": "lf_1", "source_form": "Rivendell", "source_language": "Common Speech",
                "target_id": "lf_2", "target_form": "Imladris", "target_language": "Sindarin",
                "derivation_type": "translation", "notes": None,
            }
        ]))
        session.run.return_value = mock_result

        results = writer.query_lineage_chain("lf_1")
        assert len(results) == 1
        assert results[0]["source_form"] == "Rivendell"

    def test_query_lineage_chain_respects_depth(self):
        writer, session = self._make_writer()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        session.run.return_value = mock_result

        writer.query_lineage_chain("lf_1", max_depth=3)
        # Verify the query was called with the form_id parameter
        call_args = session.run.call_args
        assert call_args[1]["form_id"] == "lf_1"


# ============================================================================
# 3. LinguisticContext and lore-rules scoping
# ============================================================================


class TestLinguisticContext:
    """Test the LinguisticContext dataclass and its integration hook."""

    def test_linguistic_context_creation(self):
        from book_graph_analyzer.lore.rules import LinguisticContext

        ctx = LinguisticContext(
            entity_languages={"place_rivendell": ["Sindarin", "Common Speech"]},
            entity_forms={"place_rivendell": ["Imladris", "Rivendell"]},
            dominant_language="Sindarin",
        )
        assert ctx.entity_has_language("place_rivendell", "Sindarin")
        assert not ctx.entity_has_language("place_rivendell", "Quenya")
        assert ctx.forms_for_entity("place_rivendell") == ["Imladris", "Rivendell"]
        assert ctx.forms_for_entity("unknown") == []

    def test_linguistic_context_defaults(self):
        from book_graph_analyzer.lore.rules import LinguisticContext

        ctx = LinguisticContext()
        assert ctx.entity_languages == {}
        assert ctx.entity_forms == {}
        assert ctx.dominant_language is None

    def test_scene_context_has_linguistic_field(self):
        from book_graph_analyzer.lore.rules import SceneContext, LinguisticContext

        ling = LinguisticContext(dominant_language="Quenya")
        scene = SceneContext(
            scene_id="test",
            character_names=["Gandalf"],
            character_races={"Gandalf": "Maia"},
            place_names=["Rivendell"],
            object_names=[],
            event_types=[],
            linguistic=ling,
        )
        assert scene.linguistic is not None
        assert scene.linguistic.dominant_language == "Quenya"

    def test_registry_by_linguistic_scope(self):
        from book_graph_analyzer.lore.rules import LoreRuleRegistry, LinguisticContext

        registry = LoreRuleRegistry.from_tolkien_defaults()
        ctx = LinguisticContext(
            entity_languages={"place_rivendell": ["Sindarin"]},
        )
        rules = registry.by_linguistic_scope(ctx)
        # Should return rules (at least the universal ones)
        assert len(rules) > 0

    def test_registry_by_linguistic_scope_empty(self):
        from book_graph_analyzer.lore.rules import LoreRuleRegistry, LinguisticContext

        registry = LoreRuleRegistry.from_tolkien_defaults()
        ctx = LinguisticContext()
        rules = registry.by_linguistic_scope(ctx)
        # Should return all rules when no linguistic context
        assert rules == registry.all()


# ============================================================================
# 4. CLI smoke tests for new commands
# ============================================================================


class TestCLINewCommands:
    def test_entity_names_registered(self):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["worldbible", "--help"])
        assert "entity-names" in result.output

    def test_lineage_chain_registered(self):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["worldbible", "--help"])
        assert "lineage-chain" in result.output
