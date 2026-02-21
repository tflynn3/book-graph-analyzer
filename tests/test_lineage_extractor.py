"""Tests for raw-text linguistic lineage extraction (#46)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from book_graph_analyzer.worldbible.lineage_extractor import (
    ExtractionResult,
    extract_lineages_from_text,
    lineage_alias_hints,
)
from book_graph_analyzer.worldbible.lineage import lineages_to_json, parse_lineage
from book_graph_analyzer.models.worldbuilding import TolkienLanguage


# ---------------------------------------------------------------------------
# Regex extraction tests
# ---------------------------------------------------------------------------

class TestRegexExtraction:
    def test_called_in_pattern(self):
        text = "Imladris, called Rivendell in Common Speech, lay hidden in the valley."
        result = extract_lineages_from_text(text)
        assert result.lineages, "Should extract at least one lineage"
        forms = {f.form for lin in result.lineages for f in lin.forms}
        assert "Imladris" in forms
        assert "Rivendell" in forms

    def test_name_for_pattern(self):
        text = "The Sindarin name for Rivendell is Imladris."
        result = extract_lineages_from_text(text)
        assert result.lineages
        forms = {f.form for lin in result.lineages for f in lin.forms}
        assert "Imladris" in forms

    def test_parenthetical_pattern(self):
        text = "Orodruin (Sindarin: 'Mountain of Blazing Fire') loomed ahead."
        result = extract_lineages_from_text(text)
        assert result.lineages
        lin = result.lineages[0]
        assert any(f.form == "Orodruin" for f in lin.forms)
        assert any(f.gloss for f in lin.forms)

    def test_means_in_pattern(self):
        text = "Mithrandir means 'Grey Pilgrim' in Sindarin."
        result = extract_lineages_from_text(text)
        assert result.lineages
        forms = {f.form for lin in result.lineages for f in lin.forms}
        assert "Mithrandir" in forms

    def test_from_the_pattern(self):
        text = "The name derives from the Quenya Oiolossë meaning 'Ever-snow-white'."
        result = extract_lineages_from_text(text)
        assert result.lineages

    def test_is_a_word_pattern(self):
        text = "Amon is a Sindarin word meaning 'hill'."
        result = extract_lineages_from_text(text)
        assert result.lineages
        assert any(
            f.language == TolkienLanguage.SINDARIN
            for lin in result.lineages
            for f in lin.forms
        )

    def test_no_hits_on_plain_prose(self):
        text = "Bilbo walked down the road and found a nice spot for lunch."
        result = extract_lineages_from_text(text)
        assert result.lineages == []
        assert result.hit_count == 0

    def test_multiple_hits_grouped(self):
        text = textwrap.dedent("""\
            Imladris, called Rivendell in Common Speech, was a refuge.
            The Sindarin name for Rivendell is Imladris.
        """)
        result = extract_lineages_from_text(text)
        assert len(result.lineages) >= 1

    def test_extraction_mode_is_regex(self):
        text = "Imladris, called Rivendell in Common Speech."
        result = extract_lineages_from_text(text)
        assert result.extraction_mode == "regex"


# ---------------------------------------------------------------------------
# Round-trip: extraction -> canonical JSON -> parse back
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_extracted_lineages_serialize_and_parse(self):
        text = "Imladris, called Rivendell in Common Speech."
        result = extract_lineages_from_text(text)
        assert result.lineages

        as_json = lineages_to_json(result.lineages)
        assert "lineages" in as_json

        for raw in as_json["lineages"]:
            lin = parse_lineage(raw)
            assert lin.entity_id
            assert len(lin.forms) >= 1


# ---------------------------------------------------------------------------
# Alias hints for resolver
# ---------------------------------------------------------------------------

class TestAliasHints:
    def test_alias_hints_from_lineages(self):
        text = "Imladris, called Rivendell in Common Speech."
        result = extract_lineages_from_text(text)
        hints = lineage_alias_hints(result.lineages)
        assert hints
        for eid, names in hints.items():
            assert isinstance(names, list)
            assert all(isinstance(n, str) for n in names)


# ---------------------------------------------------------------------------
# Resolver integration
# ---------------------------------------------------------------------------

class TestResolverIntegration:
    def test_load_language_aliases(self):
        from book_graph_analyzer.extract.resolver import EntityResolver, EntityDatabase

        text = "Imladris, called Rivendell in Common Speech."
        result = extract_lineages_from_text(text)

        resolver = EntityResolver.__new__(EntityResolver)
        resolver.db = EntityDatabase()
        resolver.settings = None
        resolver.seed_dir = Path("/nonexistent")

        count = resolver.load_language_aliases(lineages=result.lineages)
        assert count >= 2

        eid, etype, conf = resolver.db.lookup("Imladris")
        assert eid is not None
        eid2, _, _ = resolver.db.lookup("Rivendell")
        assert eid2 == eid


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestCLI:
    def test_languages_text_mode(self):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, [
            "worldbible", "languages",
            "-t", "Imladris, called Rivendell in Common Speech",
        ])
        assert result.exit_code == 0, result.output
        assert "Imladris" in result.output
        assert "Extraction mode: regex" in result.output

    def test_languages_file_mode(self, tmp_path):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        f = tmp_path / "sample.txt"
        f.write_text("Mithrandir means 'Grey Pilgrim' in Sindarin.", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, [
            "worldbible", "languages",
            "-f", str(f),
        ])
        assert result.exit_code == 0, result.output
        assert "Mithrandir" in result.output

    def test_languages_json_mode(self, tmp_path):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        data = {
            "lineages": [{
                "entity_id": "place_rivendell",
                "forms": [
                    {"id": "lf_1", "form": "Imladris", "language": "Sindarin"},
                    {"id": "lf_2", "form": "Rivendell", "language": "Common Speech"},
                ],
                "derivations": [],
            }]
        }
        f = tmp_path / "lineages.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["worldbible", "languages", str(f)])
        assert result.exit_code == 0, result.output
        assert "Imladris" in result.output

    def test_languages_no_input_error(self):
        from click.testing import CliRunner
        from book_graph_analyzer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["worldbible", "languages"])
        assert result.exit_code != 0
