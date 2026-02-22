from __future__ import annotations

from book_graph_analyzer.models.passage import Passage
from book_graph_analyzer.models.worldbuilding import EditorialLayer, EditorialStatus, AuthorPeriod, SourceStratum
from click.testing import CliRunner
import json
from book_graph_analyzer.worldbible.editorial import (
    detect_editorial_divergences,
    validate_editorial_provenance,
)


def _p(**overrides) -> Passage:
    base = dict(
        id="p1",
        text="Test passage",
        book="The Silmarillion",
        chapter="1",
        chapter_num=1,
        paragraph_num=1,
        sentence_num=1,
        char_offset=0,
    )
    base.update(overrides)
    return Passage(**base)


def test_source_stratum_defaults_on_editorial_layer():
    layer = EditorialLayer(
        source_id="src_test",
        source_title="Test",
        editorial_status=EditorialStatus.PUBLISHED,
        author_period=AuthorPeriod.MIDDLE,
    )
    assert layer.default_stratum == SourceStratum.CORE_TEXT


def test_passage_supports_provenance_fields():
    p = _p(
        source_id="src_silmarillion_1977",
        source_title="The Silmarillion",
        source_stratum="appendix",
        source_authority_weight=0.85,
        provenance_tags=["appendix", "editorial"],
        factual_claims={"balrog_wings": "literal"},
    )
    assert p.source_stratum == "appendix"
    assert p.factual_claims["balrog_wings"] == "literal"


def test_detect_editorial_divergences_factual_and_style():
    p1 = _p(
        id="p1",
        source_id="src_x",
        source_stratum="core_text",
        avg_sentence_length=8.0,
        passive_ratio=0.05,
        factual_claims={"balrog_wings": "literal"},
    )
    p2 = _p(
        id="p2",
        source_id="src_x",
        source_stratum="annotation",
        avg_sentence_length=24.5,
        passive_ratio=0.35,
        factual_claims={"balrog_wings": "metaphorical"},
    )

    out = detect_editorial_divergences([p1, p2])
    kinds = {d.kind for d in out}
    assert "factual" in kinds
    assert "style" in kinds


def test_validate_editorial_provenance_fails_when_required_fields_missing():
    p = _p(
        id="p1",
        factual_claims={"balrog_wings": "literal"},
        source_id=None,
        source_stratum=None,
        source_authority_weight=None,
        provenance_tags=[],
    )
    result = validate_editorial_provenance([p], max_missing_ratio=0.0)
    assert result.checked_count == 1
    assert result.missing_count == 1
    assert result.is_valid is False


def test_validate_editorial_provenance_respects_authority_threshold():
    p = _p(
        id="p1",
        factual_claims={"balrog_wings": "literal"},
        source_id="src_x",
        source_stratum="annotation",
        source_authority_weight=0.4,
        provenance_tags=["annotation"],
    )
    result = validate_editorial_provenance([p], min_authority_weight=0.5)
    assert result.invalid_authority_count == 1
    assert result.is_valid is False


def test_validate_editorial_provenance_detects_inconsistent_hobbit_source_fields():
    p = _p(
        id="p-hobbit",
        book="The Hobbit",
        factual_claims={"smaug_location": "erebor"},
        source_id="src_unfinished_tales",
        source_title="Unfinished Tales",
        source_stratum="annotation",
        source_authority_weight=1.0,
        provenance_tags=["annotation"],
    )
    result = validate_editorial_provenance([p], max_missing_ratio=0.0)
    assert result.inconsistent_count == 1
    assert result.is_valid is False


def test_corpus_sources_report_divergence_gated_on_missing_provenance():
    from book_graph_analyzer.cli import main

    runner = CliRunner()
    with runner.isolated_filesystem():
        corpus_dir = "data/corpora/hobbit"
        import os
        os.makedirs(corpus_dir, exist_ok=True)
        with open(f"{corpus_dir}/passages.json", "w", encoding="utf-8") as f:
            json.dump([
                {
                    "id": "p1",
                    "text": "test",
                    "book": "The Hobbit",
                    "chapter": "1",
                    "chapter_num": 1,
                    "paragraph_num": 1,
                    "sentence_num": 1,
                    "char_offset": 0,
                    "factual_claims": {"x": "y"},
                }
            ], f)

        result = runner.invoke(main, [
            "corpus", "sources", "hobbit", "--report-divergence", "--max-missing-provenance-ratio", "0",
        ])
        assert result.exit_code != 0
        assert "gated" in result.output.lower()


def test_corpus_sources_report_divergence_gated_on_inconsistent_hobbit_provenance():
    from book_graph_analyzer.cli import main

    runner = CliRunner()
    with runner.isolated_filesystem():
        corpus_dir = "data/corpora/hobbit"
        import os
        os.makedirs(corpus_dir, exist_ok=True)
        with open(f"{corpus_dir}/passages.json", "w", encoding="utf-8") as f:
            json.dump([
                {
                    "id": "p1",
                    "text": "test",
                    "book": "The Hobbit",
                    "chapter": "1",
                    "chapter_num": 1,
                    "paragraph_num": 1,
                    "sentence_num": 1,
                    "char_offset": 0,
                    "factual_claims": {"x": "y"},
                    "source_id": "src_unfinished_tales",
                    "source_title": "Unfinished Tales",
                    "source_stratum": "annotation",
                    "source_authority_weight": 1.0,
                    "provenance_tags": ["annotation"],
                }
            ], f)

        result = runner.invoke(main, [
            "corpus", "sources", "hobbit", "--report-divergence", "--max-missing-provenance-ratio", "0",
        ])
        assert result.exit_code != 0
        assert "inconsistent" in result.output.lower() or "gated" in result.output.lower()
