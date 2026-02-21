from __future__ import annotations

from book_graph_analyzer.models.passage import Passage
from book_graph_analyzer.models.worldbuilding import EditorialLayer, EditorialStatus, AuthorPeriod, SourceStratum
from book_graph_analyzer.worldbible.editorial import detect_editorial_divergences


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
