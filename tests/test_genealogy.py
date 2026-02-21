import json
from unittest.mock import MagicMock

from click.testing import CliRunner

from book_graph_analyzer.models.worldbuilding import GenealogyRelationType
from book_graph_analyzer.worldbible.genealogy import (
    build_ancestor_chain,
    extract_genealogy_from_text,
    genealogy_to_json,
    infer_generation_depths,
    load_genealogy_from_file,
    normalize_relation_type,
)


def test_normalize_relation_type_aliases():
    assert normalize_relation_type("father") == GenealogyRelationType.PARENT_OF
    assert normalize_relation_type("daughter") == GenealogyRelationType.CHILD_OF
    assert normalize_relation_type("sibling") == GenealogyRelationType.SIBLING_OF


def test_extract_genealogy_rules_adds_inverse_relations():
    text = "Aragorn son of Arathorn. Elrond father of Arwen."
    relations = extract_genealogy_from_text(text, passage_id="p1", house="House of Isildur")

    rel_types = {(r.source_name, r.target_name, r.relation_type.value) for r in relations}
    assert ("Aragorn", "Arathorn", "CHILD_OF") in rel_types
    assert ("Arathorn", "Aragorn", "PARENT_OF") in rel_types
    assert ("Elrond", "Arwen", "PARENT_OF") in rel_types
    assert ("Arwen", "Elrond", "CHILD_OF") in rel_types


def test_extract_genealogy_infers_house_from_context():
    text = "Aragorn son of Arathorn of the House of Isildur."
    relations = extract_genealogy_from_text(text, passage_id="p1")
    assert relations
    assert any(r.house == "House of Isildur" for r in relations)


def test_extract_genealogy_infers_generation_depth_for_direct_edges():
    relations = extract_genealogy_from_text("Thingol father of Luthien.")
    assert relations
    assert all(r.generation_depth == 1 for r in relations)


def test_infer_generation_depths_for_ancestor_relation_via_traversal():
    from book_graph_analyzer.models.worldbuilding import GenealogyRelation

    rels = [
        GenealogyRelation(
            source_id="char_a",
            source_name="A",
            target_id="char_b",
            target_name="B",
            relation_type=GenealogyRelationType.PARENT_OF,
        ),
        GenealogyRelation(
            source_id="char_b",
            source_name="B",
            target_id="char_c",
            target_name="C",
            relation_type=GenealogyRelationType.PARENT_OF,
        ),
        GenealogyRelation(
            source_id="char_a",
            source_name="A",
            target_id="char_c",
            target_name="C",
            relation_type=GenealogyRelationType.ANCESTOR_OF,
        ),
    ]

    out = infer_generation_depths(rels)
    anc = next(r for r in out if r.relation_type == GenealogyRelationType.ANCESTOR_OF)
    assert anc.generation_depth == 2


def test_build_ancestor_chain_traverses_multiple_generations():
    relations = extract_genealogy_from_text("Elros son of Earendil. Aragorn son of Arathorn. Arathorn son of Arador.")
    chain = build_ancestor_chain(relations, character_id="char_aragorn", depth=3)
    names = {(r.source_name, r.target_name) for r in chain}
    assert ("Aragorn", "Arathorn") in names
    assert ("Arathorn", "Arador") in names


def test_genealogy_json_roundtrip(tmp_path):
    relations = extract_genealogy_from_text("Thingol father of Luthien.")
    payload = genealogy_to_json(relations)

    p = tmp_path / "genealogy.json"
    p.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_genealogy_from_file(p)
    assert len(loaded) == len(relations)
    assert loaded[0].relation_type in (GenealogyRelationType.PARENT_OF, GenealogyRelationType.CHILD_OF)


def test_graphwriter_write_genealogy_batch_runs_queries():
    from book_graph_analyzer.graph.writer import GraphWriter

    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    writer = GraphWriter(driver=mock_driver)
    relations = extract_genealogy_from_text("Finarfin father of Finrod.")
    count = writer.write_genealogy_batch(relations, book="Silmarillion")

    assert count == len(relations)
    assert mock_session.run.call_count == len(relations)


def test_lore_genealogy_extract_cli(tmp_path):
    from book_graph_analyzer.cli import main

    f = tmp_path / "sample.txt"
    f.write_text("Aragorn son of Arathorn.", encoding="utf-8")

    result = CliRunner().invoke(main, ["lore", "genealogy", "--extract", str(f)])
    assert result.exit_code == 0
    assert "Found" in result.output
