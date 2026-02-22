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


def test_extract_genealogy_appositive_hobbit_style_clause():
    relations = extract_genealogy_from_text("Bilbo, son of Bungo Baggins, lived in the Shire.")
    rel_types = {(r.source_name, r.target_name, r.relation_type.value) for r in relations}
    assert ("Bilbo", "Bungo Baggins", "CHILD_OF") in rel_types
    assert ("Bungo Baggins", "Bilbo", "PARENT_OF") in rel_types


def test_extract_genealogy_was_the_son_of_pattern():
    relations = extract_genealogy_from_text("Faramir was the son of Denethor.")
    rel_types = {(r.source_name, r.target_name, r.relation_type.value) for r in relations}
    assert ("Faramir", "Denethor", "CHILD_OF") in rel_types


def test_extract_genealogy_grandson_pattern():
    relations = extract_genealogy_from_text("Eldarion grandson of Arathorn.")
    rel_types = {(r.source_name, r.target_name, r.relation_type.value) for r in relations}
    assert ("Eldarion", "Arathorn", "DESCENDANT_OF") in rel_types


def test_extract_genealogy_safe_llm_fallback_triggers_when_low_yield():
    class _LLM:
        def generate(self, _prompt: str) -> str:
            return json.dumps([
                {"source_name": "Theoden", "target_name": "Thengel", "relation_type": "CHILD_OF"}
            ])

    relations = extract_genealogy_from_text(
        "It is said Theoden heir of Thengel.",
        llm_client=_LLM(),
        min_relations_for_fallback=3,
    )
    rel_types = {(r.source_name, r.target_name, r.relation_type.value) for r in relations}
    assert any(
        {a, b} == {"Theoden", "Thengel"} and rel == "CHILD_OF"
        for a, b, rel in rel_types
    )


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
    assert payload["metrics"]["relation_count"] == len(relations)

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


def test_lore_genealogy_query_cli_shows_inheritance_traits(monkeypatch):
    from book_graph_analyzer import cli as cli_module

    class _MockWriter:
        def query_genealogy(self, **kwargs):
            return [
                {
                    "source": "Aragorn",
                    "rel": "DESCENDANT_OF",
                    "target": "Elendil",
                    "house": "House of Elendil",
                    "generation_depth": 39,
                    "inheritance_traits": ["kingship", "foresight"],
                }
            ]

        def close(self):
            return None

    monkeypatch.setattr("book_graph_analyzer.graph.connection.check_neo4j_connection", lambda: True)
    monkeypatch.setattr("book_graph_analyzer.graph.writer.GraphWriter", _MockWriter)

    result = CliRunner().invoke(cli_module.main, ["lore", "genealogy", "--character", "Aragorn"])
    assert result.exit_code == 0
    assert "kingship" in result.output
