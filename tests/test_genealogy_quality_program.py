import json
from pathlib import Path

from book_graph_analyzer.worldbible.genealogy import extract_genealogy_from_text


def _pairs(text: str):
    return {(r.source_name, r.target_name, r.relation_type.value) for r in extract_genealogy_from_text(text)}


def test_tt_rotk_pattern_coverage_is_heir_to_and_descended_from():
    p = _pairs("Aragorn is heir to Isildur. Aragorn descended from Elendil.")
    assert ("Aragorn", "Isildur", "DESCENDANT_OF") in p
    assert ("Aragorn", "Elendil", "DESCENDANT_OF") in p


def test_dedupe_preserves_unique_relation_identity_per_span():
    rels = extract_genealogy_from_text("Aragorn son of Arathorn. Aragorn son of Arathorn.")
    pairs = [(r.source_name, r.target_name, r.relation_type.value) for r in rels]
    # each sentence should contribute one forward + one inverse edge
    assert pairs.count(("Aragorn", "Arathorn", "CHILD_OF")) == 2
    assert pairs.count(("Arathorn", "Aragorn", "PARENT_OF")) == 2


def test_gold_fixture_size_and_book_coverage():
    gold = Path("tests/fixtures/genealogy_gold.jsonl")
    rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert 50 <= len(rows) <= 100
    books = {r["book"] for r in rows}
    assert {"hobbit", "fellowship", "two_towers", "return_of_king", "silmarillion"}.issubset(books)
