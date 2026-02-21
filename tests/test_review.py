"""Tests for Issue #13: Human Review Interface (CLI + SQLite queue)."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from book_graph_analyzer.cli import main
from book_graph_analyzer.review import ReviewStore, seed_entities, seed_conflicts, seed_rules, seed_relationships


def test_review_list_shows_counts(tmp_path: Path):
    db = tmp_path / "review.db"
    store = ReviewStore(db)
    store.add_item("entity", 0.72, {"canonical_name": "Gandalf"}, item_id="entity1")
    store.add_item("conflict", 0.68, {"summary": "Blue Wizards"}, item_id="conflict1")

    runner = CliRunner()
    res = runner.invoke(main, ["review", "list", "--db", str(db)])
    assert res.exit_code == 0
    assert "entity" in res.output
    assert "conflict" in res.output
    assert "total" in res.output


def test_review_entities_accept_flow(tmp_path: Path):
    db = tmp_path / "review.db"
    store = ReviewStore(db)
    store.add_item(
        "entity",
        0.72,
        {
            "variants": ["Mithrandir", "Gandalf"],
            "canonical_name": "Gandalf",
            "entity_type": "character",
            "contexts": ["...Mithrandir! cried Legolas..."],
        },
        item_id="entity_gandalf",
    )

    runner = CliRunner()
    # choose [A]ccept
    res = runner.invoke(main, ["review", "entities", "--db", str(db)], input="a\n")
    assert res.exit_code == 0
    item = store.get_item("entity_gandalf")
    assert item is not None
    assert item.status == "accepted"
    assert item.confidence >= 0.95


def test_review_entities_edit_flow(tmp_path: Path):
    db = tmp_path / "review.db"
    store = ReviewStore(db)
    store.add_item(
        "entity",
        0.70,
        {
            "variants": ["the ranger", "Strider"],
            "canonical_name": "Strider",
            "entity_type": "unknown",
        },
        item_id="entity_strider",
    )

    runner = CliRunner()
    # edit -> new canonical -> type -> notes
    res = runner.invoke(
        main,
        ["review", "entities", "--db", str(db)],
        input="e\nAragorn\ncharacter\nfixed\n",
    )
    assert res.exit_code == 0
    item = store.get_item("entity_strider")
    assert item is not None
    assert item.status == "edited"
    assert item.payload["canonical_name"] == "Aragorn"
    assert item.payload["entity_type"] == "character"


def test_review_conflicts_defer(tmp_path: Path):
    db = tmp_path / "review.db"
    store = ReviewStore(db)
    store.add_item(
        "conflict",
        0.69,
        {
            "summary": "Blue Wizards names differ",
            "resolution_policy": "flag_for_human",
            "claims": [{"statement": "Alatar/Pallando"}, {"statement": "Morinehtar/Romestamo"}],
        },
        item_id="conflict_blue",
    )

    runner = CliRunner()
    res = runner.invoke(main, ["review", "conflicts", "--db", str(db)], input="d\n")
    assert res.exit_code == 0
    item = store.get_item("conflict_blue")
    assert item is not None
    assert item.status == "deferred"


def test_review_rules_reject(tmp_path: Path):
    db = tmp_path / "review.db"
    store = ReviewStore(db)
    store.add_item(
        "rule",
        0.62,
        {"statement": "Fake rule", "hardness": "HARD", "cypher_check": "MATCH ..."},
        item_id="rule_fake",
    )

    runner = CliRunner()
    res = runner.invoke(main, ["review", "rules", "--db", str(db)], input="r\n")
    assert res.exit_code == 0
    item = store.get_item("rule_fake")
    assert item is not None
    assert item.status == "rejected"
    assert item.confidence <= 0.10


def test_seed_helpers_thresholds(tmp_path: Path):
    db = tmp_path / "review.db"
    store = ReviewStore(db)

    entities = [
        {"id": "e1", "canonical_name": "A", "confidence": 0.90, "needs_review": False},
        {"id": "e2", "canonical_name": "B", "confidence": 0.72, "needs_review": False},
    ]
    conflicts = [
        {"id": "c1", "resolution_policy": "flag_for_human", "summary": "x"},
        {"id": "c2", "resolution_policy": "use_later_text", "summary": "y"},
    ]
    rules = [
        {"id": "r1", "confidence": 0.82, "statement": "s"},
        {"id": "r2", "confidence": 0.92, "statement": "t"},
    ]
    rels = [
        {"id": "rel1", "confidence": 0.70},
        {"id": "rel2", "confidence": 0.90},
    ]

    assert seed_entities(store, entities) == 1
    assert seed_conflicts(store, conflicts) == 1
    assert seed_rules(store, rules) == 1
    assert seed_relationships(store, rels) == 1

    counts = store.pending_counts()
    assert counts["entity"] == 1
    assert counts["conflict"] == 1
    assert counts["rule"] == 1
    assert counts["relationship"] == 1


def test_review_decision_audit_recorded(tmp_path: Path):
    db = tmp_path / "review.db"
    store = ReviewStore(db)
    store.add_item("entity", 0.7, {"canonical_name": "X"}, item_id="entity_x")
    ok = store.decide("entity_x", "accepted", notes="looks good", log_to_neo4j=False)
    assert ok is True

    decisions = store.recent_decisions(limit=5)
    assert len(decisions) >= 1
    assert decisions[0]["item_id"] == "entity_x"
    assert decisions[0]["decision"] == "accepted"


def test_review_seed_demo_command(tmp_path: Path):
    db = tmp_path / "review.db"
    runner = CliRunner()
    res = runner.invoke(main, ["review", "seed-demo", "--db", str(db)])
    assert res.exit_code == 0

    store = ReviewStore(db)
    counts = store.pending_counts()
    assert counts["total"] >= 3
