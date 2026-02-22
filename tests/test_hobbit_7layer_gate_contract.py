from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gate_artifact_schema_has_required_hard_gate_fields():
    gate = REPO_ROOT / "gates" / "hobbit_7layer_acceptance_gate.json"
    payload = json.loads(gate.read_text(encoding="utf-8"))

    assert payload.get("gate") == "hobbit-7layer"
    assert "schema_version" in payload
    assert isinstance(payload.get("checks"), dict)
    assert isinstance(payload.get("layers"), dict)

    for key in ("acceptance_smoke", "schema", "data_structure", "graph_accuracy"):
        assert key in payload["checks"]

    for key in ("47", "49", "50", "51"):
        assert key in payload["layers"]


def test_hobbit_events_data_structure_is_graph_consistent():
    fixture = REPO_ROOT / "data" / "output" / "hobbit_events.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))

    events = payload.get("events", {})
    relations = payload.get("relations", [])

    assert isinstance(events, dict) and events, "Expected non-empty events map"
    assert isinstance(relations, list), "Expected relations list"

    event_ids = set(events.keys())
    dangling = [
        r for r in relations
        if r.get("event1_id") not in event_ids or r.get("event2_id") not in event_ids
    ]
    assert not dangling, "All relations must reference existing events"


def test_hobbit_events_graph_accuracy_minimum_temporal_signal():
    fixture = REPO_ROOT / "data" / "output" / "hobbit_events.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))

    relations = payload.get("relations", [])
    temporal_types = {
        "before",
        "after",
        "simultaneous",
        "during",
        "overlaps",
        "contains",
        "causes",
    }

    temporal_edges = [r for r in relations if str(r.get("relation", "")).lower() in temporal_types]
    assert temporal_edges, "Expected at least one temporal relation in Hobbit event graph"
