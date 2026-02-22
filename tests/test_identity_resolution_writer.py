from __future__ import annotations

from book_graph_analyzer.graph.writer import GraphWriter
from book_graph_analyzer.lore.sociolinguistic_registers import RegisterProfile


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def single(self):
        return self._rows[0] if self._rows else None


class _Session:
    def __init__(self, scripted_rows, sink):
        self.scripted_rows = scripted_rows
        self.sink = sink

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, **params):
        self.sink.append((query, params))
        for needle, rows in self.scripted_rows:
            if needle in query:
                return _Result(rows)
        return _Result([])


class _Driver:
    def __init__(self, scripted_rows):
        self.scripted_rows = scripted_rows
        self.calls = []

    def session(self):
        return _Session(self.scripted_rows, self.calls)


def _profile() -> RegisterProfile:
    return RegisterProfile(
        dominant_register="ritual",
        confidence=0.8,
        register_scores={"ritual": 0.8},
        formality_score=0.7,
        archaism_rate=0.05,
        contraction_rate=0.01,
        avg_sentence_length=12.0,
        token_count=120,
    )


def test_entity_resolver_prefers_id_match():
    d = _Driver([
        ("RETURN id(e) AS node_id", [{"node_id": 12, "id": "char_aragorn", "score": 4}]),
    ])
    w = GraphWriter(driver=d)
    out = w._resolve_entity_identity("char_aragorn")
    assert out["resolved"] is True
    assert out["ambiguous"] is False
    assert out["matched_by"] == "id"


def test_entity_resolver_marks_ambiguity():
    d = _Driver([
        ("RETURN id(e) AS node_id", [
            {"node_id": 12, "id": "char_aragorn", "score": 2},
            {"node_id": 13, "id": "char_aragorn_ii", "score": 2},
        ]),
    ])
    w = GraphWriter(driver=d)
    out = w._resolve_entity_identity("Aragorn")
    assert out["resolved"] is False
    assert out["ambiguous"] is True


def test_register_profile_uses_resolved_node_identity():
    d = _Driver([
        ("RETURN id(e) AS node_id", [{"node_id": 12, "id": "char_aragorn", "score": 4}]),
        ("MATCH (p:Passage)", [{"node_id": 44, "id": "p1", "score": 2}]),
    ])
    w = GraphWriter(driver=d)
    w.write_register_profile("char_aragorn", _profile(), source_passage_id="p1")

    assert any("RETURN id(e) AS node_id" in q for q, _ in d.calls)
    write_calls = [c for c in d.calls if "MERGE (rp:RegisterProfile" in c[0]]
    assert write_calls, "expected write query call"


def test_register_profile_idempotent_merge():
    d = _Driver([
        ("RETURN id(e) AS node_id", [{"node_id": 12, "id": "char_aragorn", "score": 4}]),
        ("MATCH (p:Passage)", []),
    ])
    w = GraphWriter(driver=d)
    w.write_register_profile("char_aragorn", _profile(), source_passage_id="p1")
    w.write_register_profile("char_aragorn", _profile(), source_passage_id="p1")

    # Both writes should use MERGE profile by entity_id (no duplicate-profile CREATE query)
    calls = [q for q, _ in d.calls if "MERGE (rp:RegisterProfile {entity_id: $entity_id})" in q]
    assert len(calls) == 2
