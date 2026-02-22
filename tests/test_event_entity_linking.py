from dataclasses import dataclass

from book_graph_analyzer.graph.writer import GraphWriter
from book_graph_analyzer.lore.events import Event, EventGraph


class _FakeResult:
    def __init__(self, rows=None, single_row=None):
        self._rows = rows or []
        self._single = single_row

    def __iter__(self):
        return iter(self._rows)

    def single(self):
        return self._single


@dataclass
class _Node:
    id: str
    labels: set[str]
    canonical_id: str
    canonical_name: str
    aliases: list[str]


class _FakeSession:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query, **kwargs):
        if "MERGE (e:Event {id: item.id})" in query:
            for item in kwargs["batch"]:
                self.state["events"].add(item["id"])
            return _FakeResult()

        if "MATCH (e1:Event {id: item.event1_id})" in query:
            return _FakeResult()

        if "MERGE (n)-[r:" in query and "RETURN count(r) AS cnt" in query:
            event_id = kwargs["event_id"]
            labels = set(kwargs["labels"])
            candidates = [c.lower() for c in kwargs["candidates"]]
            candidate_ids = [c.lower() for c in kwargs.get("candidate_ids", [])]
            role = kwargs["role"]
            rel_type = query.split("MERGE (n)-[r:")[1].split("]->(e)")[0]

            best = None
            best_score = -1
            for node in self.state["nodes"]:
                if not (node.labels & labels):
                    continue
                cid = node.canonical_id.lower()
                cname = node.canonical_name.lower()
                aliases = [a.lower() for a in node.aliases]
                for cand in candidates:
                    if not cand:
                        continue
                    matched = (
                        cid in candidate_ids
                        or
                        cname == cand
                        or cand in aliases
                        or cand in cname
                        or cname in cand
                        or any((cand in a or a in cand) for a in aliases)
                    )
                    if not matched:
                        continue
                    score = 110 if cid in candidate_ids else (100 if (cname == cand or cand in aliases) else 70)
                    if score > best_score:
                        best = node
                        best_score = score

            if best is None or event_id not in self.state["events"]:
                return _FakeResult(single_row={"cnt": 0})

            self.state["rels"].add((best.id, rel_type, event_id, role))
            return _FakeResult(single_row={"cnt": 1})

        return _FakeResult()


class _FakeDriver:
    def __init__(self):
        self.state = {
            "events": set(),
            "rels": set(),
            "nodes": [
                _Node(id="char_bilbo", labels={"Character"}, canonical_id="char_bilbo", canonical_name="Bilbo", aliases=["Bilbo Baggins"]),
                _Node(id="obj_ring", labels={"Object"}, canonical_id="obj_ring", canonical_name="The One Ring", aliases=["the ring", "ring"]),
                _Node(id="place_riv", labels={"Place"}, canonical_id="place_riv", canonical_name="Rivendell", aliases=[]),
            ],
        }

    def session(self):
        return _FakeSession(self.state)


def test_event_entity_linking_regression_produces_links(monkeypatch):
    writer = GraphWriter(driver=_FakeDriver())
    monkeypatch.setattr(writer, "initialize", lambda: None)

    g = EventGraph()
    g.add_event(
        Event(
            id="hobbit:e1",
            description="Bilbo found the Ring in the dark",
            agent="Bilbo Baggins",
            action="found",
            patient="the ring",
        )
    )

    stats = writer.write_event_graph(g, book="The Hobbit", link_entities=True)

    assert stats["events_written"] == 1
    assert stats["entity_links"] > 0


def test_event_entity_linking_idempotent_on_rerun(monkeypatch):
    driver = _FakeDriver()
    writer = GraphWriter(driver=driver)
    monkeypatch.setattr(writer, "initialize", lambda: None)

    g = EventGraph()
    g.add_event(
        Event(
            id="hobbit:e2",
            description="Bilbo came to Rivendell",
            agent="Bilbo Baggins",
            action="arrived",
            patient="Rivendell",
        )
    )

    writer.write_event_graph(g, book="The Hobbit", link_entities=True)
    first_rel_count = len(driver.state["rels"])

    writer.write_event_graph(g, book="The Hobbit", link_entities=True)
    second_rel_count = len(driver.state["rels"])

    assert first_rel_count > 0
    assert second_rel_count == first_rel_count


def test_event_entity_linking_supports_canonical_id_mentions(monkeypatch):
    writer = GraphWriter(driver=_FakeDriver())
    monkeypatch.setattr(writer, "initialize", lambda: None)

    g = EventGraph()
    g.add_event(
        Event(
            id="hobbit:e3",
            description="Bilbo keeps possession",
            agent="char_bilbo",
            action="kept",
            patient="obj_ring",
        )
    )

    stats = writer.write_event_graph(g, book="The Hobbit", link_entities=True)
    assert stats["entity_links"] >= 2
