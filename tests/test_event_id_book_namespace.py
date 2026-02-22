from book_graph_analyzer.graph.writer import GraphWriter
from book_graph_analyzer.lore.events import Event, EventGraph, EventRelation


class _FakeResult:
    def __init__(self, single_row=None):
        self._single = single_row

    def single(self):
        return self._single


class _FakeSession:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def run(self, query, **kwargs):
        if "MERGE (e:Event {id: item.id, source_book: item.source_book})" in query:
            for item in kwargs["batch"]:
                self.state["events"].add((item["id"], item["source_book"]))
            return _FakeResult()

        if "MATCH (e1:Event {id: item.event1_id, source_book: item.event1_book})" in query:
            for item in kwargs["batch"]:
                a = (item["event1_id"], item["event1_book"])
                b = (item["event2_id"], item["event2_book"])
                if a in self.state["events"] and b in self.state["events"]:
                    self.state["rels"].add((a, b))
            return _FakeResult()

        return _FakeResult(single_row={"cnt": 0})


class _FakeDriver:
    def __init__(self):
        self.state = {"events": set(), "rels": set()}

    def session(self):
        return _FakeSession(self.state)


def _build_graph(book: str) -> EventGraph:
    g = EventGraph()
    g.add_event(Event(id="event1", description=f"Start in {book}", source_book=book))
    g.add_event(Event(id="event2", description=f"End in {book}", source_book=book))
    g.add_relation(EventRelation(event1_id="event1", event2_id="event2", relation="before", confidence=0.9))
    return g


def test_same_event_ids_from_different_books_do_not_collide(monkeypatch):
    driver = _FakeDriver()
    writer = GraphWriter(driver=driver)
    monkeypatch.setattr(writer, "initialize", lambda: None)

    writer.write_event_graph(_build_graph("The Hobbit"), book="The Hobbit", link_entities=False)
    writer.write_event_graph(_build_graph("Unfinished Tales"), book="Unfinished Tales", link_entities=False)

    assert ("event1", "The Hobbit") in driver.state["events"]
    assert ("event1", "Unfinished Tales") in driver.state["events"]
    assert len(driver.state["events"]) == 4


def test_relations_bind_to_book_scoped_event_nodes(monkeypatch):
    driver = _FakeDriver()
    writer = GraphWriter(driver=driver)
    monkeypatch.setattr(writer, "initialize", lambda: None)

    writer.write_event_graph(_build_graph("The Hobbit"), book="The Hobbit", link_entities=False)
    writer.write_event_graph(_build_graph("Unfinished Tales"), book="Unfinished Tales", link_entities=False)

    assert (("event1", "The Hobbit"), ("event2", "The Hobbit")) in driver.state["rels"]
    assert (("event1", "Unfinished Tales"), ("event2", "Unfinished Tales")) in driver.state["rels"]
    assert len(driver.state["rels"]) == 2
