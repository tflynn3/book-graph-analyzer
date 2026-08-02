import json

from book_graph_analyzer.graph.writer import GraphWriter
from book_graph_analyzer.lore import events as events_module
from book_graph_analyzer.lore.events import (
    Event,
    EventExtractor,
    EventGraph,
    EventRelation,
)
from book_graph_analyzer.lore.temporal import Era


def test_event_round_trip_preserves_source_provenance():
    event = Event(
        id="gollum-loses-ring",
        description="Gollum lost the Ring",
        agent="Gollum",
        action="lost",
        patient="the Ring",
        location="the Misty Mountains",
        polarity="negative",
        modality="believed",
        epistemic_status="character_belief",
        knowledge_holder="Gollum",
        certainty="uncertain",
        era=Era.THIRD_AGE,
        year=2941,
        year_text="Third Age 2941",
        source_text="It slipped from his finger.",
        source_book="The Hobbit",
        source_location="Chapter 5, Riddles in the Dark",
        source_span_start=120,
        source_span_end=147,
        confidence=0.9,
    )

    payload = event.to_dict()
    restored = Event.from_dict(payload)

    assert payload["source_book"] == "The Hobbit"
    assert payload["source_location"] == "Chapter 5, Riddles in the Dark"
    assert restored == event


class _PayloadLLM:
    provider = "test"
    model = "test-events"

    def __init__(self, payload):
        self.payload = payload

    def generate(self, *_args, **_kwargs):
        return json.dumps(self.payload)

    @staticmethod
    def extract_json(response):
        return json.loads(response)


def _extract_payload(monkeypatch, text, payload, *, chunk_index=0):
    monkeypatch.setattr(
        events_module,
        "LLMClient",
        lambda *_args, **_kwargs: _PayloadLLM(payload),
    )
    return EventExtractor(use_llm=True)._extract_llm_once(
        text,
        "The Fellowship of the Ring",
        chunk_index=chunk_index,
    )


def test_exact_source_evidence_preserves_source_span_and_confidence(monkeypatch):
    text = (
        "At the Council, Gandalf believed\n  the Ring was dangerous. "
        "Therefore Frodo agreed to depart."
    )
    payload = {
        "events": [
            {
                "id": "belief",
                "description": "Gandalf believed the Ring was dangerous",
                "agent": "Gandalf",
                "action": "believed",
                "patient": "the Ring",
                "location": "Rivendell",
                "era": "third_age",
                "polarity": "positive",
                "modality": "believed",
                "epistemic_status": "character_belief",
                "knowledge_holder": "Gandalf",
                "certainty": "uncertain",
                "source_text": "Gandalf believed the Ring was dangerous.",
                "confidence": 0.93,
            },
            {
                "id": "departure",
                "description": "Frodo agreed to depart",
                "agent": "Frodo",
                "action": "agreed",
                "source_text": "Frodo agreed to depart.",
                "confidence": 0.88,
            },
        ],
        "relations": [
            {
                "event1": "belief",
                "relation": "causes",
                "event2": "departure",
                "source_text": "Therefore Frodo agreed to depart.",
                "confidence": 0.88,
            }
        ],
    }

    extracted, relations, reason, _raw = _extract_payload(monkeypatch, text, payload)

    assert reason == ""
    belief = extracted[0]
    assert belief.era == Era.THIRD_AGE
    assert belief.source_text == "Gandalf believed\n  the Ring was dangerous."
    assert text[belief.source_span_start : belief.source_span_end] == belief.source_text
    assert belief.source_location == (
        f"chunk 0, chars {belief.source_span_start}-{belief.source_span_end}"
    )
    assert belief.confidence == 0.93
    assert belief.modality == "believed"
    assert belief.epistemic_status == "character_belief"
    assert belief.knowledge_holder == "Gandalf"
    assert relations[0].source_text == "Therefore Frodo agreed to depart."
    assert relations[0].confidence == 0.88


def test_unsupported_source_evidence_is_cleared_and_confidence_is_capped(monkeypatch):
    text = "Frodo waited in Rivendell. Sam remained beside him."
    unsupported = "Aragorn crowned Frodo king of Gondor."
    payload = {
        "events": [
            {
                "id": "waited",
                "description": "Frodo waited",
                "source_text": unsupported,
                "confidence": 0.99,
            },
            {
                "id": "remained",
                "description": "Sam remained",
                "source_text": unsupported,
                "confidence": 0.91,
            },
        ],
        "relations": [
            {
                "event1": "waited",
                "relation": "before",
                "event2": "remained",
                "source_text": unsupported,
                "confidence": 0.98,
            }
        ],
    }

    extracted, relations, reason, _raw = _extract_payload(
        monkeypatch,
        text,
        payload,
        chunk_index=2,
    )

    assert reason == ""
    assert all(event.source_text == "" for event in extracted)
    assert all(event.source_span_start is None for event in extracted)
    assert all(event.source_span_end is None for event in extracted)
    assert all(event.source_location == "chunk 2" for event in extracted)
    assert all(event.confidence == 0.55 for event in extracted)
    assert relations[0].source_text == ""
    assert relations[0].confidence == 0.55


def test_event_graph_round_trip_preserves_relation_source_text():
    graph = EventGraph()
    graph.add_event(Event(id="a", description="A happened"))
    graph.add_event(Event(id="b", description="B happened"))
    graph.add_relation(
        EventRelation(
            event1_id="a",
            relation="before",
            event2_id="b",
            source_text="After A, B followed.",
        )
    )

    restored = EventGraph.from_dict(graph.to_dict())

    assert restored.relations[0].source_text == "After A, B followed."


def test_verb_normalization_does_not_strip_trailing_characters():
    graph = EventGraph()
    graph.add_event(Event(id="pass", description="They passed", action="passed"))
    graph.add_event(Event(id="make", description="They made it", action="made"))

    assert [event.id for event in graph.find_events(action="pass")] == ["pass"]
    assert [event.id for event in graph.find_events(action="make")] == ["make"]

    extractor = EventExtractor(use_llm=False)
    assert extractor._normalize_event_key(graph.events["pass"]) == "pass"
    assert extractor._normalize_event_key(graph.events["make"]) == "make"


class _CaptureResult:
    def __init__(self, rows=None, single_row=None):
        self.rows = rows or []
        self.single_row = single_row

    def __iter__(self):
        return iter(self.rows)

    def single(self):
        return self.single_row


class _CaptureSession:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def run(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if "RETURN count(r) AS rel_count" in query:
            return _CaptureResult(single_row={"rel_count": len(kwargs["batch"])})
        return _CaptureResult()


class _CaptureDriver:
    def __init__(self):
        self.calls = []

    def session(self):
        return _CaptureSession(self.calls)


def test_graph_writer_persists_and_queries_typed_event_evidence():
    driver = _CaptureDriver()
    writer = GraphWriter(driver=driver)
    event = Event(
        id="event-1",
        description="Gandalf believed the Ring was dangerous",
        agent="Gandalf",
        action="believed",
        patient="the Ring",
        location="Rivendell",
        polarity="positive",
        modality="believed",
        epistemic_status="character_belief",
        knowledge_holder="Gandalf",
        certainty="uncertain",
        era=Era.THIRD_AGE,
        year=3018,
        year_text="Third Age 3018",
        source_text="Gandalf believed the Ring was dangerous.",
        source_book="The Fellowship of the Ring",
        source_location="Book II, Chapter 2",
        source_span_start=15,
        source_span_end=57,
        confidence=0.93,
    )

    writer.write_event(event, book="fallback")
    single_query, single_params = driver.calls[-1]
    assert "e.year_text = $year_text" in single_query
    assert single_params["year_text"] == "Third Age 3018"

    writer.write_events_batch([event], book="fallback")
    event_query, event_params = driver.calls[-1]
    stored_event = event_params["batch"][0]
    assert "e.year_text = item.year_text" in event_query
    assert stored_event == {
        "id": "event-1",
        "description": "Gandalf believed the Ring was dangerous",
        "agent": "Gandalf",
        "action": "believed",
        "patient": "the Ring",
        "location": "Rivendell",
        "polarity": "positive",
        "modality": "believed",
        "epistemic_status": "character_belief",
        "knowledge_holder": "Gandalf",
        "certainty": "uncertain",
        "source_text": "Gandalf believed the Ring was dangerous.",
        "source_location": "Book II, Chapter 2",
        "source_span_start": 15,
        "source_span_end": 57,
        "era": "third_age",
        "year": 3018,
        "year_text": "Third Age 3018",
        "confidence": 0.93,
        "source_book": "The Fellowship of the Ring",
    }

    relation = EventRelation(
        event1_id="event-1",
        relation="causes",
        event2_id="event-2",
        source_text="Therefore Frodo agreed to depart.",
        confidence=0.88,
    )
    count = writer.write_event_relations_batch(
        [relation],
        event_book_by_id={
            "event-1": "The Fellowship of the Ring",
            "event-2": "The Fellowship of the Ring",
        },
    )
    relation_query, relation_params = driver.calls[-1]
    assert count == 1
    assert "r.source_text = item.source_text" in relation_query
    assert relation_params["batch"][0]["source_text"] == relation.source_text

    assert writer.query_events() == []
    query, _params = driver.calls[-1]
    for projection in (
        "e.location as location",
        "e.polarity as polarity",
        "e.modality as modality",
        "e.epistemic_status as epistemic_status",
        "e.knowledge_holder as knowledge_holder",
        "e.certainty as certainty",
        "e.year_text as year_text",
        "e.source_text as source_text",
        "e.source_location as source_location",
        "e.source_span_start as source_span_start",
        "e.source_span_end as source_span_end",
    ):
        assert projection in query
