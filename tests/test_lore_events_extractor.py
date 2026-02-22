import logging

from book_graph_analyzer.lore import events as events_module
from book_graph_analyzer.lore.events import EventExtractor


class _DummyLLM:
    def generate(self, prompt, temperature=0.2, max_tokens=2000):
        return "ok"

    def extract_json(self, response):
        return self.payload


class _EmptyLLM:
    provider = "openai"
    model = "gpt-4o-mini"

    def generate(self, prompt, temperature=0.2, max_tokens=2000):
        return ""

    def extract_json(self, response):
        return None


def test_extract_llm_normalizes_list_relation_ids_before_lookup(monkeypatch):
    dummy = _DummyLLM()
    dummy.payload = {
        "events": [
            {"id": "e1", "description": "Event one"},
            {"id": ["e2"], "description": "Event two"},
        ],
        "relations": [
            {"event1": "e1", "relation": "before", "event2": ["e2"]}
        ],
    }

    monkeypatch.setattr(events_module, "LLMClient", lambda: dummy)

    extractor = EventExtractor()
    extracted_events, relations = extractor._extract_llm("text", "book", chunk_index=2)

    assert len(extracted_events) == 2
    assert len(relations) == 1
    assert relations[0].event1_id == "c2_e1"
    # List id is normalized deterministically and mapped via id_map
    assert relations[0].event2_id == 'c2_["e2"]'


def test_extract_llm_drops_malformed_rows_with_warning(monkeypatch, caplog):
    dummy = _DummyLLM()
    dummy.payload = {
        "events": [
            {"id": "ok", "description": "valid"},
            {"id": "bad-no-description"},
            "not-a-dict",
        ],
        "relations": [
            {"event1": "ok", "relation": "before", "event2": "ok"},  # valid
            {"event1": "ok", "relation": "before"},  # missing event2
            {"event1": None, "event2": "ok"},  # invalid id
            "not-a-dict",
        ],
    }

    monkeypatch.setattr(events_module, "LLMClient", lambda: dummy)

    extractor = EventExtractor()
    with caplog.at_level(logging.WARNING):
        extracted_events, relations = extractor._extract_llm("text", "book")

    assert len(extracted_events) == 1
    assert len(relations) == 1
    assert "Dropped malformed LLM payload rows" in caplog.text
    assert "events=2 relations=3" in caplog.text


def test_extract_llm_warns_and_skips_on_empty_response(monkeypatch, caplog):
    monkeypatch.setattr(events_module, "LLMClient", lambda: _EmptyLLM())

    extractor = EventExtractor()
    with caplog.at_level(logging.WARNING):
        extracted_events, relations = extractor._extract_llm("text", "book", chunk_index=4)

    assert extracted_events == []
    assert relations == []
    assert "empty response" in caplog.text
