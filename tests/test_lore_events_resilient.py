import json

from book_graph_analyzer.lore import events as events_module
from book_graph_analyzer.lore.events import EventExtractor


class _FakeLLM:
    def __init__(self, model=None):
        self.model = model or "primary"
        self.provider = "openai"
        self._calls = 0

    def generate(self, *_args, **_kwargs):
        self._calls += 1
        if self.model == "gpt-4o-mini":
            return "not-json"
        return '{"events":[{"id":"e1","description":"Bilbo found ring","agent":"Bilbo","action":"found","patient":"Ring"}],"relations":[]}'

    def extract_json(self, response):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None


def test_resilient_escalates_to_fallback_and_persists_ledger(tmp_path, monkeypatch):
    def _factory(provider=None, model=None):
        return _FakeLLM(model=model or "gpt-4o-mini")

    monkeypatch.setattr(events_module, "LLMClient", _factory)

    text = "A" * 7000
    checkpoint = tmp_path / "hobbit.checkpoint.json"
    extractor = EventExtractor(use_llm=True)

    graph = extractor.extract_from_book(
        text,
        source_book="Hobbit",
        chunk_size=3000,
        checkpoint_file=str(checkpoint),
        resilient=True,
        fallback_model="gpt-4o",
    )

    assert len(graph.events) >= 1
    ledger = json.loads((tmp_path / "hobbit.checkpoint.json.ledger.json").read_text(encoding="utf-8"))
    statuses = {row["status"] for row in ledger["chunks"]}
    assert "fallback_success" in statuses


def test_resilient_resume_skips_completed_chunks(tmp_path, monkeypatch):
    class _AlwaysGood(_FakeLLM):
        def generate(self, *_args, **_kwargs):
            return '{"events":[{"id":"e1","description":"x"}],"relations":[]}'

    calls = {"n": 0}

    def _factory(provider=None, model=None):
        calls["n"] += 1
        return _AlwaysGood(model=model or "gpt-4o-mini")

    monkeypatch.setattr(events_module, "LLMClient", _factory)

    checkpoint = tmp_path / "resume.checkpoint.json"
    ledger_path = tmp_path / "resume.checkpoint.json.ledger.json"
    ledger_path.write_text(
        json.dumps({
            "chunks": [
                {"chunk_index": 0, "status": "ok", "attempts": 1, "final_model": "gpt-4o-mini"}
            ]
        }),
        encoding="utf-8",
    )

    extractor = EventExtractor(use_llm=True)
    extractor.extract_from_book(
        "B" * 7000,
        source_book="Hobbit",
        chunk_size=3000,
        checkpoint_file=str(checkpoint),
        resilient=True,
        fallback_model="gpt-4o",
    )

    # Should not process all chunks from scratch because one was already marked ok
    assert calls["n"] < 4
