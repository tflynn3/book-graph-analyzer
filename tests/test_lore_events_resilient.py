import json

from book_graph_analyzer.lore import events as events_module
from book_graph_analyzer.lore.events import Event, EventExtractor


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
    checkpoint.write_text(
        json.dumps({
            "next_chunk": 1,
            "total_chunks": 3,
            "events": [
                Event(id="e1", description="x", source_book="Hobbit").to_dict()
            ],
            "relations": [],
            "seen_keys": ["x"],
        }),
        encoding="utf-8",
    )
    ledger_path.write_text(
        json.dumps({
            "chunks": [
                {"chunk_index": 0, "status": "ok", "attempts": 1, "final_model": "gpt-4o-mini"}
            ]
        }),
        encoding="utf-8",
    )

    extractor = EventExtractor(use_llm=True)
    graph = extractor.extract_from_book(
        "B" * 7000,
        source_book="Hobbit",
        chunk_size=3000,
        checkpoint_file=str(checkpoint),
        resilient=True,
        fallback_model="gpt-4o",
    )

    # Should not process all chunks from scratch because one was already marked ok
    assert calls["n"] < 4
    assert len(graph.events) == 1  # Identical payloads are deliberately deduplicated.
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["next_chunk"] == 3


def test_resilient_resume_reprocesses_ledger_entry_without_durable_payload(
    tmp_path,
    monkeypatch,
):
    processed = []

    def fake_once(self, text, source_book, chunk_index=0, model=None):
        processed.append(chunk_index)
        event = Event(
            id=f"c{chunk_index}_e",
            description=f"event {chunk_index}",
            source_book=source_book,
        )
        return [event], [], "", "{}"

    monkeypatch.setattr(events_module, "LLMClient", lambda **_kwargs: _FakeLLM())
    monkeypatch.setattr(EventExtractor, "_extract_llm_once", fake_once)

    checkpoint = tmp_path / "ledger-only.checkpoint.json"
    checkpoint.with_suffix(".json.ledger.json").write_text(
        json.dumps({
            "chunks": [
                {
                    "chunk_index": 1,
                    "status": "ok",
                    "attempts": 1,
                    "final_model": "primary",
                }
            ]
        }),
        encoding="utf-8",
    )

    graph = EventExtractor(use_llm=True).extract_from_book(
        "C" * 7000,
        source_book="Hobbit",
        chunk_size=3000,
        checkpoint_file=str(checkpoint),
        resilient=True,
        fallback_model="fallback",
        parallel_workers=3,
    )

    assert sorted(processed) == [0, 1, 2]
    assert len(graph.events) == 3
