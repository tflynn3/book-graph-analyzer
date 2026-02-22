import time

from book_graph_analyzer.lore.events import Event, EventExtractor


def _chunked_text(chunk_count: int, chunk_size: int = 120) -> str:
    return ("A" * chunk_size) * chunk_count


def test_parallel_mode_deterministic_output(monkeypatch):
    def fake_extract(self, text, source_book, chunk_index=0):
        # Intentionally stagger completion to force out-of-order finishes.
        time.sleep(0.01 * (3 - (chunk_index % 3)))
        ev = Event(
            id=f"c{chunk_index}_e",
            description=f"event {chunk_index}",
            agent=f"Agent{chunk_index}",
            action="did",
            patient=f"Thing{chunk_index}",
            source_book=source_book,
        )
        return [ev], []

    monkeypatch.setattr(EventExtractor, "_extract_llm", fake_extract)

    text = _chunked_text(8)
    seq = EventExtractor(use_llm=True).extract_from_book(
        text,
        source_book="Hobbit",
        chunk_size=120,
        overlap=0,
        parallel_workers=1,
    )
    par = EventExtractor(use_llm=True).extract_from_book(
        text,
        source_book="Hobbit",
        chunk_size=120,
        overlap=0,
        parallel_workers=4,
        max_inflight=4,
    )

    assert list(seq.events.keys()) == list(par.events.keys())
    assert [r.to_dict() for r in seq.relations] == [r.to_dict() for r in par.relations]


def test_resilient_parallel_ledger_integrity(tmp_path, monkeypatch):
    def fake_once(self, text, source_book, chunk_index=0, model=None):
        if model == "primary":
            return [], [], "malformed_json", "bad"
        ev = Event(id=f"c{chunk_index}_ok", description=f"ok {chunk_index}", source_book=source_book)
        return [ev], [], "", "{}"

    monkeypatch.setattr(EventExtractor, "_extract_llm_once", fake_once)

    checkpoint = tmp_path / "parallel.checkpoint.json"
    graph = EventExtractor(use_llm=True).extract_from_book(
        _chunked_text(6),
        source_book="Hobbit",
        chunk_size=120,
        overlap=0,
        checkpoint_file=str(checkpoint),
        resilient=True,
        fallback_model="fallback",
        parallel_workers=4,
        max_inflight=4,
    )

    assert len(graph.events) == 6
    ledger = (tmp_path / "parallel.checkpoint.json.ledger.json").read_text(encoding="utf-8")
    assert "fallback_success" in ledger
