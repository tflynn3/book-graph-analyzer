import json

from book_graph_analyzer.extract.resilient_chunk_runner import (
    ResilientChunkRunner,
    ChunkAttemptResult,
    ChunkStatus,
)


def test_runner_retries_then_fallback_success(tmp_path):
    ledger = tmp_path / "ledger.json"
    runner = ResilientChunkRunner(ledger)

    calls = []

    def process_attempt(i, chunk, model, attempt_no):
        calls.append((i, model, attempt_no))
        if attempt_no < 3:
            return ChunkAttemptResult(False, reason="malformed_json", model=model)
        return ChunkAttemptResult(True, model=model)

    persisted = {"count": 0}

    def persist_artifact():
        persisted["count"] += 1

    state = runner.run(
        chunks=["a"],
        primary_model="m1",
        fallback_model="m2",
        process_attempt=process_attempt,
        persist_artifact=persist_artifact,
    )

    assert calls == [(0, "m1", 1), (0, "m1", 2), (0, "m2", 3)]
    assert state[0].status == ChunkStatus.FALLBACK_SUCCESS
    data = json.loads(ledger.read_text(encoding="utf-8"))
    assert data["metrics"]["fallback_success"] == 1


def test_runner_resume_only_failed_chunks(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "chunks": [
                    {"chunk_index": 0, "status": "ok", "attempts": 1, "final_model": "m1"},
                    {"chunk_index": 1, "status": "failed_unprocessed", "attempts": 3, "final_model": "m2"},
                ]
            }
        ),
        encoding="utf-8",
    )

    runner = ResilientChunkRunner(ledger)
    seen = []

    def process_attempt(i, chunk, model, attempt_no):
        seen.append(i)
        return ChunkAttemptResult(True, model=model)

    runner.run(
        chunks=["a", "b"],
        primary_model="m1",
        fallback_model="m2",
        process_attempt=process_attempt,
        persist_artifact=lambda: None,
    )

    assert seen == [1]
