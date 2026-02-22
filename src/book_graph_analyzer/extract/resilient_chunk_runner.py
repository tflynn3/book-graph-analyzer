"""Reusable resilient chunk processing runner.

Implements a small state machine for chunked LLM extraction with:
- retry on same model
- retry on fallback model
- ledger persistence for resume
- periodic artifact writes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait


class ChunkStatus(str, Enum):
    OK = "ok"
    RETRY_SUCCESS = "retry_success"
    FALLBACK_SUCCESS = "fallback_success"
    FAILED_UNPROCESSED = "failed_unprocessed"


@dataclass
class ChunkAttemptResult:
    success: bool
    reason: str = ""
    model: str = ""
    payload_snippet_path: Optional[str] = None
    data: object | None = None


@dataclass
class ChunkLedgerEntry:
    chunk_index: int
    status: ChunkStatus
    attempts: int
    final_model: str
    reason: str = ""
    payload_snippet_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "chunk_index": self.chunk_index,
            "status": self.status.value,
            "attempts": self.attempts,
            "final_model": self.final_model,
            "reason": self.reason,
            "payload_snippet_path": self.payload_snippet_path,
        }


@dataclass
class ChunkRunMetrics:
    ok: int = 0
    retried: int = 0
    fallback_success: int = 0
    failed: int = 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "retried": self.retried,
            "fallback_success": self.fallback_success,
            "failed": self.failed,
        }


@dataclass
class ChunkRunState:
    ledger: dict[int, ChunkLedgerEntry] = field(default_factory=dict)


class ResilientChunkRunner:
    """Reusable resilient runner for chunk-based extraction."""

    def __init__(self, ledger_file: str | Path, artifact_write_every: int = 1):
        self.ledger_file = Path(ledger_file)
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_write_every = max(1, artifact_write_every)
        self.state = ChunkRunState()
        self.metrics = ChunkRunMetrics()
        self._load_ledger()

    def _atomic_write_json(self, path: Path, data: dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _load_ledger(self) -> None:
        if not self.ledger_file.exists():
            return
        try:
            raw = json.loads(self.ledger_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        for row in raw.get("chunks", []):
            try:
                entry = ChunkLedgerEntry(
                    chunk_index=int(row["chunk_index"]),
                    status=ChunkStatus(row["status"]),
                    attempts=int(row.get("attempts", 0)),
                    final_model=str(row.get("final_model", "")),
                    reason=str(row.get("reason", "")),
                    payload_snippet_path=row.get("payload_snippet_path"),
                )
                self.state.ledger[entry.chunk_index] = entry
            except Exception:
                continue

    def _persist_ledger(self) -> None:
        rows = [self.state.ledger[i].to_dict() for i in sorted(self.state.ledger.keys())]
        self._atomic_write_json(self.ledger_file, {"chunks": rows, "metrics": self.metrics.to_dict()})

    def is_done(self, chunk_index: int) -> bool:
        e = self.state.ledger.get(chunk_index)
        return bool(e and e.status in {ChunkStatus.OK, ChunkStatus.RETRY_SUCCESS, ChunkStatus.FALLBACK_SUCCESS})

    def run(
        self,
        *,
        chunks: list[str],
        primary_model: str,
        fallback_model: str,
        process_attempt: Callable[[int, str, str, int], ChunkAttemptResult],
        persist_artifact: Callable[[], None],
        on_chunk_complete: Optional[Callable[[int, ChunkAttemptResult], None]] = None,
        workers: int = 1,
        max_inflight: Optional[int] = None,
    ) -> dict[int, ChunkLedgerEntry]:
        def _execute_chunk(i: int, chunk: str) -> tuple[int, ChunkLedgerEntry, ChunkAttemptResult]:
            attempts = [primary_model, primary_model, fallback_model]
            final_entry: Optional[ChunkLedgerEntry] = None
            final_result: Optional[ChunkAttemptResult] = None

            for attempt_no, model in enumerate(attempts, start=1):
                result = process_attempt(i, chunk, model, attempt_no)
                final_result = result
                if result.success:
                    status = ChunkStatus.OK
                    if attempt_no == 2:
                        status = ChunkStatus.RETRY_SUCCESS
                    elif attempt_no == 3:
                        status = ChunkStatus.FALLBACK_SUCCESS
                    final_entry = ChunkLedgerEntry(
                        chunk_index=i,
                        status=status,
                        attempts=attempt_no,
                        final_model=result.model or model,
                        reason=result.reason,
                        payload_snippet_path=result.payload_snippet_path,
                    )
                    break

                final_entry = ChunkLedgerEntry(
                    chunk_index=i,
                    status=ChunkStatus.FAILED_UNPROCESSED,
                    attempts=attempt_no,
                    final_model=result.model or model,
                    reason=result.reason,
                    payload_snippet_path=result.payload_snippet_path,
                )

            assert final_entry is not None and final_result is not None
            return i, final_entry, final_result

        workers = max(1, workers)
        max_inflight = max(1, max_inflight or workers)

        def _commit_chunk(i: int, final_entry: ChunkLedgerEntry, final_result: ChunkAttemptResult) -> None:
            self.state.ledger[i] = final_entry

            if final_entry.status == ChunkStatus.OK:
                self.metrics.ok += 1
            elif final_entry.status == ChunkStatus.RETRY_SUCCESS:
                self.metrics.retried += 1
            elif final_entry.status == ChunkStatus.FALLBACK_SUCCESS:
                self.metrics.fallback_success += 1
            else:
                self.metrics.failed += 1

            if on_chunk_complete and final_result.success:
                on_chunk_complete(i, final_result)

            if (len(self.state.ledger)) % self.artifact_write_every == 0:
                persist_artifact()
            self._persist_ledger()

        pending: list[tuple[int, str]] = []
        for i, chunk in enumerate(chunks):
            if self.is_done(i):
                continue
            pending.append((i, chunk))

        if workers == 1:
            for i, chunk in pending:
                idx, entry, result = _execute_chunk(i, chunk)
                _commit_chunk(idx, entry, result)
            persist_artifact()
            self._persist_ledger()
            return self.state.ledger

        in_flight: dict[Future, tuple[int, str]] = {}
        pending_iter = iter(pending)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                while len(in_flight) < max_inflight:
                    try:
                        i, chunk = next(pending_iter)
                    except StopIteration:
                        break
                    in_flight[pool.submit(_execute_chunk, i, chunk)] = (i, chunk)

                if not in_flight:
                    break

                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    in_flight.pop(fut, None)
                    idx, entry, result = fut.result()
                    _commit_chunk(idx, entry, result)

        persist_artifact()
        self._persist_ledger()
        return self.state.ledger
