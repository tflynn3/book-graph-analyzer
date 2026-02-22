# Parallel Event Chunk Extraction (optional)

- New optional CLI flags:
  - `--parallel-workers N` (default `1`, sequential)
  - `--max-inflight N` (optional throttle)
- Applies to:
  - `bga lore events ...`
  - `bga corpus events ...`

## Behavior and safety

- Default remains backward-compatible (`workers=1`).
- Parallelism is extraction-only (chunk processing).
- Neo4j writes are unchanged and remain single-pass after extraction.
- Resilient mode keeps per-chunk retry/fallback and ledger tracking.

## Determinism

Chunks may finish out-of-order, but merge order is deterministic by chunk index.
This keeps event/relation output stable between sequential and parallel runs.

## Recommended usage

- Start with `--parallel-workers 2` or `4` for long books.
- If API throttling appears, reduce workers or set `--max-inflight`.
- For short texts, parallel mode is usually unnecessary.
