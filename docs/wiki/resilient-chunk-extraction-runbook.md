# Resilient Chunk Extraction Mode (Runbook)

## What it does

For chunked LLM extraction flows (currently lore events + corpus events), resilient mode applies this policy per chunk:

1. Try parse/normalize + persist normally.
2. If malformed/failed: retry once on the same model.
3. If still failing: retry once on a stronger fallback model.
4. If still failing: mark chunk `failed_unprocessed` and continue the run.

This prevents long runs from aborting on a few bad chunks.

## Artifacts

Given checkpoint file `X.checkpoint.json`, resilient mode writes:

- `X.checkpoint.json` → periodic safe-write checkpoint (partial extracted events/relations)
- `X.checkpoint.json.ledger.json` → chunk ledger with status per chunk
- `X.checkpoint.json.payloads/` → payload snippets for failed chunks (debug/audit)

Ledger statuses:

- `ok`
- `retry_success`
- `fallback_success`
- `failed_unprocessed`

## Resume behavior

On rerun with the same checkpoint/ledger:

- Completed chunks (`ok/retry_success/fallback_success`) are skipped.
- Only failed/unprocessed chunks are retried.
- End-of-run summary reports: `ok`, `retried`, `fallback_success`, `failed`.

## CLI usage

### Single book events

```bash
bga lore events data/texts/the_hobbit.txt \
  -o data/output/hobbit_events.json \
  --chunk-size 3000 \
  --checkpoint data/checkpoints/hobbit_events.checkpoint.json \
  --resilient
```

### Corpus events

```bash
bga corpus events tolkien_works \
  -o data/output/tolkien_events.json \
  --chunk-size 3000 \
  --resilient \
  --checkpoint-dir data/checkpoints
```

## Operational notes

- Defaults remain backward-compatible (`--resilient` is opt-in).
- Failed chunks are never silently dropped: ledger reason + payload snippet path are persisted.
- You can inspect failed chunks quickly via `*.ledger.json` and rerun with same checkpoint to retry only those chunks.
