# Issue #48 — Closeout Slice 7

## What this slice closes

- Added end-to-end source/editorial metadata persistence for spatiotemporal events.
- Added source metadata propagation onto persisted timeline conflicts.
- Improved cross-book reconciliation ingestion compatibility for corpus pipelines.
- Added a real fixture-backed integration test for corpus reconciliation.

## Key implementation notes

- `SpatiotemporalEvent` now carries:
  - `source_id`
  - `editorial_status`
  - `source_authority_weight`
- `TimelineConflict` now carries:
  - `event_a_source_book`, `event_b_source_book`
  - `event_a_source_authority_weight`, `event_b_source_authority_weight`
- `ExtractionBridge.bridge_event()` infers editorial layers from source book and populates metadata.
- `ConflictDetector.detect_conflicts()` backfills conflict source metadata from involved events.
- `GraphWriter` now persists and queries the new event/conflict source fields.
- `CorpusReconciler.add_book_from_json()` now supports:
  - normalized spatiotemporal event JSON (`{"events": [...]}`)
  - raw lore-event JSON (`{"events": {...}}` and `{"events": [...]}`)
- Imported events are namespaced as `<book_id>:<event_id>` to avoid cross-book ID collisions.

## Validation

- New test: `tests/test_issue48_closeout_integration.py`
- Uses real fixtures:
  - `data/output/hobbit_events.json`
  - `data/output/unfinished_tales_events.json`
- Verifies:
  - fixture loading through corpus reconciler path
  - event ID namespacing
  - editorial/source metadata inferred and retained end-to-end
