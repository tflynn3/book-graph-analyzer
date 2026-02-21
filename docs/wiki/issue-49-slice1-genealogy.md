# Issue #49 Slice 1 — Genealogy MVP Progress

This slice delivers a functional genealogy pipeline under `bga lore genealogy`:

- Added `worldbible.genealogy` module:
  - Genealogy relation normalization
  - Rule-based extraction patterns (`son of`, `father of`, `married`, etc.)
  - Optional LLM fallback (best-effort, only used when passed)
  - JSON load/save helpers
- Implemented graph persistence in `GraphWriter.write_genealogy_batch()`
- Added query helper `GraphWriter.query_genealogy()` for CLI retrieval
- Added tests for extraction, normalization, JSON round-trip, writer calls, and CLI path

## Remaining scope (future slices)

- Better NER/entity resolution to reduce false positives in free text
- House inference from nearby context and title patterns
- Generation-depth inference via graph traversal (currently pass-through)
- Graph constraints/indexes for genealogy-specific edge patterns
- Provenance weighting and conflict-aware genealogy validation
