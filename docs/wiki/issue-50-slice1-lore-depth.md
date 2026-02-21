# Issue #50 Slice 1 — Lore Depth Engine Kickoff

Status: ✅ implemented on `feat/50-lore-depth-slice1`

## Scope delivered

1. First-class models
- `LoreArtifactType` (`song`, `poem`, `artifact`)
- `LoreArtifact`
- `BrokenReference`
- `LoreDepthExtractionResult`

2. Extraction
- `extract_lore_depth(text, source_book, passage_id)` in `src/book_graph_analyzer/lore/depth.py`
- Detects:
  - artifact-like mentions (`song/poem/artifact/relic/...`)
  - unresolved markers (`[[...]]`, `unknown/unnamed/forgotten ...`)

3. GraphWriter integration
- `write_lore_artifacts_batch(...)`
- `write_broken_references_batch(...)`
- `query_lore_artifacts(...)`
- `query_unresolved_references(...)`

4. CLI integration (existing groups)
- `bga worldbible artifacts <path> [--output out.json] [--write-graph]`
- `bga lore unresolved-refs [--book BOOK] [--limit N]`

5. Tests
- `tests/test_issue_50_lore_depth_slice1.py`

## Notes

This slice is additive/backward-compatible and intentionally heuristic-first.
Later slices should improve extraction precision and candidate-link resolution.
