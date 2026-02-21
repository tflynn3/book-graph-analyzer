# Issue #50 Slice 2 — Lore Depth Precision + Candidate Linking

Status: ✅ implemented on `feat/50-lore-depth-slice2`

## Scope delivered

1. Broken-reference precision/disambiguation
- Context windows (`context_before`, `context_after`) added during extraction
- Basic expected-type inference from local context
- Optional LLM fallback (`--llm-fallback`) for unresolved mentions not caught by heuristics

2. Resolver-backed candidate linking
- Added `ReferenceCandidate` model
- New helper `link_broken_reference_candidates(...)` uses disambiguation resolver
- Adds exact and fuzzy candidates with confidence/source metadata

3. Provenance/conflict weighting + queue exposure
- `BrokenReference` now includes:
  - `candidates`
  - `provenance_notes`
  - `conflict_weight`
- `LoreDepthExtractionResult.unresolved_queue` prioritizes unresolved refs by score
- Graph writer persists new metadata and queue query is exposed

4. CLI upgrades
- `worldbible artifacts` flags:
  - `--context-window`
  - `--link-candidates/--no-link-candidates`
  - `--llm-fallback`
- `lore unresolved-refs` now returns weighted queue with top candidate

5. Tests
- Added `tests/test_issue_50_lore_depth_slice2.py`
