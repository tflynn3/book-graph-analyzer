# Shadow Graph Statistical Engine v1 (MAP-Elites + Grounding Hardening)

This closeout implements the v1 hardening and scale-up work for shadow graph candidate generation and selection.

## Acceptance mapping

### 1) P0/P1 hardening

- Semantic grounding checks:
  - `story audit` now computes evidence alignment from `chapter_<n>_trace.json.sections[].text_excerpt` against chapter text.
  - Fails when alignment ratio drops below threshold or trace sections omit `source_canon_node_ids`.
- Required-term matcher:
  - Token-boundary phrase matching replaces naive substring checks.
  - Alias-aware matching supported by `constraints.required_element_aliases`.
  - Audit emits per-scene required-term coverage.
- Deterministic seed fallback:
  - `grow-shadow` uses stable SHA-256 based seed from `(project_slug, plan, constraints)`.
- Accurate relation write counts:
  - Relation batch Cypher now returns `count(r)` and writer stats use matched count.
- Non-empty `source_book` namespace policy:
  - Graph writer now hard-fails event writes with empty namespace book.

### 2) Sample / score / select flow for 500+

- Candidate generation scales via `constraints.search.target_candidates` (default `max(500, scenes*24)`).
- Interpretable scoring fields per candidate:
  - `score_components` (transition, character participation, motif grounding, constraint bonus)
  - `score_total`
  - `behavior_descriptor`
- MAP-Elites-style artifact:
  - `elites_grid` stores best candidate per behavior cell.
- Added `sampling` metadata to `shadow_candidates.json` for observability.

### 3) Tests

Added/expanded tests for:
- alias-aware + token-boundary required-term enforcement
- deterministic seed behavior + 500+ candidate generation
- semantic grounding alignment failures
- non-empty `source_book` enforcement
- relation write count accuracy regression

## Operational notes

- Existing projects can opt in to larger pools via:

```json
{
  "search": { "target_candidates": 800 }
}
```

- Required-term aliases are optional and backward compatible:

```json
{
  "required_elements": ["Tol-in-Gaurhoth confrontation"],
  "required_element_aliases": {
    "Tol-in-Gaurhoth confrontation": ["isle of werewolves confrontation"]
  }
}
```