# Issue #93 — Lore Depth Precision/Quality Gates

## Scope
Adds quality gates for unresolved-reference extraction and candidate-linking audit buckets.

## What shipped
- `evaluate_unresolved_quality_gates(...)` in `lore.depth`
  - computes unresolved context coverage
  - computes unresolved candidate coverage
  - returns pass/fail with failure IDs
- `build_candidate_linking_audit_buckets(...)` in `lore.depth`
  - buckets unresolved references into:
    - `no_candidates`
    - `weak_top_candidate`
    - `ambiguous_top_candidates`
    - `high_confidence_top_candidate`
- `bga worldbible artifacts` quality-gate controls:
  - `--quality-gate/--no-quality-gate` (default: on)
  - `--min-context-coverage` (default: `0.85`)
  - `--min-candidate-coverage` (default: `0.60`)
  - exits with code `2` when gate fails
- JSON output now includes `quality_gate` report payload when output is requested.

## Why
This provides a hard stop when unresolved-reference extraction quality drops, and a structured audit surface for candidate-linking triage.

## Tests
- `tests/test_issue_93_lore_depth_quality.py`
  - audit bucket classification
  - gate failure reporting
  - CLI non-zero exit behavior when gate fails
