# Genealogy Quality Targets

This document defines acceptance targets for genealogy extraction quality across Tolkien corpus books.

## Primary quality targets

- Precision: **>= 0.90**
- Recall: **>= 0.75** (initial baseline for rollout)
- Precision guardrail per slice: **no more than 2% absolute precision drop**

## Per-book minimum extraction/acceptance thresholds

These are minimums for rollout gating (extracted = raw candidate count, accepted = post-validation retained count):

- The Hobbit: extracted >= 8, accepted >= 6
- The Fellowship of the Ring: extracted >= 10, accepted >= 7
- The Two Towers: extracted >= 10, accepted >= 8
- The Return of the King: extracted >= 10, accepted >= 8
- The Silmarillion: extracted >= 15, accepted >= 11

## Graph integrity targets

- Self-links: **0** (`source_id != target_id` for all retained relations)
- Inverse consistency: **100%** (every relation has the expected inverse)
- Unresolved IDs: **0** (`source_id` and `target_id` both non-empty)
- Contradictions: **0 hard contradictions** in direct parent/child claims
- Missing evidence: **0** retained relations with empty evidence text

## Rollout policy

Books are rolled out in this order:
1. Hobbit
2. Fellowship
3. Two Towers
4. Return of the King
5. Silmarillion

A book is materialized only if:
- precision and recall targets pass,
- per-book extracted/accepted minimums pass,
- graph integrity checks pass.
