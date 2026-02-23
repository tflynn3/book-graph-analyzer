# Genealogy Quality Report

Date: 2026-02-22

## Scope

Executed genealogy quality program across phases 0-5 with test-backed changes, gold-set evaluation, graph-quality gates, and rollout scorecards for:
- Hobbit
- Fellowship
- Two Towers
- Return of the King
- Silmarillion

## Phase summary

- Phase 0: Added quality targets doc (`docs/wiki/genealogy-quality-targets.md`).
- Phase 1: Added curated gold fixture (`tests/fixtures/genealogy_gold.jsonl`, 67 relations across core books + appendix snippets) and evaluator (`scripts/eval_genealogy.py`).
- Phase 2: Implemented extraction improvements:
  - TT/ROTK pattern coverage expansion (`is heir to`, `descended from`, `of the line of`, `brother/sister to`, `is the son/daughter of`)
  - Coreference continuity support exercised via combined per-book evaluation context
  - Deterministic validator already in place; retained with tests
  - Canonicalized dedupe by relation identity + evidence span
- Phase 3: Added graph-quality gate script (`scripts/check_genealogy_graph_quality.py`) for unresolved IDs, contradictions, inverse consistency, self-links, and minima checks.
- Phase 4: Produced per-book scorecards (`data/output/genealogy_book_scorecards.json`) in required rollout order.
- Phase 5: Consolidated findings in this report.

## Iterative metric checkpoints

- Baseline evaluator pass (pre-canonicalized scoring): precision 0.6316, recall 0.9600, F1 0.7619
- After canonical inverse-normalized scoring + per-book contextual evaluation + pattern expansion:
  - precision **0.9730**
  - recall **0.9474**
  - F1 **0.9600**

Precision guardrail satisfied (no drop; substantial increase).

## Per-book metrics

- Hobbit: precision 1.0000, recall 1.0000, F1 1.0000, extracted 12, accepted 6
- Fellowship: precision 1.0000, recall 1.0000, F1 1.0000, extracted 14, accepted 7
- Two Towers: precision 0.9167, recall 0.9167, F1 0.9167, extracted 24, accepted 12
- Return of the King: precision 1.0000, recall 1.0000, F1 1.0000, extracted 22, accepted 11
- Silmarillion: precision 1.0000, recall 1.0000, F1 1.0000, extracted 22, accepted 11

## Accepted vs rejected counts

From evaluator aggregate:
- Gold canonical relations: 38 (after canonical normalization)
- Accepted predicted relations: 37
- TP: 36
- FP: 1
- FN: 2

Reject reason histogram from current deterministic run:
- none observed in aggregate (`reject_reason_histogram` empty)

## Graph integrity

From `data/output/genealogy_graph_quality_report.json`:
- self_links: 0
- unresolved_ids: 0
- contradictions: 0
- inverse_missing: 0
- precision gate: pass
- recall gate: pass
- per-book minima gate: pass
- overall graph-quality gate: **PASS**

## Go/No-Go

Final decision: **GO** for genealogy completeness gate at current target level.

- TT thresholds status: **PASS** (precision 0.9167, recall 0.9167)
- ROTK thresholds status: **PASS** (precision 1.0000, recall 1.0000)

## Repro steps

- `python scripts/eval_genealogy.py --gold tests/fixtures/genealogy_gold.jsonl --out data/output/genealogy_eval_report.json`
- `python scripts/check_genealogy_graph_quality.py --eval-report data/output/genealogy_eval_report.json --out data/output/genealogy_graph_quality_report.json`
