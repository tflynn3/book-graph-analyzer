# Genealogy Metrics Report — 2026-02-22

## Summary
Closeout metrics for genealogy improvements #3 (coref/context) and #4 (LLM+validator).

## Recall gain fixture (regression/integration)
Fixture: `tests/test_genealogy_coref_llm_validate.py::test_regression_recall_gain_vs_baseline_with_precision_guardrails`

- Baseline recall: **0.667** (2/3)
- New recall: **1.000** (3/3)
- Relative gain: **+50%** on fixture
- Precision safeguards asserted:
  - no self-links
  - validator rejects low-confidence / schema-invalid / evidence-misaligned proposals

## Per-book extraction counts (before vs after)
Method:
- **Before**: deterministic explicit-name patterns only (pre-coref behavior class)
- **After**: current `extract_genealogy_from_text(...)` with coref/context pass
- Corpus files:
  - `data/texts/lotr-corpus/fellowship.txt`
  - `data/texts/lotr-corpus/twotowers.txt`
  - `data/texts/lotr-corpus/return.txt`

Results:
- Fellowship: before **32**, after **32**, delta **0**
- Two Towers: before **34**, after **34**, delta **0**
- Return of the King: before **22**, after **22**, delta **0**

Interpretation:
- No regression on main LOTR corpus counts.
- Fixture-based recall gain confirms improved coverage for pronoun/title patterns not materially present in these three corpora.

## Precision / safety checks on per-book run
- Self-links: **0** for all books
- Relations with missing evidence spans: **0** for all books
- Relations with confidence < 0.65: **0** for all books
