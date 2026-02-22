# Issue #90 — Genealogy Non-population Follow-up

This follow-up hardens the Hobbit acceptance gate so genealogy cannot silently pass with missing or low-value output.

## What changed

- Added a **stage-required guard** for Hobbit runs:
  - `pipeline worldbuilding` now fails fast if `--pillars` omits `genealogy`.
- Kept the existing **non-zero relation guard** for Hobbit runs.
- Added **quality gates** for populated Hobbit genealogy output:
  - At least one relation must include non-null `generation_depth`.
  - Every relation must include `inheritance_traits` as a list (trait rationale output field present in artifact schema).

## Test evidence

`tests/test_pipeline_worldbuilding_genealogy.py`

- `test_pipeline_worldbuilding_hobbit_gate_requires_non_population_genealogy`
- `test_pipeline_worldbuilding_hobbit_gate_accepts_non_population_genealogy`
- `test_pipeline_worldbuilding_hobbit_gate_requires_genealogy_stage`
- `test_pipeline_worldbuilding_hobbit_gate_enforces_quality_fields`

This closes the integration gap from #49 where feature-level genealogy existed but acceptance orchestration did not enforce stage execution and artifact quality for Hobbit gate runs.

