# Issues #47 / #49 / #50 / #51 — Milestone Closeout Evidence

This page captures acceptance-focused evidence added in the final polish pass.

## Scope

- #47 Sociolinguistic Registers
- #49 Genealogical Layer
- #50 Impression-of-Depth Engine
- #51 Editorial Meta-Layer

## Acceptance Evidence (deterministic test fixtures)

Implemented in `tests/test_milestone_47_49_50_51_acceptance.py`:

1. **#47 socioreg acceptance fixture**
   - corpus fixture produces **>=4 detected dominant register families**
   - includes per-entity ordered samples and asserts drift signals are emitted

2. **#49 genealogy acceptance fixture**
   - multi-generational chain extraction (`Elendil -> Isildur -> Valandil -> Eldacar`)
   - chain traversal assertion proves multi-generational lineage reconstruction path
   - trait rationale survives output payload via `inheritance_traits` in `genealogy_to_json(...)`

3. **#50 lore-depth acceptance fixture**
   - artifact mentions and unresolved references extracted from the same passage
   - unresolved references include contextual windows and expected-type inference
   - candidate-link pass executes on unresolved queue

4. **#51 editorial-layer acceptance fixture**
   - sample analysis with **3 strata** (`core_text`, `appendix`, `gloss`)
   - divergence detector reports both factual contradictions and style drift

## User-facing output consistency

- Genealogy output artifacts already include `inheritance_traits` in JSON export path.
- This polish pass confirms that those trait rationales are asserted in tests as closeout evidence.

## Remaining caveat

These fixtures are deterministic acceptance proofs. Canonical corpus benchmark snapshots can still be expanded in follow-up hardening if desired (precision/recall tuning and broader ambiguity sets).
