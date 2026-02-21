# Issue #47 — Sociolinguistic Registers (Slice 1)

This closeout slice extends the MVP into a safer, corpus-aware workflow.

## Implemented

- `SociolinguisticRegisterClassifier` (rule-based)
  - register scoring from lexical cues
  - profile metrics: formality, archaism, contractions, sentence length
  - optional model-assisted hook with strict safe fallback to rule-only result
- `detect_register_drift(baseline, current)`
  - compares dominant register and metric shifts
  - severity bucketing (`low`/`medium`/`high`)
- `profile_corpus_registers(samples)`
  - corpus-wide dominant register distribution
  - per-entity latest profile snapshot
  - strongest longitudinal drift transitions
- `GraphWriter` helpers
  - `write_register_profile(entity_id, profile, source_passage_id)`
  - `write_register_observation(entity_id, profile, observed_at, source_passage_id)`
  - `query_register_drift(entity_id, min_delta, limit)`
  - `query_register_observations(entity_id, limit)`
  - `query_register_drift_summary(entity_id, min_delta, limit)`
- CLI (`lore` group)
  - `bga lore socioreg-profile --text ...`
  - `bga lore socioreg-drift --baseline ... --current ...`
  - `bga lore socioreg-corpus --input samples.json`
  - JSON ergonomics: `--json` on profile/drift/corpus commands
  - optional style/voice alignment hint via `--voice-formality`

## Notes

- This is intentionally deterministic and lightweight; no LLM dependency.
- Designed to compose with existing prose-register tooling (`bga lore classify`) rather than replace it.

## Remaining scope

- wire `socioreg-corpus` directly into `bga corpus run` pipeline output (currently file-in/file-out)
- optional persistent corpus report nodes (aggregate snapshots) in graph
- connect drift summary into generation QA score weighting (currently CLI/report only)
