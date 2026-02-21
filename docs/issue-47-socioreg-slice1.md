# Issue #47 — Sociolinguistic Registers (Slice 1)

This slice adds an integrated, rule-first MVP for sociolinguistic register tracking.

## Implemented

- `SociolinguisticRegisterClassifier` (rule-based)
  - register scoring from lexical cues
  - profile metrics: formality, archaism, contractions, sentence length
- `detect_register_drift(baseline, current)`
  - compares dominant register and metric shifts
  - severity bucketing (`low`/`medium`/`high`)
- `GraphWriter` helpers
  - `write_register_profile(entity_id, profile, source_passage_id)`
  - `write_register_observation(entity_id, profile, observed_at, source_passage_id)`
  - `query_register_drift(entity_id, min_delta, limit)`
- CLI (`lore` group)
  - `bga lore socioreg-profile --text ...`
  - `bga lore socioreg-drift --baseline ... --current ...`

## Notes

- This is intentionally deterministic and lightweight; no LLM dependency.
- Designed to compose with existing prose-register tooling (`bga lore classify`) rather than replace it.

## Remaining scope

- plug into corpus extraction flow for automatic profile generation per character/era
- add optional fallback classifier (embedding/LLM) behind a feature flag
- add Neo4j query views for longitudinal drift dashboards
- connect drift checks to generation QA gates
