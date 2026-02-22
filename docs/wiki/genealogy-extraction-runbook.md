# Genealogy Extraction Runbook (Coref + LLM Validator)

## Scope
This runbook documents the genealogy pipeline after closeout of:
- **#3** local coreference/context resolution
- **#4** LLM proposal + deterministic validation

## Pipeline stages
1. **Deterministic extraction (primary)**
   - Regex rules for genealogy relations (`son of`, `father of`, `married`, etc.)
   - Appositive coverage: `Bilbo, son of Bungo Baggins, ...`
2. **Context/coreference pass (local window)**
   - Pronoun resolution to recent explicit subject (`he/she/his/her`)
   - Title carry-over across adjacent sentences (`King Aragorn ... The king ...`)
   - Name normalization + short-form carry-over
3. **LLM proposal stage (optional)**
   - LLM proposes relation candidates with required span fields:
     - `source_name`, `target_name`, `relation_type`
     - `evidence_text`, `evidence_start`, `evidence_end`, `confidence`
4. **Deterministic validator gate (hard filter)**
   - Schema validation
   - Entity resolvability checks
   - Evidence-span alignment checks against source text
   - Confidence floor (`<0.65` rejected)
   - Rejection reason codes:
     - `schema_invalid`
     - `entity_unresolvable`
     - `evidence_misaligned`
     - `low_confidence`
     - `unsupported_relation`

## Output contract
Each extracted relation now includes:
- `confidence`
- `evidence_text`
- `evidence_start`
- `evidence_end`
- `resolution_confidence`

These fields are present in both in-memory model (`GenealogyRelation`) and `genealogy_to_json(...)` payloads.

## Test coverage
- Unit tests: `tests/test_genealogy_coref_llm_validate.py`
  - pronoun resolution
  - title carry-over
  - evidence/confidence fields
  - validator reason-code rejections
  - valid proposal acceptance through extractor
- Integration/regression:
  - recall gain fixture vs baseline
  - precision safeguards (self-link prevention)

## Re-run commands
```powershell
.\.venv\Scripts\python -m pytest -q tests/test_genealogy.py tests/test_genealogy_coref_llm_validate.py
```

## Operational notes
- Deterministic extraction remains the authoritative source.
- LLM stage is additive and guarded; invalid proposals are dropped, not partially applied.
- For production runs where precision is critical, keep `llm_client` optional/off unless evidence spans are required.
