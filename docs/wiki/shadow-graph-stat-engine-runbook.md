# Runbook: Shadow Graph Statistical Engine v1

## Prerequisites

- `bga story init` project scaffold exists
- `project.json`, `constraints.json`, `plan.json` are present
- At least one events file is configured in `project.json.event_files`

## Recommended config

```json
{
  "required_elements": ["Thingol", "oath"],
  "required_element_aliases": {
    "Thingol": ["King Thingol"]
  },
  "search": {
    "target_candidates": 600
  },
  "enforcement": {
    "required_terms": true,
    "max_retries": 2
  }
}
```

## Pipeline

1. Build context priors

```bash
bga story context --project <slug> --graph-stats
```

2. Generate candidate pool + elites artifacts

```bash
bga story grow-shadow --project <slug> --auto
```

Check `shadow_candidates.json`:
- `sampling.generated_candidates` (expect >= 500 for large runs)
- `seed` (stable across identical inputs)
- `elites_grid`

3. Solve trajectory

```bash
bga story solve --project <slug>
```

4. Draft chapter

```bash
bga story draft --project <slug> --chapter 1 --grounded
```

5. Audit grounding + constraints

```bash
bga story audit --project <slug> --chapter 1 --enforce-required-terms
```

## Failure triage

- `Missing required terms`:
  - check `required_element_aliases`
  - verify terms can appear naturally in scene summaries/actions
- Audit grounding fail (`evidence_alignment.ratio` low):
  - inspect `chapter_<n>_trace.json` excerpts
  - ensure trace sections carry `source_canon_node_ids`
- Event ingestion fail (`source_book must be non-empty`):
  - ensure ingest provides `book` and event `source_book` values

## Regression checks

```bash
pytest -q tests/test_story_cli.py tests/test_event_id_book_namespace.py
```