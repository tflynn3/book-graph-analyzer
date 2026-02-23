# Story Workflow CLI (Product UX)

The `story` command group provides a friendly workflow for project setup, auto-planning, and validation.

## 1) Initialize a project

Interactive wizard:

```bash
bga story init
```

Non-interactive (CI-friendly):

```bash
bga story init \
  --name "Mithrandir in the East" \
  --slug mithrandir-east \
  --premise "A covert mission beyond Rhun." \
  --genre fantasy \
  --target-chapters 8 \
  --scenes-per-chapter 3 \
  --non-interactive
```

Creates scaffold under:

- `data/projects/<slug>/project.json`
- `data/projects/<slug>/constraints.json`
- `data/projects/<slug>/story_bible.md`
- `data/projects/<slug>/plan.json` (placeholder)

## 2) Auto-generate a plan

```bash
bga story plan --project mithrandir-east --auto
```

Produces chapter + scene artifact:

- `data/projects/<slug>/plan.json`

## 3) Validate continuity/style/canon checks

```bash
bga story validate --project mithrandir-east
```

Outputs:

- Human-readable: `data/projects/<slug>/validation_report.md`
- JSON artifact: `data/projects/<slug>/validation_report.json`

You can also override JSON output path:

```bash
bga story validate --project mithrandir-east --json-out artifacts/mithrandir-validate.json
```

## 4) Draft grounded chapters with required-term enforcement

```bash
bga story draft --project mithrandir-east --chapter 1
```

Behavior:

- Reads `constraints.json.required_elements` as required terms.
- Injects required terms into grounded generation constraints.
- Post-checks generated chapter text and regenerates when required terms are missing.
- Retry cap defaults to `constraints.json.enforcement.max_retries` (default: `2`).
- Fails clearly if still missing after retries.

Artifacts:

- `data/projects/<slug>/chapters/chapter-01.md`
- `data/projects/<slug>/chapters/chapter-01.draft.json` (attempt + enforcement metadata)

Tradeoffs:

- Stronger canon adherence for mandatory phrases/anchors.
- Can reduce prose naturalness if required terms are very long or highly specific.
- Retry loops increase generation cost/latency when the model repeatedly misses terms.

## 5) Audit chapter required/forbidden terms

```bash
bga story audit --project mithrandir-east --chapter 1 --enforce-required-terms
```

When enforcement is on, missing required terms are emitted as `ERROR` issues (status `FAIL`).
When enforcement is off, missing required terms are warnings only.

## Notes

- Basic flow requires no manual JSON editing.
- Canon integration is read-only in this iteration (uses configured canon file when present).
- `story plan` currently supports `--auto` mode only.
