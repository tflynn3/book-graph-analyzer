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

## Notes

- Basic flow requires no manual JSON editing.
- Canon integration is read-only in this iteration (uses configured canon file when present).
- `story plan` currently supports `--auto` mode only.
