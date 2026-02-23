# Story Workflow CLI (Product UX)

The `story` command group provides a friendly workflow for project setup, planning, validation, and graph-grounded chapter synthesis.

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

## 4) Build graph-native context stats

```bash
bga story context --project mithrandir-east --graph-stats
```

Generates:

- `data/projects/<slug>/context_stats.json`

Stats include:

- event transition probabilities
- motif/reference density priors
- character participation priors
- register/style budgets

## 5) Grow probabilistic shadow graph

```bash
bga story grow-shadow --project mithrandir-east --auto
```

Generates:

- `data/projects/<slug>/shadow_graph.json`
- `data/projects/<slug>/shadow_candidates.json`

## 6) Solve best valid trajectory

```bash
bga story solve --project mithrandir-east
```

Generates:

- `data/projects/<slug>/shadow_solution.json`

## 7) Draft grounded chapter prose

```bash
bga story draft --project mithrandir-east --chapter 1 --grounded
```

Generates:

- `data/projects/<slug>/chapter_01.md`
- `data/projects/<slug>/chapter_01_trace.json`

## 8) Audit chapter grounding and constraints

```bash
bga story audit --project mithrandir-east --chapter 1
```

Generates:

- `data/projects/<slug>/chapter_01_audit.json`
- `data/projects/<slug>/chapter_01_audit.md`

## Notes

- Basic flow requires no manual JSON editing.
- Canon integration is read-only in this iteration (uses configured canon file when present).
- `story plan` currently supports `--auto` mode only.
- `story grow-shadow` currently supports `--auto` mode only.
