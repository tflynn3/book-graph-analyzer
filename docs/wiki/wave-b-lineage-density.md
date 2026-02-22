# Wave B — Linguistic lineage density expansion

## What changed

- Added `book_graph_analyzer.worldbible.lineage_density` for corpus-backed lineage generation.
- Expanded lineage generation to use real per-book surface text from:
  - event descriptions/agents/patients/actions (`data/output/*events.json`)
  - lore artifact names/descriptions (`data/output/layer_load/*_lore_depth.json`)
- Added canonical namespace + join-safe generation through `parse_lineage()`.
- Added per-book lineage thresholds and pass/fail evaluation.
- Added idempotent per-book graph rewrite logic in `scripts/waveB_lineage_density.py`.

## Run

```bash
python scripts/waveB_lineage_density.py
```

Outputs:
- rewritten `data/output/layer_load/*_lineages.json`
- `data/output/layer_load/lineage_density_report.json`

## Acceptance metrics

- join_rate >= 0.95 for each book
- counts meet per-book thresholds in `BOOK_THRESHOLDS`
