# Pipeline Walkthrough

## Standard Pipeline

```bash
bga ingest data/texts/the_hobbit.txt --title "The Hobbit"
bga extract entities data/texts/the_hobbit.txt --title "The Hobbit" --show-new
bga pipeline full data/texts/the_hobbit.txt --title "The Hobbit"
```

Then use `bga worldbible --help` and `bga generate --help` for generation workflows.

---

## World-Building Pipeline

!!! note "Milestone: Tolkien World-Building (#45–#51)"
    The world-building pipeline extends the standard pipeline with five
    deep-lore layers. Each pillar is being implemented incrementally.
    See the [World-Building RFC](tolkien-worldbuilding-rfc.md) for the full design.

### Available Commands (Placeholder)

```bash
# Run all world-building pillars on a text
bga pipeline worldbuilding the_silmarillion.txt -t "The Silmarillion"

# Run specific pillars only
bga pipeline worldbuilding lotr.txt --pillars linguistic --pillars genealogy

# Linguistic lineage (etymology chains)
bga worldbible languages hobbit_bible.json

# Deep genealogy (family trees)
bga lore genealogy --character Aragorn --depth 5

# Editorial source tracking
bga corpus sources tolkien_works --show-authority
```

### The Five Pillars

| Pillar | CLI Group | Issue | Status |
|--------|-----------|-------|--------|
| Linguistic Lineage | `worldbible languages` | #46 | ✅ v1 |
| Deep Genealogy | `lore genealogy` | #47 | 🟡 MVP slice |
| Editorial Layers | `corpus sources` | #48/#51 | 🟨 Partial+ (source strata tagging, provenance metadata, MVP divergence report) |
| Cultural Rules | `worldbible cultures --rules` | #49 | 🔲 Planned |
| Spatiotemporal Engine | `lore timeline-reconcile`, `lore timeline-bridge` | #48 | ✅ v2 (era mismatch + extraction bridge) |
| Cosmological Timeline | `lore timeline --cosmological` | #50 | 🔲 Planned |

### Integration with Standard Pipeline

The world-building layers are **not** a separate pipeline. They extend
existing modules:

- **Models** — `models.worldbuilding` adds `LinguisticLineage`, `GenealogyRelation`, `EditorialLayer`
- **Graph** — `graph.writer` gains `write_linguistic_lineage()`, `write_genealogy_batch()`, `write_editorial_provenance()`
- **CLI** — New commands live under existing groups (`worldbible`, `lore`, `corpus`, `pipeline`)

### Timeline Reconciliation

The spatiotemporal engine (`spatiotemporal/`) detects timeline inconsistencies:

```bash
# Direct spatiotemporal event checking
bga lore timeline-reconcile events.json
bga lore timeline-reconcile events.json -l locations.json
bga lore timeline-reconcile events.json --format json -o report.json

# Integrated: extract -> normalize -> reconcile (slice 2)
bga lore timeline-bridge hobbit_events.json
bga lore timeline-bridge events.json -l locations.json --format json -o report.json
```

**Conflict types detected:**
- **Temporal overlap** — same character at two locations at overlapping times
- **Travel infeasibility** — entity moves faster than physically possible
- **Era mismatch** — entity's events span non-adjacent eras (likely extraction error)

**Extraction bridge (slice 2):**
The `timeline-bridge` command reads events from `bga lore events` output,
normalizes temporal expressions through the spatiotemporal normalizer, detects
conflicts including era mismatches, and reports extraction-vs-normalized
confidence deltas. This surfaces cases where the extraction was overconfident
or where normalization boosted confidence.

Slice 6 additions:
- report now includes source-attribution counts by `source_book`
- LLM causal extraction auto-batches large event sets and still falls back safely
- `bga ingest` prints inferred editorial source metadata when recognized
