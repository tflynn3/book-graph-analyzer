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

### Available Commands

```bash
# Run all world-building pillars on a text
bga pipeline worldbuilding the_silmarillion.txt -t "The Silmarillion"

# Run specific pillars only
bga pipeline worldbuilding lotr.txt --pillars linguistic --pillars genealogy

# Persist genealogy artifact from canonical world-building pipeline
bga pipeline worldbuilding data/texts/the_hobbit.txt --pillars genealogy -o data/output

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
| Sociolinguistic Registers | `lore socioreg-profile`, `lore socioreg-drift`, `lore socioreg-corpus` | #47 | 🟡 closeout evidence added |
| Genealogical Layer | `pipeline worldbuilding --pillars genealogy`, `lore genealogy` | #49 | ✅ canonical pipeline stage active |
| Impression-of-Depth | `worldbible artifacts`, `lore unresolved-refs` | #50 | 🟡 closeout evidence added |
| Editorial Meta-Layer | `corpus sources`, `corpus timeline-divergence` | #51 | 🟡 closeout evidence added |
| Spatiotemporal Engine | `lore timeline-reconcile`, `lore timeline-bridge` | #48 | ✅ v2 (era mismatch + extraction bridge) |

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

Slice 7 closeout additions:
- `CorpusReconciler.add_book_from_json()` now accepts both normalized spatiotemporal payloads and raw `lore events` payloads (`{"events": {...}}` and `{"events": [...]}`), bridging automatically.
- Cross-book reconcile now namespaces imported event IDs as `<book_id>:<event_id>` to avoid accidental ID collisions across books.
- Bridged events persist editorial/source metadata (`source_id`, `editorial_status`, `source_authority_weight`) inferred from corpus book titles.
- Persisted `TimelineConflict` nodes now include source metadata for both involved events (`event_a_*` / `event_b_*` source fields).
