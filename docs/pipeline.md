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
| Linguistic Lineage | `worldbible languages` | #46 | 🔲 Stub |
| Deep Genealogy | `lore genealogy` | #47 | 🔲 Stub |
| Editorial Layers | `corpus sources` | #48 | 🔲 Stub |
| Cultural Rules | `worldbible cultures --rules` | #49 | 🔲 Planned |
| Cosmological Timeline | `lore timeline --cosmological` | #50 | 🔲 Planned |

### Integration with Standard Pipeline

The world-building layers are **not** a separate pipeline. They extend
existing modules:

- **Models** — `models.worldbuilding` adds `LinguisticLineage`, `GenealogyRelation`, `EditorialLayer`
- **Graph** — `graph.writer` gains `write_linguistic_lineage()`, `write_genealogy_batch()`, `write_editorial_provenance()`
- **CLI** — New commands live under existing groups (`worldbible`, `lore`, `corpus`, `pipeline`)
