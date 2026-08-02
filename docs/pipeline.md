# Pipeline Walkthrough

## Standard Pipeline

```bash
bga ingest data/texts/the_hobbit.txt --title "The Hobbit"
bga extract entities data/texts/the_hobbit.txt --title "The Hobbit" --show-new
bga pipeline full data/texts/the_hobbit.txt --title "The Hobbit"
bga corpus process tolkien_works
```

Then use `bga worldbible --help` and `bga generate --help` for generation workflows.

### Which command to use

- `bga pipeline full` is the primary single-book graph rebuild path. It runs the seeded entity, relationship, and proposition extractor, then writes passages, entity mentions, canon relationships, unresolved references, dense proposition nodes, style, and voice artifacts when Neo4j is available.
- `bga corpus process` now uses the same seeded extraction path per book, then refreshes the cross-book entity index. Use it when you want the lore graph rebuilt across an entire corpus from one command.
- `bga analyze` remains the generic zero-seed path. It is useful for exploratory extraction on arbitrary novels, but it is not the preferred Tolkien lore-graph rebuild workflow.
- The passage splitter now trims obvious contents-page / foreword / appendix scaffolding from the public LOTR text dumps and prefers standalone body markers like `_Chapter 1_` when both TOC entries and real chapter headings are present.

### Interpreting the graph

- `(:Character|:Place|:Object)` plus direct relationship edges form the stricter canon graph.
- `(:Proposition)` nodes capture dense sentence-level semantics. A single sentence can yield multiple propositions such as movement, possession, speech, and attribute facts.
- Resolved proposition arguments become `ARGUMENT_IN` links from canonical entities to proposition nodes.
- Unresolved proposition arguments become `HAS_UNRESOLVED_ARGUMENT` links from proposition nodes to `UnresolvedReference` nodes instead of being dropped.
- Unresolved references are now bucketed by `reference_class` such as `pronoun`, `discourse_deictic`, `bridging`, `body_part`, and `canon_candidate` so review queues and quality reports can distinguish different failure modes.
- The proposition extractor now performs deterministic quote-local reference grounding before fallback: first-person pronouns inside attributed dialogue resolve to the speaker, second-person pronouns can resolve to an explicitly addressed character, and recent third-person antecedents can carry across adjacent passages.
- Dialogue extraction now reads both double-quoted and single-quoted speech. Short scare quotes and other quoted narrative fragments are filtered out before voice profiling so words like `"colonists"` or `"gatherers"` do not become fake speakers.
- Direct canon-edge projection is now type-gated. For example, `TRAVELED_TO` only persists for `Character -> Place` pairs and `POSSESSES` only persists for `Character -> Object` pairs, so obviously bad pairings stay out of the strict graph even if a coarse extractor proposed them.
- The direct relationship extractor also skips non-asserted clauses such as negated or modalized claims (`could not kill`, `would go`, etc.), so counterfactual or hypothetical wording does not become canon edges.
- Voice profiles now attach only to already grounded character nodes. Unmatched quote-attribution guesses are skipped instead of creating new `Character` nodes with no passage provenance, multiple speaker aliases that resolve to the same entity ID are merged before the profile is written, and `corpus process` writes one cumulative cross-book profile per character instead of letting the last processed book overwrite earlier voice data.
- `canon_candidate` unresolved routing now rejects obvious function-word noise before the review queue is populated, so items like bare prepositions and pronouns do not masquerade as plausible lore entities.
- Relationship-side unresolved references now use the same `reference_class` taxonomy as proposition unresolveds, so pronouns like `he` stay in the pronoun bucket instead of inflating the canon-candidate review queue.
- `bga lore resolve-unresolved` can now run a staged hosted-model repair pass over the live `:UnresolvedReference` queue. It writes `llm_resolution_*` audit properties back to the unresolved nodes and only auto-applies safe `existing` matches that map to current character inventory IDs.

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

## Genealogy extraction runbook

- Coreference/context + LLM-validator closeout runbook:
  - `docs/wiki/genealogy-extraction-runbook.md`
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
