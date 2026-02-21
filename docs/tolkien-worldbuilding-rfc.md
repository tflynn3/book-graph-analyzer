# RFC: Tolkien World-Building Integration

> **Status:** In Progress - #46 Linguistic Engine v1 complete, #48 Spatiotemporal Engine v5 (LLM causal extraction, confidence calibration, location seeding) complete  
> **Milestone:** [Tolkien World-Building](https://github.com/tflynn3/book-graph-analyzer/milestone/2)
> **Issues:** #45-#51
> **Author:** BGA Core
> **Date:** 2026-02-21

---

## 1. Motivation

Book Graph Analyzer already extracts entities, relationships, style fingerprints,
voice profiles, world-bible rules, lore conflicts, and emotional arcs from
Tolkien's texts.  What it does **not** yet model are the deeper world-building
layers that make Tolkien's legendarium unique:

| Gap | Why it matters |
|-----|----------------|
| **Linguistic lineage** | Tolkien invented languages *first*, then derived cultures. Name etymology → cultural identity → lore validation. |
| **Deep genealogy** | The Silmarillion tracks 30+ generations. Genealogy drives inheritance of traits, titles, and artefacts. |
| **Editorial layers** | The same event appears in _Unfinished Tales_, _HoME_, and published Silmarillion with contradictory detail. Provenance matters. |
| **Cultural rule systems** | Cultures have laws, taboos, rites; these constrain valid scene generation far more than generic "world rules". |
| **Cosmological timeline** | Arda's timeline spans Ainulindalë → Fourth Age; temporal anchoring must handle mythic time. |

This RFC maps these five pillars onto the **existing** BGA pipeline and defines
how each integrates - no parallel datastore, no separate pipeline.

---

## 2. The Five Pillars

### 2.1 Linguistic Lineage (Issue #46)

**What:** Track etymology chains - e.g., _Imladris_ (Sindarin) → _Rivendell_ (Common Speech) → _Karningul_ (Westron).

**Integration points:**

| Module | Hook |
|--------|------|
| `models.entities` | New `LinguisticLineage` model attached to any `EntityBase` |
| `extract.resolver` | Alias resolution gains language-tag awareness |
| `graph.writer` | `write_linguistic_lineage()` creates `(:LanguageForm)-[:DERIVED_FROM]->(:LanguageForm)` chains |
| `worldbible.extractor` | Language category extraction feeds lineage data |
| `lore.rules` | Rules can scope to a linguistic register ("Quenya-only term") |

### 2.2 Deep Genealogy (Issue #47)

**What:** Model parent/child/sibling/spouse with generational depth, inheritance of traits, and house membership.

**Integration points:**

| Module | Hook |
|--------|------|
| `models.relationships` | New `GenealogyRelation` model with `generation_depth`, `house`, `inheritance_traits` |
| `extract.relationships` | Genealogy-specific extraction patterns (birth/marriage/death) |
| `graph.writer` | `write_genealogy_batch()` creates typed family edges with generational metadata |
| `lore.checker` | Genealogy constraint validation ("X cannot be Y's ancestor if…") |
| `generate.context` | Family-tree context assembly for character-centric scenes |

### 2.3 Editorial / Textual Layers (Issue #48)

**What:** Track which version of an event comes from which source text and editorial period, enabling provenance-aware lore checking.

**Integration points:**

| Module | Hook |
|--------|------|
| `models` | New `EditorialLayer` metadata model |
| `ingest.loader` | Source-text metadata tagging (published vs. draft vs. HoME) |
| `lore.conflicts` | Conflict resolution gains `source_authority` weight |
| `worldbible.models` | `WorldRule.editorial_layer` field |
| `graph.writer` | `write_editorial_provenance()` links nodes to `(:Source)` |

### 2.4 Cultural Rule Systems (Issue #49)

**What:** Formalize per-culture constraints: Elvish immortality rules, Dwarvish craft-secrecy, Hobbit social customs.

**Integration points:**

| Module | Hook |
|--------|------|
| `worldbible.models` | `CulturalProfile` gains structured `rule_set` field |
| `lore.rules` | Culture-scoped `LoreRule` instances |
| `lore.checker` | Culture-aware validation (e.g., "Elves don't die of old age") |
| `generate.context` | Culture rules injected into scene prompts |

### 2.5 Cosmological Timeline (Issue #50)

**What:** Extend temporal model to cover Ainulindalë, Years of the Lamps/Trees, and mythic "before time" periods.

**Integration points:**

| Module | Hook |
|--------|------|
| `graph.temporal` | Extended `ERA_ORDER` with pre-First-Age eras (already partially done) |
| `graph.writer` | `init_era_chain()` already covers this - extend with sub-age granularity |
| `models.era_reference` | Mythic-era support in `EraReference` |
| `lore.temporal` | Temporal validation for pre-First-Age events |

---

## 3. Pipeline Integration Map

```
┌─────────────────────────────────────────────────────────────┐
│                     Existing Pipeline                       │
│                                                             │
│  ingest ──► extract ──► graph ──► lore ──► generate         │
│    │           │          │        │          │              │
│    │           │          │        │          │              │
│    ▼           ▼          ▼        ▼          ▼              │
│  ┌───┐     ┌─────┐    ┌─────┐  ┌─────┐   ┌──────┐         │
│  │src│     │lang │    │gene-│  │cult │   │world │         │
│  │tag│     │alias│    │alogy│  │rule │   │build │         │
│  │   │     │res. │    │edges│  │valid│   │ctx   │         │
│  └─┬─┘     └──┬──┘    └──┬──┘  └──┬──┘   └──┬───┘         │
│    │           │          │        │          │              │
│    ▼           ▼          ▼        ▼          ▼              │
│  editorial  linguistic  deep     cultural   cosmological    │
│  layer      lineage     geneal.  rules      timeline        │
│  metadata   models      models   system     extension       │
│                                                             │
│  ─────────── New World-Building Layers ──────────────       │
└─────────────────────────────────────────────────────────────┘
```

**Key principle:** Every new capability is an *extension* of an existing module,
not a new top-level package.

---

## 4. Data Model Additions

### New Node Types (Neo4j)

```cypher
// Linguistic lineage
(:LanguageForm {
  id: string,           // "lang_imladris"
  form: string,         // "Imladris"
  language: string,     // "Sindarin"
  entity_id: string,    // FK to Character/Place/Object
  gloss: string         // "Deep dale of the cleft"
})

(:LanguageForm)-[:DERIVED_FROM {
  derivation_type: string  // "translation", "adaptation", "cognate"
}]->(:LanguageForm)

// Editorial provenance
(:Source {
  id: string,              // "src_silmarillion_1977"
  title: string,
  author_period: string,   // "late", "middle", "early"
  publication_year: int,
  editorial_status: string // "published", "draft", "notes"
})

(:Entity)-[:ATTESTED_IN {
  confidence: float,
  page_ref: string
}]->(:Source)
```

### Extended Relationship Properties

```cypher
// Genealogy edges gain:
//   generation_depth: int
//   house: string           // "House of Finwë"
//   inheritance_traits: [string]

// All existing edges gain optional:
//   editorial_source_id: string
```

---

## 5. Implementation Plan

### Phase 1 - Kickoff Slice (this PR)

- [x] RFC document (this file)
- [x] Model stubs: `LinguisticLineage`, `GenealogyRelation`, `EditorialLayer`
- [x] `GraphWriter` extension method stubs
- [x] CLI placeholder commands under existing groups
- [x] Docs updates (ARCHITECTURE.md, DATA_MODEL.md, pipeline.md, nav)
- [x] Tests validating integration touchpoints

### Phase 2 - Linguistic Lineage (Issue #46)

- Implement `LinguisticLineage` extraction from Tolkien texts
- Build `DERIVED_FROM` chain writer
- Integrate with entity resolver for language-aware alias matching
- CLI: `bga worldbible languages`

### Phase 3 - Deep Genealogy (Issue #47)

- Slice 1 (this PR): Sociolinguistic register profile + drift MVP
  - Rule-first classifier (`SociolinguisticRegisterClassifier`)
  - Drift calculator (`detect_register_drift`)
  - Graph persistence/query helpers in `GraphWriter`
  - CLI commands: `bga lore socioreg-profile`, `bga lore socioreg-drift`
- Implement genealogy extraction patterns
- Build generational depth calculator
- Family-tree context assembly for generation
- CLI: `bga lore genealogy`

### Phase 4 - Editorial Layers (Issue #48)

- Source-text metadata tagging in ingest
- Provenance-weighted lore conflict resolution
- CLI: `bga corpus sources`

### Phase 5 - Cultural Rules (Issue #49)

- Structured culture rule extraction
- Culture-scoped lore validation
- CLI: `bga worldbible cultures --rules`

### Phase 6 - Cosmological Timeline (Issue #50)

- Sub-age granularity in era chain
- Mythic-time temporal validation
- CLI: `bga lore timeline --cosmological`

### Cross-cutting: Integration Testing (Issue #51)

- End-to-end pipeline test with all five pillars
- Regression suite ensuring backward compatibility

---

## 6. Non-Goals (Explicit)

- **No new top-level CLI command groups.** Everything lives under `lore`, `worldbible`, `corpus`, or `pipeline`.
- **No separate database.** All data goes into the existing Neo4j graph.
- **No breaking changes.** All additions are backward-compatible.
- **No LLM dependency for core models.** Models are pure Pydantic/dataclass; LLM usage is optional at extraction time.

---

## 7. Open Questions

1. Should `LanguageForm` be a first-class Neo4j node or a property on existing entities? → **Decision: First-class node** (enables cross-entity etymology queries).
2. How deep should genealogy auto-extraction go vs. manual seed data? → Deferred to Issue #47.
3. Should editorial layers affect lore-rule confidence scores? → Yes, via `source_authority` weight (Issue #48).

---

## 8. Progress Log

| Issue | Pillar | Status | Notes |
|-------|--------|--------|-------|
| #45 | Kickoff Slice | ✅ Complete | Models, stubs, CLI placeholders, tests |
| #46 | Linguistic Engine v1 | ✅ Complete | `GraphWriter.write_linguistic_lineage`, JSON parser, CLI `worldbible languages`, 50 tests |
| #47 | Deep Genealogy + Sociolinguistic Registers | 🟡 Slice 1 in progress | Genealogy slice-1 parser/normalization/rules + batch writer/query/CLI, plus sociolinguistic register profile/drift MVP with GraphWriter + CLI integration |
| #48 | Editorial Layers | 🔲 Not started | Stub raises `NotImplementedError` |
| #49 | Cultural Rules | 🔲 Not started | |
| #50 | Cosmological Timeline | 🔲 Not started | |
| #51 | Integration Testing | 🔲 Not started | |

### #46 Remaining TODOs

- [ ] LLM-powered lineage extraction from raw text (currently requires structured JSON input)
- [ ] Integration with `extract.resolver` for language-aware alias matching
- [ ] Batch-optimized Cypher (current impl uses per-lineage transactions)
- [ ] `worldbible.extractor` integration for automatic language category extraction
- [ ] Query helpers: "all names for entity X across languages" as a CLI subcommand

### #50 Slice 1 (Impression-of-Depth Engine) Update

Implemented in this slice:
- Added first-class lore-depth models (`LoreArtifact`, `BrokenReference`) in `models/lore_depth.py`
- Added extraction helper `lore.depth.extract_lore_depth(...)` for artifact-like mentions and unresolved markers
- Added GraphWriter persistence/query helpers for lore depth nodes and unresolved-reference reporting
- Added CLI commands under existing families:
  - `worldbible artifacts`
  - `lore unresolved-refs`
- Added tests in `tests/test_issue_50_lore_depth_slice1.py`

Remaining for Issue #50:
- stronger context-aware extraction and disambiguation
- resolver-backed candidate linking for broken references
- richer provenance/conflict weighting across editorial layers
- generation-time consumption of unresolved reference queues
