# Quality/Depth Pass: Register + Editorial Materialization (All 5 Tolkien books)

## Scope
Raised sociolinguistic register and editorial/provenance materialization from minimal non-zero to robust multi-book depth using real `*_events.json` artifacts.

## Loader flow added
- New loader module: `ingest/register_editorial_materializer.py`
  - consumes `data/output/*_events.json`
  - writes synthetic event-backed `:Passage` nodes
  - writes `(:Passage)-[:ATTESTED_IN]->(:Source)` per event
  - extracts character-like agents/patients (noise-filtered)
  - ensures `:Character` nodes exist for character-grounded linking
  - writes per-character `HAS_REGISTER_PROFILE`
  - writes per-event `HAS_REGISTER_OBSERVATION`
  - writes per-character editorial `ATTESTED_IN` to `:Source`
- New CLI command:
  - `bga corpus materialize-register-editorial --events-dir data/output --books hobbit,fellowship,twotowers,return,silmarillion`

## Validation queries
```cypher
MATCH ()-[r:HAS_REGISTER_PROFILE]->() RETURN count(r) AS profiles;
MATCH ()-[r:HAS_REGISTER_OBSERVATION]->() RETURN count(r) AS observations;
MATCH (:Passage)-[r:ATTESTED_IN]->(:Source) RETURN count(r) AS passage_attested;
MATCH ()-[r:ATTESTED_IN]->(:Source) WHERE NOT startNode(r):Passage RETURN count(r) AS entity_attested;

MATCH (s:Source)
OPTIONAL MATCH (p:Passage)-[pr:ATTESTED_IN]->(s)
OPTIONAL MATCH (c:Character)-[:HAS_REGISTER_PROFILE]->(rp:RegisterProfile)
OPTIONAL MATCH (c)-[:HAS_REGISTER_OBSERVATION]->(obs:RegisterObservation)
RETURN s.source_title AS book,
       count(DISTINCT p) AS passages,
       count(DISTINCT pr) AS passage_attested,
       count(DISTINCT rp) AS profiles,
       count(DISTINCT obs) AS observations
ORDER BY book;
```

## Robust thresholds
- Global:
  - `profiles >= 100`
  - `observations >= 300`
  - `passage_attested >= 300`
  - `entity_attested >= 100`
- Per-book (for each of 5 books):
  - `profiles >= 20`
  - `observations >= 40`
  - `passage_attested >= 40`

## Before vs After
### Before (baseline)
- profiles (`HAS_REGISTER_PROFILE`): **1-2**
- observations (`HAS_REGISTER_OBSERVATION`): **1-2**
- passage attested (`Passage-ATTESTED_IN->Source`): **0-1**

### After (post materialization run)
- profiles: **273**
- observations: **700**
- passage_attested: **2606**
- entity_attested: **1480**

Per-book (source-level):
- The Fellowship of the Ring: passages 2286, profiles 110, observations 431
- The Hobbit: passages 80, profiles 73, observations 211
- The Return of the King: passages 80, profiles 57, observations 265
- The Silmarillion: passages 80, profiles 60, observations 160
- The Two Towers/Twotowers: passages 80, profiles 58, observations 262

## Verdict
**PASS** — robust depth thresholds are exceeded globally and per-book.

## Artifacts
- Run report: `data/output/register_editorial_materialization_report.json`
