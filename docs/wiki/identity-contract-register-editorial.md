# Identity Contract: Register + Editorial Materialization

## Root cause
Register/editorial writes assumed direct `MATCH (e {id: $entity_id})` and exact passage/source IDs.
In mixed corpora, nodes may be keyed via `canonical_id` / `canonical_name` or have casing/alias variation, so joins silently miss.

## Contract (writer resolution order)
For entity references used by socioreg/editorial writes:
1. `id` (exact)
2. `canonical_id`
3. `canonical_name` (case-insensitive)
4. `aliases` (case-insensitive)

Ambiguous top-score matches are treated as **non-writable** (fail closed).

For passage references:
1. exact `Passage.id`
2. lowercase exact
3. containment fallback (guarded; ambiguous ties rejected)

## Schema support
Added/ensured:
- `CONSTRAINT register_profile_entity` on `:RegisterProfile(entity_id)` unique
- `INDEX source_title` on `:Source(source_title)`
- `INDEX register_obs_entity_time` on `:RegisterObservation(entity_id, observed_at)`

## Backfill
Dry run:

```bash
python scripts/backfill_register_editorial_materialization.py --dry-run
```

Apply:

```bash
python scripts/backfill_register_editorial_materialization.py --apply
```

## Validation queries

```cypher
MATCH ()-[r:HAS_REGISTER_PROFILE]->() RETURN count(r) AS profiles;
MATCH ()-[r:HAS_REGISTER_OBSERVATION]->() RETURN count(r) AS observations;
MATCH (:Passage)-[r:ATTESTED_IN]->(:Source) RETURN count(r) AS passage_attested;
```
