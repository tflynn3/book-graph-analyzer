# Beat Symmetry Master Plan (Extraction ↔ Generation)

## Why this exists

Current generation can produce structurally valid beats, but semantic grounding is still weak (e.g., era/book leakage and `Unknown` participants). The core fix is **symmetry**:

- Extract beats from source prose into a rigorous beat graph
- Learn statistics from those extracted beats
- Generate beats from those same statistics/schema
- Validate generated beats against the same constraints used in extraction

In short: one ontology, one stats layer, two directions.

---

## Immediate problems this fixes

1. **Book/era leakage** (Bilbo in First Age contexts)
2. **Weak participant grounding** (`Unknown` placeholders)
3. **Template drift** (generated beats not matching extracted narrative structure)
4. **Non-comparable quality** (no round-trip metric between extraction and generation)

---

## End-to-end symmetric pipeline

`prose -> beat extraction -> beat graph + stats -> beat generation -> prose realization -> re-extraction -> round-trip comparison`

### 1) Extraction side
- Input: source prose (book-scoped)
- Output: `UnifiedBeat[]` + edges (`CAUSES`, `RESOLVES`, `REFERENCES`, `INVOLVES`)
- Persist to graph + artifact files

### 2) Statistics side
- Build per-book/per-era statistical profiles from extracted beats
- Export compact generation profiles (no raw prose required)

### 3) Generation side
- Consume profile + constraints to produce `UnifiedBeat[]`
- Enforce causal and grounding constraints in-beam
- Realize prose from beat sequence

### 4) Validation side
- Run hard structural checks + hard lore gates + soft style/stat checks
- Re-extract generated prose and compare profile drift

---

## UnifiedBeat schema (single source of truth)

Each beat should be valid for both extraction and generation (`origin: extraction|generation|template`).

Required core fields:
- `beat_id`, `story_id`, `chapter`, `scene_id`, `position`
- `beat_class` (enum, e.g. `introduction`, `reintroduction`, `escalation`, `reveal`, `payoff`, `reversal`, `crisis`, `resolution`)
- `action`
- `participants[]` (resolved canonical ids + surface forms)
- `motifs[]`
- `preconditions[]`, `effects[]`
- `cause_refs[]`, `resolves_beat_ids[]`
- `source_book`, `source_event_id`, `source_span` (for extracted beats)
- `source_canon_node_ids[]`
- `tension_delta`, `tension_absolute`
- `style_register_hints`
- `grounding_confidence`
- `scoring_breakdown`

Conditional invariants:
- `payoff` must have `resolves_beat_ids`
- `foreshadowing` must include a seed claim
- `reintroduction` must include gap metadata (`reintroduced_after_n_scenes`)

---

## Node/edge model

Primary nodes:
- `(:UnifiedBeat)`
- `(:UnifiedBeatSheet)`
- `(:BeatStats)`

Primary edges:
- `(:UnifiedBeat)-[:CAUSES]->(:UnifiedBeat)`
- `(:UnifiedBeat)-[:RESOLVES]->(:UnifiedBeat)`
- `(:UnifiedBeat)-[:REFERENCES]->(:CanonNode)`
- `(:UnifiedBeat)-[:INVOLVES]->(:Character)`
- `(:UnifiedBeat)-[:OCCURS_AT]->(:Scene)`
- `(:UnifiedBeat)-[:ANCHORED_TO]->(:SourcePassage)`

v1 representation strategy:
- Keep lore/style embedded on beat payload for speed
- Optionally materialize style/lore edges later for analytics

---

## Statistics required for generation

Per-book / per-era profiles:
1. Beat class transition matrices (1st and 2nd order)
2. Position-conditioned beat priors (early/mid/late scene)
3. Character intro/reintro rates
4. Reintroduction gap distribution
5. Tension curve profile (mean + variance by normalized position)
6. Causal depth/width distributions
7. Motif recurrence and spacing distributions
8. Style/register distribution and prose budget CV
9. Cross-book edge and era-compatibility priors

Recommended additions:
- intro dynamics model (Hawkes-style self-/cross-excitation)
- cross-book adjacency matrix by source/target book

---

## Generation algorithm (deterministic-first)

1. Allocate beats by act/scene using profile priors and constraints
2. Sample beat classes position-conditionally
3. Sample participants/motifs/actions with hard canonical filters
4. Build causal DAG with precondition/effect simulation
5. Early-stop once scene goal complete (with min-beat floor)
6. Score variable-length paths with anti-padding penalty
7. Emit `UnifiedBeat[]` and only then realize prose

No raw source prose is required for generation inputs — only profiles and constraints.

---

## Hard gates (must-fail)

1. **Canonical resolution gate**
   - Every participant/action anchor must resolve to valid canon node(s)
2. **Era gate**
   - Beat entities/actions must be era-compatible with project profile
3. **Book-scope gate**
   - Cross-book edges must stay under configured budget
4. **Causal gate**
   - Preconditions satisfied; no impossible transitions
5. **Traceability gate**
   - Every generated beat has provenance path to profile + constraints
6. **Constraint gate**
   - required/forbidden term checks and scenario constraints

Soft gates (warn):
- style mismatch
- statistical drift
- novelty over/under-shoot

---

## Round-trip fidelity target

Symmetry criterion:

`stats(extract(realize(generate(stats_target)))) ≈ stats_target`

Track with:
- Wasserstein / KS / χ² drift metrics across key distributions
- per-metric thresholds in CI

---

## CLI/API additions

- `bga beat extract --book <id> --out beats.jsonl`
- `bga beat stats --input beats.jsonl --out beat_stats.json`
- `bga beat generate --stats beat_stats.json --constraints constraints.json --out generated_beats.json`
- `bga beat validate --beats generated_beats.json --strict-canon --strict-era --strict-crossbook`
- `bga beat diff --a stats_a.json --b stats_b.json`
- `bga story generate --beat-file generated_beats.json`

---

## Rollout plan (small PR slices)

1. **PR-1:** `UnifiedBeat` model + validators + schema versioning
2. **PR-2:** Extraction adapter (`prose -> UnifiedBeat[]`) + fixtures
3. **PR-3:** `BeatStatAggregator` + stats artifacts
4. **PR-4:** Generator consumes stats profile (deterministic mode)
5. **PR-5:** Hard canonical/era/crossbook gates in validate
6. **PR-6:** Story integration (`story generate --beat-file`)
7. **PR-7:** Round-trip diff command + drift thresholds in CI
8. **PR-8:** Migration of legacy beat artifacts and deprecation flags

---

## Recommended defaults for Tolkien safety

```json
{
  "grounding": {
    "allowed_books": ["silmarillion"],
    "allowed_eras": ["First Age"],
    "max_crossbook_edge_rate": 0.05,
    "max_unknown_participant_rate": 0.0,
    "require_resolved_participants": true
  }
}
```

---

## Decision summary

- Yes, symmetry is the right architecture.
- Beats must be first-class extracted entities, not only generated templates.
- Character introduction/reintroduction is a required beat feature (schema + stats + generation logic).
- Lore/style should be enforced in validators, not just attached as metadata.
- Success is measured by hard grounding gates plus round-trip statistical fidelity.
