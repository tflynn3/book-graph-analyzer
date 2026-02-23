# Scene → Beat Expansion Design (v1)

## Problem

Current `story` generation selects roughly one shadow event per scene. That is too coarse for prose:

- Real narrative density is often many events per scene (sometimes dozens+)
- Current draft output reads templated and under-detailed
- Grounding and audit are scene-level, not beat-level

## Goal

Add a **beat expansion layer** between scene-level planning and prose drafting.

Keep current scene-level planning as-is, then expand each selected scene candidate into a sequence of micro-events (beats), score/select them, and draft from beats.

## Non-Goals (v1)

- Full autonomous long-form literary quality optimization
- Replacing the current shadow planner
- Mandatory LLM prose generation in v1 (optional integration point only)

---

## Proposed Pipeline

Current:
`context -> grow-shadow -> solve -> draft -> audit`

Proposed:
`context -> grow-shadow -> solve -> expand-beats -> select-beats -> draft-from-beats -> audit`

### New commands

- `bga story expand-beats --project <slug> [--chapter N] [--target-beats-per-scene K]`
- `bga story select-beats --project <slug> [--chapter N] [--beam-width B]`
- `bga story draft --project <slug> --chapter N --grounded [--from-beats]`
  - default to `--from-beats` when beat artifacts exist

---

## Data Contracts

### 1) `shadow_beats.json`

Top-level:

- `schema_version: "shadow-beats-v1"`
- `project_slug`
- `generated_at`
- `source_solution_sha` (optional provenance)
- `scenes: []`

Per scene:

- `scene_id`
- `source_candidate_id`
- `source_shadow_event_id`
- `beat_candidates: []`

Beat candidate fields:

- `beat_id` (`<scene_id>-beat-<n>-cand-<m>`)
- `scene_id`
- `position` (1..N)
- `description` (atomic action/state change)
- `action`
- `participants` (entity ids/names)
- `motifs`
- `cause_refs` (prior beat ids)
- `effect_refs` (future beat hints)
- `source_canon_node_ids` (for traceability)
- `scores`:
  - `canon_grounding`
  - `causal_coherence`
  - `scene_goal_progress`
  - `novelty`
  - `style_fit`
  - `constraint_penalty`
  - `total`
- `hard_constraints_ok`

### 2) `shadow_beats_selected.json`

- `schema_version: "shadow-beat-selection-v1"`
- `project_slug`
- `chapter`
- `selected_beats: []` (ordered sequence)
- `selection_objective`
- `best_score`

### 2.5) Failure-safe state and eventual unified trace

For v1, separate `shadow_beats.json` and `shadow_beats_selected.json` keeps implementation simple.

To reduce desync risk when a job fails mid-pipeline:

- write `shadow_beats_selected.json.tmp` and atomically rename to `shadow_beats_selected.json` only on success
- include `source_shadow_beats_hash` in `shadow_beats_selected.json`
- on load, verify the hash against current `shadow_beats.json`; fail fast if mismatch

Future direction (v2): fold both into a single append-only trace artifact (`story_trace.jsonl` or unified `story_trace.json`) to make partial writes and replay easier.

### 3) `chapter_XX_trace.json` extension

Add optional beat-level fields per section:

- `beat_id`
- `source_beat_ids` (if multiple beats mapped to one paragraph)
- `source_canon_node_ids` (already present, continue using)

---

## Beat Expansion Algorithm (v1)

For each selected scene candidate from `shadow_solution.json`:

1. **Seed beat skeleton**
   - Build `K` slots (default 12, configurable)
   - Ensure coverage of: setup, escalation, turn, consequence

2. **Generate beat candidates per slot (autoregressive with beam context)**
   - Deterministic seeded sampling from:
     - scene action + motifs
     - participant priors
     - canon-linked action templates
   - keep 2–4 candidates per slot for tractable selection
   - slot `N+1` candidate generation receives context from each surviving beam path through slot `N` (for causal coherence and valid `cause_refs`)

3. **Score each beat candidate**
   - weighted score with hard penalties for forbidden terms / broken grounding

4. **Select beat path**
   - beam search across slots
   - objective favors coherence + grounding + diversity
   - enforce required terms/aliases at beat-sequence level

5. **Emit artifacts**
   - save all candidates + selected sequence

### Initial score formula

`total = 0.3158*canon_grounding + 0.2632*causal_coherence + 0.2105*scene_goal_progress + 0.1053*style_fit + 0.1053*novelty - 0.15*constraint_penalty`

Notes:
- weights configurable in `constraints.json.story.beat_scoring`
- clamp `[0,1]` for components
- positive weights are normalized to 1.0 for easier reasoning

---

## Drafting from Beats

`story draft --from-beats`:

- consume ordered selected beats
- map 1–3 beats per paragraph
- preserve explicit trace links (`paragraph -> beat_id(s) -> canon ids`)
- if beat artifacts missing, fallback to current scene-level draft behavior

Optional later:
- LLM rewrite pass over deterministic beat scaffold (guarded by trace lock)

---

## Audit & Acceptance Gates

Extend `story audit` with beat-aware checks:

1. **Beat coverage**
   - expected beats vs traced beats ratio >= 0.95

2. **Causal integrity**
   - no orphan beats (`cause_refs` unresolved)

3. **Grounding integrity**
   - beat paragraphs must retain `source_canon_node_ids`

4. **Constraint enforcement**
   - required terms/aliases satisfied in chapter text and beat sequence
   - forbidden terms absent

5. **Density sanity**
   - beats per scene in configured range (default dynamic range, see below)

### v1 acceptance criteria

- For reference Beren/Lúthien test chapter:
  - >= 8 selected beats for chapter 1 with 2 scenes
  - audit `status=pass`
  - required terms pass
  - no invalid trace refs
- deterministic rerun with same inputs produces same selected beat ids

---

## CLI / Config

### `constraints.json` additions (optional)

```json
{
  "story": {
    "beat_expansion": {
      "enabled": true,
      "target_beats_per_scene": 12,
      "min_beats_per_scene": 8,
      "max_beats_per_scene": 24,
      "dynamic_budget": {
        "enabled": true,
        "base": 8,
        "participants_weight": 1.0,
        "motifs_weight": 0.5,
        "scene_complexity_weight": 1.5
      },
      "candidate_beats_per_slot": 3,
      "beam_width": 8
    },
    "beat_scoring": {
      "canon_grounding": 0.3158,
      "causal_coherence": 0.2632,
      "scene_goal_progress": 0.2105,
      "style_fit": 0.1053,
      "novelty": 0.1053,
      "constraint_penalty": 0.15
    }
  }
}
```

Dynamic budget suggestion (v1):

`beats_target = clamp(min,max, base + participants_weight*unique_participants + motifs_weight*unique_motifs + scene_complexity_weight*scene_complexity_score)`

---

## Implementation Plan (fast, low-risk)

### Phase 1 (ship quickly)

- Add data schemas + writers for beat artifacts
- Implement deterministic beat slot generation + scoring
- Implement beat selection beam search
- Add `--from-beats` drafting path using template prose
- Add tests (unit + focused e2e)
- Add explicit constraint-debug fields on beats:
  - `failed_constraints: string[]` (replacing/augmenting boolean-only `hard_constraints_ok`)
  - validation rule: every `cause_refs` target must have lower `position` than current beat

### Phase 2

- Better causal templates / action ontology
- stronger motif sanitization (remove stopword motifs like `are`, `for`)
- adaptive beats-per-scene by scene complexity

### Phase 3 (optional)

- Optional LLM prose realization from beat plans
- style/post-edit controls with strict trace preservation

---

## Test Strategy

1. Unit:
- beat scoring math
- deterministic seed behavior
- constraint enforcement at beat-level

2. Integration:
- expand+select over small synthetic scene
- expected artifact schema validation

3. E2E:
- Beren/Lúthien mini pipeline with `--from-beats`
- audit pass + density threshold

---

## Risks & Mitigations

1. **Mode collapse / repetitive beats**
   - add repetition penalty + novelty bonus

2. **Nonsense motif leakage**
   - motif normalization + stopword filter before beat generation

3. **Compute blow-up**
   - cap candidates per slot, bounded beam width, chapter-scoped execution

4. **Traceability regressions**
   - hard requirement: every selected beat has source references

---

## Suggested PR slicing

1. PR A: schemas + artifact IO + docs skeleton
2. PR B: expand-beats command + scoring + unit tests
3. PR C: select-beats + draft-from-beats + audit updates + e2e

This keeps reviewable chunks small while delivering usable value quickly.
