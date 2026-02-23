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

2. **Generate beat candidates per slot**
   - Deterministic seeded sampling from:
     - scene action + motifs
     - participant priors
     - canon-linked action templates
   - keep 2–4 candidates per slot for tractable selection

3. **Score each beat candidate**
   - weighted score with hard penalties for forbidden terms / broken grounding

4. **Select beat path**
   - beam search across slots
   - objective favors coherence + grounding + diversity
   - enforce required terms/aliases at beat-sequence level

5. **Emit artifacts**
   - save all candidates + selected sequence

### Initial score formula

`total = 0.30*canon_grounding + 0.25*causal_coherence + 0.20*scene_goal_progress + 0.10*style_fit + 0.10*novelty - 0.15*constraint_penalty`

Notes:
- weights configurable in `constraints.json.story.beat_scoring`
- clamp `[0,1]` for components

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
   - beats per scene in configured range (default 8–24)

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
      "candidate_beats_per_slot": 3,
      "beam_width": 8
    },
    "beat_scoring": {
      "canon_grounding": 0.30,
      "causal_coherence": 0.25,
      "scene_goal_progress": 0.20,
      "style_fit": 0.10,
      "novelty": 0.10,
      "constraint_penalty": 0.15
    }
  }
}
```

---

## Implementation Plan (fast, low-risk)

### Phase 1 (ship quickly)

- Add data schemas + writers for beat artifacts
- Implement deterministic beat slot generation + scoring
- Implement beat selection beam search
- Add `--from-beats` drafting path using template prose
- Add tests (unit + focused e2e)

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
