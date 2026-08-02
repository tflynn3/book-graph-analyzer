# Story Workflow CLI (Product UX)

The `story` command group provides a friendly workflow for project setup, planning, validation, and graph-grounded chapter synthesis.

## 1) Initialize a project

Interactive wizard:

```bash
bga story init
```

Non-interactive (CI-friendly):

```bash
bga story init \
  --name "Mithrandir in the East" \
  --slug mithrandir-east \
  --premise "A covert mission beyond Rhun." \
  --genre fantasy \
  --story-era "Third Age" \
  --story-year 3018 \
  --target-chapters 8 \
  --scenes-per-chapter 3 \
  --non-interactive
```

Creates scaffold under:

- `data/projects/<slug>/project.json`
- `data/projects/<slug>/constraints.json`
- `data/projects/<slug>/story_bible.md`
- `data/projects/<slug>/plan.json` (placeholder)

`project.json` now carries a `timeline` block. This is the hard story-time scope used by
`context`, `grow-shadow`, `solve`, `draft`, and `audit`.

## 2) Auto-generate a plan

```bash
bga story plan --project mithrandir-east --auto
```

Produces chapter + scene artifact:

- `data/projects/<slug>/plan.json`

## 3) Validate continuity/style/canon checks

```bash
bga story validate --project mithrandir-east
```

Outputs:

- Human-readable: `data/projects/<slug>/validation_report.md`
- JSON artifact: `data/projects/<slug>/validation_report.json`

You can also override JSON output path:

```bash
bga story validate --project mithrandir-east --json-out artifacts/mithrandir-validate.json
```

## 4) Build graph-native context stats

```bash
bga story context --project mithrandir-east --graph-stats
```

Generates:

- `data/projects/<slug>/context_stats.json`

Stats include:

- event transition probabilities
- motif/reference density priors
- character participation priors
- per-entity temporal presence (`entity_temporal_presence`)
- story-time guardrails (`timeline.future_guardrail_entities`)
- hybrid local canon neighborhood (`local_story_neighborhood`) built from project seed entities
- resolvable event evidence (`canon_evidence`) with source file, event ID, book/location provenance, era, and year
- register/style budgets

Events known to occur after `project.json.timeline` are excluded from these priors. Unknown dates remain unknown; the context builder does not synthesize a year.

## 5) Grow probabilistic shadow graph

```bash
bga story grow-shadow --project mithrandir-east --auto
```

Generates:

- `data/projects/<slug>/shadow_graph.json`
- `data/projects/<slug>/shadow_candidates.json`

Statistical engine v1 hardening:

- Deterministic seed uses a stable SHA-256 hash of `(project_slug, plan, constraints)`.
- Candidate sampling scales to large pools (default target: `max(500, scenes*24)`), overridable by `constraints.search.target_candidates`.
- `shadow_candidates.json` includes `seed`, `sampling`, per-candidate interpretable `score_components` + `score_total`, and `elites_grid` behavior-cell winners.
- Candidate cast selection now hard-filters temporally invalid present actors.
  Example: a First Age project may remember older lore, but Bilbo/Frodo/Gandalf are rejected as active scene participants.
- Candidate selection now prefers the project's local canon neighborhood over corpus-global popularity.
  The command first looks for matching seed entities in Neo4j propositions/passages, then falls back to matching event artifacts when the live graph is incomplete.
- `CanonEvidence` nodes are materialized in `shadow_graph.json`; candidate and beat `source_canon_node_ids` must resolve to these records. Character and motif names alone are not treated as evidence.

## 6) Solve best valid trajectory

```bash
bga story solve --project mithrandir-east
```

Generates:

- `data/projects/<slug>/shadow_solution.json`

Solver now re-checks temporal validity before accepting a candidate path, so stale or hand-edited
candidate artifacts cannot bypass story-time gating.

When `sample-shadow`, `score-shadow`, and `select-shadow` artifacts exist, the selected whole-story samples are converted into scene priors and contribute to beam solving. The sampling branch is therefore advisory but no longer disconnected from `solve`.

## 6.5) Expand scene beats (deterministic/template)

```bash
bga story beats expand --project mithrandir-east --method template
```

Generates:

- `data/projects/<slug>/shadow_beats.json`
- `data/projects/<slug>/shadow_beats_selected_sidecar.json` (when `shadow_selected.json` exists)

Notes:

- Offline-first (no LLM required).
- Stable deterministic IDs + seed for repeatable output.
- Validation includes `cause_ref_issues` and per-beat `failed_constraints[]`.

Validate beats (optionally scoped):

```bash
bga story beats validate --project mithrandir-east --chapter 1
bga story beats validate --project mithrandir-east --scene ch01-sc02 --strict
bga story beats validate --project mithrandir-east --strict --strict-warnings
```

Outputs:

- `data/projects/<slug>/shadow_beats_validation.json` (or `--json-out <path>`)

Show beat summary for a scope:

```bash
bga story beats show --project mithrandir-east --chapter 1
bga story beats show --project mithrandir-east --scene ch01-sc02
```

Summary includes count, beat ids, beat types, and top issues.

Clean beat artifacts safely:

```bash
bga story beats clean --project mithrandir-east --dry-run
bga story beats clean --project mithrandir-east --chapter 1
```

- With scope flags (`--chapter` / `--scene`), `clean` removes only matching beats from `shadow_beats.json`.
- Without scope flags, `clean` removes beat artifact files (`shadow_beats.json`, sidecar, and beat validation report).

## 7) Draft grounded chapter prose

```bash
bga story draft --project mithrandir-east --chapter 1 --grounded
```

For reproducible local chapter generation without a live LLM or Neo4j writer, use the deterministic template renderer:

```bash
bga story draft --project beren-luthien-expanded --chapter 1 --grounded --renderer template
```

Generates:

- `data/projects/<slug>/chapter_01.md`
- `data/projects/<slug>/chapter_01_trace.json`
- `data/projects/<slug>/chapter_01_draft.json`

Behavior:

- Uses the real `generate.SceneGenerator` stack to draft per-scene prose from the solved shadow trajectory.
- With `--renderer template`, emits deterministic grounded prose through the same chapter/trace/audit artifacts and skips graph writes.
- Loads the project `story_bible.md` as structured manual constraints (asserted bullets under World Rules, Continuity Rules, Geography, Culture, and related rule headings).
- A project may set `voice_profiles_file` in `project.json`; otherwise the checked-in Hobbit profile artifact is used when available. Only matching speakers are patched/scored.
- Neo4j event retrieval is bounded by the project era/year, and source IDs plus source book/location are retained in the retrieved evidence.
- Template-rendered scenes are marked `FLAGGED` with zero/unverified model scores. Passing prose-quality numbers are never fabricated for the deterministic renderer.
- Runs configured hard quality gates during construction for minimum scene/chapter length, minimum dialogue share, minimum type-token ratio, and maximum average sentence length.
- Passes `constraints.json.quality.target_scene_words` to the LLM as an explicit approximate length request. A scene's `target_words`, or its share of a chapter-level `target_words`, takes precedence when present in `plan.json`.
- Supports chapter-specific quality overrides through `constraints.json.quality_by_chapter`. Keys are chapter numbers as strings and may override any supported `quality` setting; this is useful for intentionally quiet or dialogue-heavy chapters without weakening the project-wide gate.
- Directs the LLM to express reasoning through observed signs, memory, disagreement, and decisions; to avoid modern analytical/process diction; and to treat canon evidence as a boundary rather than a source scene to reconstruct.
- Persists deterministic scene IDs (`<project-slug>-<scene-id>`) so reruns update the same shadow-state / generation nodes instead of duplicating them.
- Seeds chapter outline metadata into the generation graph before drafting so later scenes can retrieve active plot-thread context.
- Injects story-time guidance into the scene goal. Past figures/events may be referenced when allowed by project timeline, but later-era names are explicitly forbidden.

Required-term enforcement behavior:

- Reads `constraints.json.required_elements` as required terms.
- Uses token-boundary phrase matching (not naive substring checks).
- Supports aliases via `constraints.json.required_element_aliases`.
- Regenerates up to `constraints.json.enforcement.max_retries` (default `2`) when required terms are missing.
- Fails clearly if terms are still missing after retries.

Tradeoffs:

- Improves deterministic canon anchor coverage.
- May reduce prose naturalness when required terms are very long/rigid.
- Retry loops increase generation latency when misses occur.

## 8) Audit chapter grounding and constraints

```bash
bga story audit --project mithrandir-east --chapter 1
```

Use strict mode explicitly when desired:

```bash
bga story audit --project mithrandir-east --chapter 1 --enforce-required-terms
```

When enforcement is enabled, missing required terms are treated as errors (`status=fail`).

Grounding hardening checks:

- Audit reports per-scene required-term coverage (`constraints.required_scene_coverage`).
- Semantic evidence alignment checks verify trace excerpts align with chapter text.
- Trace sections must carry non-empty `source_canon_node_ids`.
- Every canon source ref must resolve to a node in `shadow_graph.json`; empty or invented refs fail the audit.
- Quality checks fail placeholder prose, too-short scenes/chapters when `constraints.quality.min_scene_words` or `constraints.quality.min_chapter_words` are set, low dialogue share when `constraints.quality.min_dialogue_ratio` is set, overlong average sentence length when `constraints.quality.max_avg_sentence_words` is set, and out-of-domain Tolkien names when `constraints.quality.forbid_out_of_domain_entities` is enabled.
- Audit now reports `temporal_alignment` separately:
  - `past_references`: older-era names mentioned in a historically valid way
  - `future_mentions`: later-era contamination that fails the chapter

Generates:

- `data/projects/<slug>/chapter_01_audit.json`
- `data/projects/<slug>/chapter_01_audit.md`

## Notes

- Basic flow requires no manual JSON editing.
- Canon integration is read-only in this iteration (uses configured canon file when present).
- `story plan` currently supports `--auto` mode only.
- `story grow-shadow` currently supports `--auto` mode only.
- Install semantic retrieval with `pip install -e '.[embeddings]'`. The extra contains Chroma, DuckDB, and Sentence Transformers; its integration tests are skipped in a base installation.

## Draft doctor strict gate

`bga draft doctor --strict` blocks both high- and medium-severity findings in its structural, repetition, causality, register, voice, and ending-cadence categories. Its register checks include targeted editorial leaks and modern process language such as `stopping rule`, `controlled risk`, and `provenance`. Low-severity findings remain revision guidance. A large set of material medium findings can no longer produce a strict PASS.
