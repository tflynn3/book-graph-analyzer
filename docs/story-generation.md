# Story Generation

## Single scene

```bash
bga generate scene -g "Bilbo and Gandalf argue about risk" -c Bilbo -c Gandalf -p "Bag End" -w data/worldbible/hobbit_ch1_bible.json
```

## Outline

```bash
bga generate outline --help
```

## Full novel

```bash
bga generate novel --outline data/output/outline.json --checkpoint data/checkpoints --resume --world-bible data/worldbible/hobbit_ch1_bible.json --output data/output/novel_draft.md
```

## Shadow-graph chapter drafting

```bash
bga story draft --project mithrandir-east --chapter 1 --grounded
```

This path now uses the same `SceneGenerator` stack as `generate novel`, but drives it from `shadow_solution.json` and related story artifacts instead of a standalone outline file.
It also honors `project.json.timeline`: past references are allowed when the project permits them, while later-era character leakage is treated as a hard failure during draft/audit.

## Local Tolkien chapter demo

The checked-in Beren/Luthien sample can produce a full local Chapter 1 without Neo4j or a paid LLM by using the deterministic template renderer:

```bash
bga story context --project beren-luthien-expanded --graph-stats
bga story grow-shadow --project beren-luthien-expanded --auto
bga story solve --project beren-luthien-expanded
bga story beats expand --project beren-luthien-expanded
bga story draft --project beren-luthien-expanded --chapter 1 --grounded --renderer template
bga story audit --project beren-luthien-expanded --chapter 1
```

Generated artifacts are written under `data/projects/beren-luthien-expanded/`. The audit fails on missing required anchors, placeholder prose, forbidden terms, empty or unresolvable canon-evidence refs, too-short template scenes, low dialogue ratio, too-short chapters, overlong average sentence length, or wrong-era Tolkien names.
The grounded draft command also runs the configured hard quality gates before writing final chapter artifacts, so the chapter demo cannot silently produce a too-short or statistically off-register draft.

The template renderer is a deterministic regression baseline, not a quality evaluator. Its scene score fields remain zero/unverified and its scenes are flagged for review. Use the LLM renderer plus a populated story bible and voice profiles for evidence-backed lore and voice passes.
