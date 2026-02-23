# Shadow Sampler / Scorer / Selector v1

This page documents the **graph-native shadow candidate pipeline** for grounded story drafting.

## Commands

### 1) Sample candidate shadow graphs

```bash
bga story sample-shadow \
  --project <slug> \
  --n <int> \
  --method anneal \
  [--seed <int>] \
  [--steps <int>] \
  [--temp-start <float>] \
  [--temp-end <float>]
```

- Generates many candidate trajectories by local graph mutations.
- Uses a Metropolis/annealing acceptance rule.
- Deterministic when `--seed` is provided.
- Output:
  - `data/projects/<slug>/shadow_samples.jsonl`

Each JSONL row contains:
- `candidate_id`
- `seed`
- `anneal_energy`
- `acceptance_ratio`
- `state` (scene-by-scene shadow events)

---

### 2) Score sampled candidates

```bash
bga story score-shadow \
  --project <slug> \
  [--weights '{"canon_consistency":0.25,...}'] \
  [--pareto]
```

Scores each sampled candidate with explicit components:

- `canon_consistency_penalty` (lower is better)
- `canon_consistency` (derived, higher is better)
- `transition_likelihood`
- `arc_coherence`
- `style_register`
- `novelty_diversity`

Outputs:
- `data/projects/<slug>/shadow_scores.json`
- `data/projects/<slug>/shadow_pareto_front.json` (when `--pareto`)

Default weighted objective:
- canon consistency: 0.25
- transition likelihood: 0.25
- arc coherence: 0.20
- style/register: 0.15
- novelty/diversity: 0.15

---

### 3) Select top-K

```bash
bga story select-shadow --project <slug> --top <k>
```

- Selects highest weighted-score candidates.
- Stable ordering rule: `weighted_score DESC`, then `candidate_id ASC`.
- Output:
  - `data/projects/<slug>/shadow_selected.json`

## Interpretation guide

- **High canon consistency + high transition likelihood**: safer, canon-adjacent candidates.
- **High novelty/diversity**: broader exploration, useful for creative branches.
- **Low style/register**: likely off-budget for scene length/register targets.
- **Pareto front**: useful when you want trade-off options (not only a single weighted winner).

## Typical run sequence

```bash
bga story context --project <slug> --graph-stats
bga story sample-shadow --project <slug> --n 500 --method anneal --seed 42
bga story score-shadow --project <slug> --pareto
bga story select-shadow --project <slug> --top 10
```