# Shadow-Graph Statistical Synthesizer — Research Brief (MVP v1)

Date: 2026-02-22

## Scope
Researched practical methods for:
1. Narrative planning on knowledge/scene graphs
2. Constrained generation with symbolic planners + LLM realization
3. Probabilistic graph growth/search on event graphs
4. Style transfer controls and budgeted motifs/references

## Key findings (applied)

### 1) Narrative planning on graphs
- Classical narrative planning (IPOCL) emphasizes **causal coherence + character intentionality**, not just event ordering.
- Recent KG-assisted storytelling work shows a practical loop: **initialize KG → generate scene conditioned on KG → update KG**, and users can edit KG for control.

**Implication for this repo**
- Keep a graph-first intermediate (`shadow_graph.json`) with typed nodes (`ShadowScene`, `ShadowEvent`, `ShadowMotif`) and explicit edge semantics (`NEXT`, `HAS_EVENT`, `INVOLVES`, `USES_MOTIF`).
- Keep scene plan compatibility (`plan.json`) but map every plan scene to graph nodes.

### 2) Constrained generation (symbolic + LM)
- Lexically constrained decoding (GBS) and logic-constrained decoding (NeuroLogic) demonstrate that constraint satisfaction can be done at decode/search time.
- For MVP and compatibility, full custom decoder integration is heavy; equivalent value can be achieved by **constraint-aware candidate generation + selection** before prose realization.

**Implication for this repo**
- Enforce hard constraints in `grow-shadow` and `solve` (forbidden terms rejected; required elements tracked).
- Keep `draft --grounded` strictly downstream of solved graph trajectory (no freeform draft path in this workflow).

### 3) Probabilistic growth/search on event graphs
- Narrative exploration with MCTS and branching tree UI is effective for “what-if” exploration.
- For CLI MVP, beam search is lower complexity and deterministic enough for repeatable tests.

**Implication for this repo**
- `context --graph-stats`: compute event transition priors from extracted events.
- `grow-shadow --auto`: generate multiple candidates per scene with plausibility score from transition priors + participation priors + motif priors.
- `solve`: beam-select best valid trajectory.

### 4) Style controls + motif budgets
- Controllable generation literature (e.g., FUDGE) supports lightweight modular control signals.
- For this codebase, useful controls are **budget features** instead of model fine-tuning: words/scene targets, lore/song motif budgets, dialogue ratio target.

**Implication for this repo**
- Emit `register_style_budgets` in `context_stats.json`.
- Use budgets as policy signals in generation/audit artifacts.

## Selected implementation choices for MVP v1
1. **Artifact-first architecture** (JSON contracts) over direct Neo4j dependency for commands:
   - Works with existing repo data artifacts (`*_events.json`)
   - Easier to test offline
2. **Statistical priors from corpus event files**:
   - Event transition probabilities (Markov-like)
   - Character participation priors
   - Motif/reference token priors
3. **Probabilistic candidate graph growth**:
   - 3 candidates per scene
   - Plausibility score combines transition + character + motif priors
4. **Beam solve**:
   - Deterministic, transparent objective
   - Constraint checks integrated
5. **Grounded drafting with traceability**:
   - Chapter prose generated from solved trajectory only
   - Mandatory section trace mapping to shadow/canon identifiers
6. **Audit command**:
   - Coverage, constraint, and trace-reference checks

## Why this is appropriate for current codebase
- Aligns with existing `story` command surface and project artifacts.
- Preserves backward compatibility (`init/plan/validate` still work).
- Provides a true graph-native workflow path without requiring immediate infrastructure changes.
- Testable with fast unit/integration tests using local JSON fixtures.

## References
1. Riedl, M. O., & Young, R. M. (2010). *Narrative Planning: Balancing Plot and Character*. JAIR.  
   - https://jair.org/index.php/jair/article/view/10669
   - https://arxiv.org/abs/1401.3841
2. Hokamp, C., & Liu, Q. (2017). *Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search*. ACL.  
   - https://aclanthology.org/P17-1141/
3. Lu, X. et al. (2021). *NeuroLogic Decoding: (Un)supervised Neural Text Generation with Predicate Logic Constraints*. NAACL.  
   - https://arxiv.org/abs/2010.12884
4. Yang, K., & Klein, D. (2021). *FUDGE: Controlled Text Generation With Future Discriminators*. NAACL.  
   - https://aclanthology.org/2021.naacl-main.276/
5. Ghaffari, P., & Hokamp, C. (2025). *Narrative Studio: Visual narrative exploration using LLMs and Monte Carlo Tree Search*.  
   - https://arxiv.org/html/2504.02426
6. Andronis, A. et al. (2025). *Guiding Generative Storytelling with Knowledge Graphs*.  
   - https://arxiv.org/html/2505.24803v2
