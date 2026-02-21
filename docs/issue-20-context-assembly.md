# Issue #20 Design: Context Assembly Engine

## Research references (patterns used)

1. **Hierarchical Neural Story Generation** (Fan et al., 2018)  
   <https://arxiv.org/abs/1805.04833>  
   Pattern used: separate high-level planning context from realization text; keep prompts structured and layered.

2. **Generative Agents** (Park et al., 2023)  
   <https://arxiv.org/abs/2304.03442>  
   Pattern used: retrieve compact, behavior-relevant memory snippets (recent events + current state) instead of full transcript replay.

3. **CAMEL: Communicative Agents** (Li et al., 2023)  
   <https://arxiv.org/abs/2303.17760>  
   Pattern used: role/task coherence via concise shared context blocks so cooperating components stay aligned.

4. **MemGPT** (Packer et al., 2023)  
   <https://arxiv.org/abs/2310.08560>  
   Pattern used: explicit memory tiering and bounded context surface (trimmed sections and caps) to control token growth.

## Design decisions for this repo

- Introduced `generate/context.py` with:
  - `AssembledContext` dataclass for structured state (`character_states`, `recent_summaries`, `place_facts`, `active_plot_threads`)
  - `ContextAssembler` that combines Shadow Graph state + canonical Place facts + outline-derived threads
- Prompt serialization via `AssembledContext.to_prompt_block()` is intentionally compact:
  - max 3 recent summaries
  - max 3 place facts
  - max 4 active plot threads
- `SceneGenerator.generate_scene()` now accepts `assembled_context` directly (backward-compatible with existing `previous_context: str`).
- Auto-assembly is supported when `SceneGenerator` is configured with `shadow_graph` and a `story_id` is available.
- Added `Scene.context_snapshot` for debugging/auditability of exactly what context was injected at generation time.

## Scope boundaries

- No change to ShadowGraph mutation/extraction semantics.
- No invasive outliner refactor; active threads are extracted from existing `Story.outline` and recent `Chapter.outline` text in Neo4j.
- Existing callers that pass `previous_context` continue to work unchanged.
