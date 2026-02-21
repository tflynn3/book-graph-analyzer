# Issue #21 Design: Outliner Engine (Whitespace Interpolator)

## Goal
Add an `OutlinerEngine` that can:
1. Resolve two canonical anchor events from Neo4j EventGraph.
2. Generate a structured chapter outline in the whitespace between anchors.
3. Inject constraints from World Bible + known in-gap events.
4. Expose this end-to-end via CLI (`bga generate outline`) and save JSON output.

## Research references (concise)
1. Dynamic hierarchical outlining for long-form story generation (DOME): outlines improve coherence and long-horizon planning.  
   <https://aclanthology.org/2025.naacl-long.63/>
2. ArXiv HTML mirror of DOME showing planner/writer split and JSON planning prompts.  
   <https://arxiv.org/html/2412.13575v1>
3. StoryWriter multi-agent framework: distinct planning stage before prose generation.  
   <https://openreview.net/pdf?id=FQj3KK0Qg4>
4. OpenAI structured outputs guidance (schema-constrained JSON) for reliable machine parsing.  
   <https://platform.openai.com/docs/guides/structured-outputs>
5. JSON Schema 2020-12 core spec for robust, tool-friendly structured payloads.  
   <https://json-schema.org/draft/2020-12/json-schema-core.html>

## Design decisions

### 1) New module: `generate/outliner.py`
- Added core models:
  - `CanonicalEvent`
  - `ChapterOutline`
  - `StoryOutline`
- Added `OutlinerEngine` methods:
  - `find_anchor_points(character, point_a_hint, point_b_hint)`
  - `generate_story_outline(anchor_a, anchor_b, num_chapters, character)`
  - `generate_chapter_outline(story_outline, chapter_num, num_scenes)`

### 2) Anchor resolution strategy
- Query `(:Event)` nodes by character relevance (`agent/description/patient`) and hint overlap.
- Rank by simple score + year ordering.
- Return concrete graph-backed anchor event payloads.

### 3) Constraint injection strategy
- World Bible: select relevant rules by matching character/era terms; fallback to a short global rule subset.
- EventGraph: gather known in-gap events and inject as exclusion list (`DO NOT DEPICT OR REWRITE`).
- Prompt output required as strict JSON chapter beats (intent-level, not prose).

### 4) Existing model extension
- `generate.models.Chapter` now includes:
  - `canonical_constraint`
  - `plot_thread_opens`
  - `plot_thread_closes`
- Included in `Chapter.to_dict()` for serialization stability.

### 5) CLI integration
- New command:
  - `bga generate outline --character Tuor --from "arrives in Nevrast" --to "reaches Gondolin" --chapters 10`
- Supports optional `--world-bible` and `--output`.
- If `--output` omitted, writes to `data/output/outline_<character>_<id>.json`.

## Scope notes
- Keeps implementation focused to issue #21 (planner/outliner layer).
- Does not attempt full scene prose generation pipeline changes.
- Uses existing LLM abstraction and Neo4j connection patterns already in repository.
