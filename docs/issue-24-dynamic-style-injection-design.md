# Issue #24 Design: Dynamic Style Injection

## Goal
Replace static Tolkien style instructions with scene-aware style constraints derived from passage data in Neo4j.

## Design decisions

1. **Scene type classification is deterministic first (keyword heuristic).**
   - Keeps generation predictable and testable.
   - Uses scene taxonomy from issue description.

2. **Style constraints come from existing `Passage` node metrics, not hardcoded values.**
   - Query by `scene_type` plus mapped tags.
   - Aggregate: average sentence length, dialogue density, passive ratio, archaic density, and top vocabulary.

3. **Graceful degradation.**
   - If Neo4j unavailable or fewer than `min_samples` (default 5), use static fallback style block (current behavior intent).

4. **Observability added to generated `Scene`.**
   - `scene_type: Optional[str]`
   - `style_constraints_used: Optional[dict]`

5. **Prompt integration is explicit.**
   - `GENERATION_PROMPT` now contains `{style_constraints}` slot.
   - The style block is fully injected at runtime per scene.

## Notes
- This implementation uses `Passage` properties (`scene_type`, `avg_sentence_length`, `dialogue_density`, `passive_ratio`, `archaic_word_count`) already present in the codebase write path.
- If/when explicit `PassageClassification` nodes are added to graph schema, the injector query can be extended to read from those nodes directly.
