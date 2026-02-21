# Issue #22 Design: Hierarchical Generation Driver

## Goal
Implement a resume-safe `NovelDriver` that orchestrates generation from `StoryOutline` into full `Story` output with per-scene checkpoints and Shadow Graph updates.

## Design decisions

1. **Checkpoint granularity: per scene**
   - Checkpoint file path: `{checkpoint_dir}/{story_id}/story.json`.
   - Save after each scene so failures lose minimal work.
   - Resume behavior: if a scene slot already has non-empty text, skip regeneration.

2. **Driver owns orchestration, not generation internals**
   - `NovelDriver` coordinates four steps for each scene:
     1) context assembly,
     2) scene generation,
     3) shadow graph delta extraction + commit,
     4) checkpoint persist.
   - `SceneGenerator` remains the source of generation quality scoring and critique loop.

3. **Structured scene plans with graceful fallback**
   - If chapter beat contains JSON `{"scenes": [...]}` (from chapter expansion), each scene uses its own goal/setting/characters.
   - If no structured scenes exist, the driver falls back to one scene per chapter beat.

4. **Model round-trip support**
   - Added `from_dict()` on Story/Chapter/Scene/SceneScores and `AssembledContext` so checkpoints are fully recoverable.

5. **CLI integration**
   - Added `bga generate novel --outline ... --checkpoint ... --resume` command.
   - Loads outline JSON, builds dependencies (`ShadowGraph`, `SceneGenerator`, `ContextAssembler`, `NovelDriver`), executes pipeline, writes final story JSON.

## Tradeoffs
- Resume skip detection uses story scene text (not hash/versioning), which is simple and robust for current requirements.
- Character-state reconstruction in `AssembledContext.from_dict` is intentionally minimal; scene generation uses fresh assembly each step so this is sufficient for checkpoint recovery.
