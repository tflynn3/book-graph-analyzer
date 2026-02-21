# Issue #23 Design — Staged Multi-Agent Pipeline Refactor

## Research References (patterns reused)

1. Self-Refine (Madaan et al.) — iterative draft -> feedback -> revision loop, directly aligned with staged post-generation correction.  
   https://arxiv.org/abs/2303.17651
2. Reflexion (Shinn et al.) — selective verbal feedback loops for cost-aware improvement, similar to conditional stage execution.  
   https://arxiv.org/abs/2303.11366
3. ReAct (Yao et al.) — decomposes model behavior into explicit phases, reinforcing stage boundaries and observability.  
   https://arxiv.org/abs/2210.03629
4. Constitutional AI (Bai et al.) — critique/revision for policy/lore alignment, mirrors lore enforcement as a bounded pass.  
   https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback

## Design Decisions for this repo

- Added explicit `StagedPipeline` as post-draft orchestrator with ordered stages:
  - `drafter` (existing `SceneGenerator.generate_scene` LLM generation)
  - `lore_enforce` (existing critique+revise loop, now isolated)
  - `voice_patch` (new dialogue-only targeted pass)
- Added `Scene.pipeline_stages_run: list[str]` for stage observability and regression debugging.
- Added generation config toggles:
  - `enable_voice_patch`
  - `voice_patch_threshold`
  - `lore_enforce_only_major`
- Implemented a narrow `VoicePatcher`:
  - Uses existing `voice.dialogue.extract_dialogue()` for speaker-attributed lines.
  - Computes deterministic profile deviation (contraction/formality/length).
  - Calls LLM only when deviation exceeds threshold.
  - Rewrites quoted dialogue content only; leaves narration unchanged.
- Refactored generator flow:
  - moved critique loop into `_run_lore_enforcement()`
  - `_score_scene()` now consumes staged lore violations rather than performing an extra lore LLM critique call.

## Cost/behavior impact

- Low-risk scene: drafter only (single generation call).
- Suspicious scene: lore stage enabled via lexical gate (`lore_enforce_only_major=True` policy).
- Voice stage runs only for high deviation against canonical voice profile.

## Tradeoffs

- Lore-stage lexical gate is intentionally conservative and heuristic; it can miss subtle lore drift.
- Voice patch currently targets quoted lines with extraction confidence assumptions; ambiguous speaker attribution is skipped rather than over-editing narration.
