"""Staged post-generation pipeline."""

from __future__ import annotations

import re
from typing import Optional

from .models import GenerationConfig, Scene
from .voice_patcher import VoicePatcher


class StagedPipeline:
    """Three-stage pipeline: drafter -> lore enforce -> voice patch."""

    def __init__(
        self,
        scene_generator,
        voice_patcher: VoicePatcher,
        config: Optional[GenerationConfig] = None,
    ):
        self.scene_generator = scene_generator
        self.voice_patcher = voice_patcher
        self.config = config or GenerationConfig()

    def run(
        self,
        scene: Scene,
        neo4j_context: dict,
        voice_profiles: Optional[dict] = None,
    ) -> tuple[Scene, list[dict]]:
        scene.pipeline_stages_run = ["drafter"]
        lore_violations: list[dict] = []

        if self._should_run_lore_enforcement(scene, neo4j_context):
            result = self.scene_generator._run_lore_enforcement(scene, neo4j_context)
            if isinstance(result, tuple) and len(result) == 3:
                scene, lore_violations, lore_verified = result
            else:
                # Compatibility for custom generators using the former
                # two-item result. Their explicit result counts as verified.
                scene, lore_violations = result
                lore_verified = True
            scene.pipeline_stages_run.append(
                "lore_enforce" if lore_verified else "lore_enforce_unverified"
            )

        if self.config.enable_voice_patch and voice_profiles:
            deviation = self.voice_patcher.estimate_max_deviation(scene, voice_profiles)
            if deviation >= self.config.voice_patch_threshold:
                scene = self.voice_patcher.patch(scene, voice_profiles, self.config.voice_patch_threshold)
                scene.pipeline_stages_run.append("voice_patch")

        return scene, lore_violations

    def _should_run_lore_enforcement(self, scene: Scene, neo4j_context: Optional[dict] = None) -> bool:
        if not self.config.lore_enforce_only_major:
            return True

        world_bible = getattr(self.scene_generator, "world_bible", None)
        world_rules = getattr(world_bible, "rules", None)
        if isinstance(world_rules, dict) and any(bool(rules) for rules in world_rules.values()):
            return True

        if scene.context_snapshot is not None:
            snapshot = scene.context_snapshot
            character_facts = any(
                state.location
                or state.possessions
                or state.conditions
                or state.last_scene
                for state in snapshot.character_states
            )
            place_facts = snapshot.place_facts or {}
            grounded_place = bool(
                place_facts.get("description")
                or place_facts.get("region")
                or place_facts.get("facts")
            )
            if (
                character_facts
                or snapshot.recent_summaries
                or grounded_place
                or snapshot.active_plot_threads
            ):
                return True

        if neo4j_context and (neo4j_context.get("recent_events") or neo4j_context.get("relationships")):
            return True

        # Cheap lexical gate: run expensive critique only when text shows obvious red flags.
        suspicious = [
            r"\bgun\b", r"\bpistol\b", r"\btelephone\b", r"\bcar\b", r"\binternet\b",
            r"\bokay\b", r"\bokay\b", r"\bemail\b", r"\bscience\b", r"\bapartment\b",
        ]
        lower = scene.text.lower()
        if any(re.search(p, lower) for p in suspicious):
            return True

        # Skip lore stage in low-risk cases to keep the pass cheap.
        return False
