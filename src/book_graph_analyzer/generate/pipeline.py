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

        if self._should_run_lore_enforcement(scene):
            scene, lore_violations = self.scene_generator._run_lore_enforcement(scene, neo4j_context)
            if lore_violations or scene.revision_count > 0:
                scene.pipeline_stages_run.append("lore_enforce")

        if self.config.enable_voice_patch and voice_profiles:
            deviation = self.voice_patcher.estimate_max_deviation(scene, voice_profiles)
            if deviation >= self.config.voice_patch_threshold:
                scene = self.voice_patcher.patch(scene, voice_profiles, self.config.voice_patch_threshold)
                scene.pipeline_stages_run.append("voice_patch")

        return scene, lore_violations

    def _should_run_lore_enforcement(self, scene: Scene) -> bool:
        if not self.config.lore_enforce_only_major:
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
