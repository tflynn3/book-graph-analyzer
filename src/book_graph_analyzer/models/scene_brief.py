"""SceneBrief and PipelineResult models for the Scene Generation Pipeline.

SceneBrief  — the structured input spec for generating a scene.
PipelineCheckResult — the result of a single pre-flight validation check.
PipelineResult — the full output of the pipeline: all checks + assembled prompt.

The pipeline in generate/pipeline.py accepts a SceneBrief and runs:
  1. LoreRule validation (Issue #6 / #7)
  2. Register tagging (Issue #9)
  3. Emotional arc validation (Issue #8)
  4. NarrativeWeight computation (Issue #5)
  5. Generation prompt assembly
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SceneBrief:
    """Structured input specification for scene generation.

    Contains everything the pipeline needs to validate and generate a scene.
    Serialises to/from JSON for CLI use.
    """

    # ── Characters ─────────────────────────────────────────────────────────
    characters: list[str] = field(default_factory=list)
    # List of character IDs (e.g. 'frodo_baggins', 'samwise_gamgee')

    # ── Setting ─────────────────────────────────────────────────────────────
    location: Optional[str] = None         # Place name e.g. 'The plains of Gorgoroth'
    story_era: Optional[str] = None        # Era e.g. 'Third Age'
    story_year: Optional[int] = None       # Year in era e.g. 3019

    # ── Scene content ────────────────────────────────────────────────────────
    scene_summary: str = ""
    # One or two sentences describing what the scene is about.
    # E.g. "Frodo and Sam reach the edge of Mordor. The Ring weighs heavily."

    event_types: list[str] = field(default_factory=list)
    # e.g. ['travel', 'dialogue', 'internal_monologue']

    object_names: list[str] = field(default_factory=list)
    # Objects present in the scene e.g. ['The One Ring', 'Sting']

    # ── Style targets ────────────────────────────────────────────────────────
    target_register: Optional[str] = None
    # ProseRegister value e.g. 'elegiac'. None = auto-detect from scene_summary.

    target_emotional_states: dict[str, str] = field(default_factory=dict)
    # Map from character_id → target TolkienRegister.
    # E.g. {'frodo_baggins': 'burden', 'samwise_gamgee': 'resolute'}

    # ── Narrative weight targets ─────────────────────────────────────────────
    narrative_weight_targets: dict[str, float] = field(default_factory=dict)
    # Map from NarrativeWeight component name → target score.
    # E.g. {'temporal_depth': 0.7, 'thematic_threads': 0.6}

    # ── Validation controls ──────────────────────────────────────────────────
    lore_check_enabled: bool = True
    arc_check_enabled: bool = True
    register_tag_enabled: bool = True
    weight_check_enabled: bool = True

    lore_check_categories: Optional[list[str]] = None
    # None = check all categories. List to restrict e.g. ['magic', 'race']

    era_references: list[str] = field(default_factory=list)
    # Additional eras this scene references (for temporal depth scoring)

    def to_dict(self) -> dict:
        return {
            "characters": self.characters,
            "location": self.location,
            "story_era": self.story_era,
            "story_year": self.story_year,
            "scene_summary": self.scene_summary,
            "event_types": self.event_types,
            "object_names": self.object_names,
            "target_register": self.target_register,
            "target_emotional_states": self.target_emotional_states,
            "narrative_weight_targets": self.narrative_weight_targets,
            "lore_check_enabled": self.lore_check_enabled,
            "arc_check_enabled": self.arc_check_enabled,
            "register_tag_enabled": self.register_tag_enabled,
            "weight_check_enabled": self.weight_check_enabled,
            "lore_check_categories": self.lore_check_categories,
            "era_references": self.era_references,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SceneBrief":
        return cls(
            characters=list(d.get("characters", [])),
            location=d.get("location"),
            story_era=d.get("story_era"),
            story_year=d.get("story_year"),
            scene_summary=d.get("scene_summary", ""),
            event_types=list(d.get("event_types", [])),
            object_names=list(d.get("object_names", [])),
            target_register=d.get("target_register"),
            target_emotional_states=dict(d.get("target_emotional_states", {})),
            narrative_weight_targets=dict(d.get("narrative_weight_targets", {})),
            lore_check_enabled=bool(d.get("lore_check_enabled", True)),
            arc_check_enabled=bool(d.get("arc_check_enabled", True)),
            register_tag_enabled=bool(d.get("register_tag_enabled", True)),
            weight_check_enabled=bool(d.get("weight_check_enabled", True)),
            lore_check_categories=d.get("lore_check_categories"),
            era_references=list(d.get("era_references", [])),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "SceneBrief":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_json_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def short_label(self) -> str:
        """One-line label for display."""
        chars = ", ".join(self.characters[:3]) or "no characters"
        loc = f" at {self.location}" if self.location else ""
        year = f" (TA {self.story_year})" if self.story_year else ""
        return f"{chars}{loc}{year}"


@dataclass
class PipelineCheckResult:
    """Result of one named pre-flight check in the pipeline."""
    check_name: str
    passed: bool
    blocking: bool = True     # If True, a fail here blocks generation
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def status_icon(self) -> str:
        if self.passed:
            return "✓"
        return "✗" if self.blocking else "⚠"

    def one_line(self) -> str:
        icon = self.status_icon()
        summary = f"{icon} {self.check_name}: "
        if self.passed:
            summary += "OK"
        elif self.blocking:
            summary += f"BLOCKED — {self.errors[0][:80] if self.errors else 'violation'}"
        else:
            summary += f"WARN — {self.warnings[0][:80] if self.warnings else 'warning'}"
        return summary


@dataclass
class PipelineResult:
    """Full output of the pre-flight pipeline.

    Contains all check results, the assembled generation prompt,
    and computed metrics for the scene.
    """

    brief: SceneBrief

    # ── Per-check results ────────────────────────────────────────────────────
    lore_check: Optional[PipelineCheckResult] = None
    arc_checks: dict[str, PipelineCheckResult] = field(default_factory=dict)
    register_check: Optional[PipelineCheckResult] = None
    weight_check: Optional[PipelineCheckResult] = None

    # ── Computed values ──────────────────────────────────────────────────────
    detected_register: Optional[str] = None          # auto-detected from scene_summary
    final_register: Optional[str] = None             # = target_register ?? detected_register
    narrative_weight_overall: float = 0.0
    narrative_weight_summary: str = ""
    improvement_suggestions: list[str] = field(default_factory=list)

    # ── Lore violation detail ────────────────────────────────────────────────
    hard_violations: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)

    # ── Pipeline status ──────────────────────────────────────────────────────
    pipeline_passed: bool = True
    pipeline_errors: list[str] = field(default_factory=list)
    pipeline_warnings: list[str] = field(default_factory=list)

    # ── Assembled generation prompt ──────────────────────────────────────────
    generation_prompt: str = ""

    # ── Optional generated text (filled by actual LLM generation) ───────────
    generated_text: Optional[str] = None

    def all_checks(self) -> list[PipelineCheckResult]:
        checks = []
        if self.lore_check:
            checks.append(self.lore_check)
        checks.extend(self.arc_checks.values())
        if self.register_check:
            checks.append(self.register_check)
        if self.weight_check:
            checks.append(self.weight_check)
        return checks

    def summary(self) -> str:
        lines = [
            f"Pipeline Pre-flight: {self.brief.short_label()}",
            "",
        ]
        for check in self.all_checks():
            lines.append(f"  {check.one_line()}")

        lines += [
            "",
            f"  Final register: {self.final_register or 'not set'}",
            f"  Narrative weight: {self.narrative_weight_overall:.3f}",
        ]

        if self.hard_violations:
            lines += ["", "  Hard violations (BLOCKED):"]
            for v in self.hard_violations:
                lines.append(f"    • {v}")

        if self.soft_warnings:
            lines += ["", "  Soft warnings:"]
            for w in self.soft_warnings[:5]:
                lines.append(f"    ~ {w}")

        if self.improvement_suggestions:
            lines += ["", "  Improvement suggestions:"]
            for s in self.improvement_suggestions[:3]:
                lines.append(f"    → {s[:100]}")

        overall = "[PASS]" if self.pipeline_passed else "[FAIL]"
        lines += ["", f"  Overall: {overall}"]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        def _check_to_dict(c: Optional[PipelineCheckResult]) -> Optional[dict]:
            if c is None:
                return None
            return {
                "check_name": c.check_name,
                "passed": c.passed,
                "blocking": c.blocking,
                "warnings": c.warnings,
                "errors": c.errors,
            }

        return {
            "brief": self.brief.to_dict(),
            "pipeline_passed": self.pipeline_passed,
            "final_register": self.final_register,
            "narrative_weight_overall": self.narrative_weight_overall,
            "hard_violations": self.hard_violations,
            "soft_warnings": self.soft_warnings,
            "improvement_suggestions": self.improvement_suggestions,
            "lore_check": _check_to_dict(self.lore_check),
            "arc_checks": {
                k: _check_to_dict(v)
                for k, v in self.arc_checks.items()
            },
            "register_check": _check_to_dict(self.register_check),
            "weight_check": _check_to_dict(self.weight_check),
            "generation_prompt": self.generation_prompt[:2000],  # Truncated for storage
        }
