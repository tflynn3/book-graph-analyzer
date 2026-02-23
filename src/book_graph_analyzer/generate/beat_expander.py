from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class BeatSheetBeat:
    beat_id: str
    position: int
    beat_type: str
    intent: str
    prose_budget_words: int
    cause_refs: list[str] = field(default_factory=list)
    failed_constraints: list[str] = field(default_factory=list)


@dataclass
class BeatSheetV1:
    schema_version: str = "beat-sheet-v1"
    project_slug: str = ""
    scene_id: str = ""
    method: str = "template"
    beats: list[BeatSheetBeat] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_slug": self.project_slug,
            "scene_id": self.scene_id,
            "method": self.method,
            "beats": [asdict(b) for b in self.beats],
        }


@dataclass
class ShadowBeatsV1:
    schema_version: str = "shadow-beats-v1"
    project_slug: str = ""
    method: str = "template"
    seed: int = 0
    beats: list[BeatSheetBeat] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_slug": self.project_slug,
            "method": self.method,
            "seed": self.seed,
            "beats": [asdict(b) for b in self.beats],
            "validation": self.validation,
        }
