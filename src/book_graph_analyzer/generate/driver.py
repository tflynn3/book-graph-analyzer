"""Hierarchical generation driver for Story -> Chapter -> Scene orchestration."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .context import ContextAssembler
from .generator import SceneGenerator
from .models import Chapter, Story
from .outliner import ChapterOutline, StoryOutline
from .shadow.graph import ShadowGraph


class NovelDriver:
    """Drive end-to-end novel generation with per-scene checkpoints."""

    def __init__(
        self,
        scene_generator: SceneGenerator,
        context_assembler: ContextAssembler,
        shadow_graph: ShadowGraph,
        checkpoint_dir: str,
    ):
        self.scene_generator = scene_generator
        self.context_assembler = context_assembler
        self.shadow_graph = shadow_graph
        self.checkpoint_dir = Path(checkpoint_dir)

    def generate_novel(
        self,
        story_outline: StoryOutline,
        resume: bool = True,
    ) -> Story:
        """Generate all scenes from a StoryOutline into a Story object."""
        story = self._load_checkpoint(story_outline.id) if resume else None
        if not story:
            story = Story(
                id=story_outline.id,
                title=f"{story_outline.character} — Generated Novel",
                premise=f"Interpolate from '{story_outline.anchor_a.description}' to '{story_outline.anchor_b.description}'",
                outline=f"Character focus: {story_outline.character}",
                chapters=[self._chapter_from_outline(ch) for ch in story_outline.chapters],
            )

        total_words = sum(scene.word_count for chapter in story.chapters for scene in chapter.scenes)

        for ch_idx, chapter_outline in enumerate(story_outline.chapters, start=1):
            chapter = self._ensure_chapter(story, chapter_outline, ch_idx)
            scene_beats = self._scene_beats(chapter_outline)

            for sc_idx, beat in enumerate(scene_beats, start=1):
                existing = self._find_scene(chapter, sc_idx)
                if existing and existing.text.strip():
                    continue

                assembled = self.context_assembler.assemble(
                    story_id=story.id,
                    characters=beat["characters"],
                    place=beat["setting"],
                    chapter_num=ch_idx,
                    scene_num=sc_idx,
                )

                scene = self.scene_generator.generate_scene(
                    scene_goal=beat["goal"],
                    characters=beat["characters"],
                    place=beat["setting"],
                    assembled_context=assembled,
                    story_id=story.id,
                    chapter_num=ch_idx,
                    scene_num=sc_idx,
                )

                scene.number = sc_idx
                if existing:
                    chapter.scenes[sc_idx - 1] = scene
                else:
                    chapter.scenes.append(scene)

                delta = self.shadow_graph.extract_delta_from_scene(
                    scene_text=scene.text,
                    characters=scene.characters,
                    scene_id=scene.id,
                    chapter_num=ch_idx,
                    scene_num=sc_idx,
                )
                if not delta.scene_summary:
                    delta.scene_summary = scene.summary or beat["goal"]
                self.shadow_graph.commit_state_delta(delta)

                total_words += scene.word_count
                print(
                    f"Chapter {ch_idx} / {len(story_outline.chapters)} — "
                    f"Scene {sc_idx} / {len(scene_beats)}"
                )
                print(f"Generating: \"{beat['goal']}\"")
                print(
                    f"  Characters: {', '.join(scene.characters)} | Place: {beat['setting']}"
                )
                print(
                    "  Scores: "
                    f"lore={scene.scores.lore_score:.2f} "
                    f"style={scene.scores.style_score:.2f} "
                    f"narrative={scene.scores.narrative_score:.2f} "
                    f"overall={scene.scores.overall:.2f}"
                )
                print(f"  Words so far: {total_words:,}\n")

                story.updated_at = datetime.now()
                self._save_checkpoint(story)

        story.updated_at = datetime.now()
        self._save_checkpoint(story)
        return story

    def _load_checkpoint(self, story_id: str) -> Optional[Story]:
        checkpoint_path = self._checkpoint_path(story_id)
        if not checkpoint_path.exists():
            return None

        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return Story.from_dict(data)

    def _save_checkpoint(self, story: Story) -> None:
        checkpoint_path = self._checkpoint_path(story.id)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps(story.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _checkpoint_path(self, story_id: str) -> Path:
        return self.checkpoint_dir / story_id / "story.json"

    @staticmethod
    def _chapter_from_outline(chapter_outline: ChapterOutline) -> Chapter:
        return chapter_outline.to_chapter()

    @staticmethod
    def _ensure_chapter(story: Story, chapter_outline: ChapterOutline, chapter_number: int) -> Chapter:
        while len(story.chapters) < chapter_number:
            story.chapters.append(chapter_outline.to_chapter())
        chapter = story.chapters[chapter_number - 1]
        if not chapter.id:
            chapter.id = f"ch_{chapter_number:02d}"
        chapter.number = chapter_number
        chapter.title = chapter.title or chapter_outline.title
        chapter.summary = chapter.summary or chapter_outline.beat
        return chapter

    @staticmethod
    def _find_scene(chapter: Chapter, scene_number: int):
        if scene_number <= len(chapter.scenes):
            return chapter.scenes[scene_number - 1]
        return None

    @staticmethod
    def _scene_beats(chapter_outline: ChapterOutline) -> list[dict]:
        """Return per-scene goals from structured chapter beat JSON when available."""
        parsed = NovelDriver._extract_json(chapter_outline.beat)
        scenes = parsed.get("scenes", []) if isinstance(parsed, dict) else []

        beats: list[dict] = []
        if scenes:
            for idx, scene in enumerate(scenes, start=1):
                chars = list(scene.get("characters", chapter_outline.characters or []))
                beats.append(
                    {
                        "scene": int(scene.get("scene", idx) or idx),
                        "goal": str(scene.get("goal") or scene.get("intent") or chapter_outline.beat),
                        "setting": str(scene.get("setting") or chapter_outline.setting or "Unknown"),
                        "characters": chars,
                    }
                )

        if not beats:
            beats.append(
                {
                    "scene": 1,
                    "goal": chapter_outline.beat,
                    "setting": chapter_outline.setting or "Unknown",
                    "characters": chapter_outline.characters or ["Unknown"],
                }
            )
        return beats

    @staticmethod
    def _extract_json(text: str) -> dict:
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
