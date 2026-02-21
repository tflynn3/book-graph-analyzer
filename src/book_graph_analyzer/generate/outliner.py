"""Outliner engine for whitespace interpolation between canonical events."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..graph.connection import get_driver
from ..llm import LLMClient
from ..worldbible import WorldBible
from .models import Chapter


@dataclass
class CanonicalEvent:
    """Event anchor from the canonical EventGraph."""

    id: str
    description: str
    era: str = ""
    year: Optional[int] = None
    agent: str = ""
    source_book: str = ""


@dataclass
class ChapterOutline:
    """Structured chapter beat returned by the outliner."""

    number: int
    title: str
    beat: str
    characters: list[str] = field(default_factory=list)
    setting: str = ""
    canonical_constraint: str = ""
    plot_thread_opens: Optional[str] = None
    plot_thread_closes: Optional[str] = None

    def to_chapter(self) -> Chapter:
        return Chapter(
            id=f"ch_{self.number:02d}",
            number=self.number,
            title=self.title,
            summary=self.beat,
            outline=self.beat,
            canonical_constraint=self.canonical_constraint,
            plot_thread_opens=self.plot_thread_opens or "",
            plot_thread_closes=self.plot_thread_closes or "",
        )

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "beat": self.beat,
            "characters": self.characters,
            "setting": self.setting,
            "canonical_constraint": self.canonical_constraint,
            "plot_thread_opens": self.plot_thread_opens,
            "plot_thread_closes": self.plot_thread_closes,
        }


@dataclass
class StoryOutline:
    """Full story outline between two canonical anchors."""

    id: str
    character: str
    anchor_a: CanonicalEvent
    anchor_b: CanonicalEvent
    chapters: list[ChapterOutline] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "character": self.character,
            "anchor_a": self.anchor_a.__dict__,
            "anchor_b": self.anchor_b.__dict__,
            "chapters": [c.to_dict() for c in self.chapters],
            "generated_at": self.generated_at.isoformat(),
        }


class OutlinerEngine:
    """Generates hierarchical story outlines between canonical anchor points."""

    STORY_OUTLINE_PROMPT = """You are generating a chapter-level plot outline between two canonical events.

CHARACTER: {character}

ANCHOR A (already happened):
- {anchor_a}

ANCHOR B (must happen next):
- {anchor_b}

WORLD BIBLE HARD CONSTRAINTS:
{world_rules}

KNOWN CANONICAL EVENTS IN THIS GAP (DO NOT DEPICT OR REWRITE THESE):
{blocked_events}

Write exactly {num_chapters} chapter beats as JSON only.
Use intent-level beats, not prose scenes.

Output schema:
{{
  "chapters": [
    {{
      "number": 1,
      "title": "...",
      "beat": "...",
      "characters": ["..."],
      "setting": "...",
      "canonical_constraint": "...",
      "plot_thread_opens": "... or null",
      "plot_thread_closes": "... or null"
    }}
  ]
}}
"""

    CHAPTER_OUTLINE_PROMPT = """Expand this chapter beat into a scene-by-scene outline.

CHAPTER NUMBER: {chapter_num}
CHAPTER TITLE: {title}
BEAT: {beat}
CHARACTERS: {characters}
SETTING: {setting}

Return JSON with this schema:
{{
  "chapter": {chapter_num},
  "scenes": [
    {{"scene": 1, "intent": "...", "goal": "...", "setting": "...", "characters": ["..."]}}
  ]
}}

Generate exactly {num_scenes} scenes.
"""

    def __init__(self, llm: Optional[LLMClient] = None, driver=None):
        self.llm = llm or LLMClient()
        self.driver = driver if driver is not None else get_driver()
        self.world_bible: Optional[WorldBible] = None

    def load_world_bible(self, path: str) -> None:
        self.world_bible = WorldBible.load(path)

    def find_anchor_points(
        self,
        character: str,
        point_a_hint: str,
        point_b_hint: str,
    ) -> tuple[CanonicalEvent, CanonicalEvent]:
        """Query EventGraph for the closest matching canonical events."""
        if not self.driver:
            raise RuntimeError("Neo4j driver not available")

        with self.driver.session() as session:
            anchor_a = self._query_anchor(session, character, point_a_hint)
            anchor_b = self._query_anchor(session, character, point_b_hint)

        return anchor_a, anchor_b

    def generate_story_outline(
        self,
        anchor_a: CanonicalEvent,
        anchor_b: CanonicalEvent,
        num_chapters: int = 10,
        character: str = "",
    ) -> StoryOutline:
        """LLM-generate a chapter outline interpolating between two canon points."""
        blocked = self._events_in_gap(anchor_a, anchor_b, character)
        prompt = self.STORY_OUTLINE_PROMPT.format(
            character=character or anchor_a.agent or "Unknown",
            anchor_a=self._event_text(anchor_a),
            anchor_b=self._event_text(anchor_b),
            world_rules=self._hard_rules_text(character, [anchor_a.era, anchor_b.era]),
            blocked_events=self._blocked_events_text(blocked),
            num_chapters=num_chapters,
        )

        raw = self.llm.generate(prompt, temperature=0.3, max_tokens=2500)
        payload = self._extract_json(raw)

        chapter_rows = payload.get("chapters", []) if isinstance(payload, dict) else []
        chapters = [self._parse_chapter_row(r, idx + 1) for idx, r in enumerate(chapter_rows)]

        return StoryOutline(
            id=f"outline_{str(uuid.uuid4())[:8]}",
            character=character or anchor_a.agent,
            anchor_a=anchor_a,
            anchor_b=anchor_b,
            chapters=chapters,
        )

    def generate_chapter_outline(
        self,
        story_outline: StoryOutline,
        chapter_num: int,
        num_scenes: int = 5,
    ) -> ChapterOutline:
        """Expand a chapter beat into a scene-by-scene outline."""
        chapter = next((c for c in story_outline.chapters if c.number == chapter_num), None)
        if not chapter:
            raise ValueError(f"Chapter {chapter_num} not found")

        prompt = self.CHAPTER_OUTLINE_PROMPT.format(
            chapter_num=chapter.number,
            title=chapter.title,
            beat=chapter.beat,
            characters=", ".join(chapter.characters) or "None",
            setting=chapter.setting or "Unknown",
            num_scenes=num_scenes,
        )

        raw = self.llm.generate(prompt, temperature=0.4, max_tokens=1500)
        payload = self._extract_json(raw)

        scenes = payload.get("scenes", []) if isinstance(payload, dict) else []
        chapter.beat = json.dumps({"chapter": chapter.number, "scenes": scenes}, ensure_ascii=False)
        return chapter

    def _query_anchor(self, session, character: str, hint: str) -> CanonicalEvent:
        result = session.run(
            """
            MATCH (e:Event)
            WHERE (
                toLower(coalesce(e.agent, "")) CONTAINS toLower($character)
                OR toLower(coalesce(e.description, "")) CONTAINS toLower($character)
                OR toLower(coalesce(e.patient, "")) CONTAINS toLower($character)
            )
            WITH e,
                 (CASE WHEN toLower(coalesce(e.description, "")) CONTAINS toLower($hint) THEN 3 ELSE 0 END) +
                 (CASE WHEN toLower(coalesce(e.action, "")) CONTAINS toLower($hint) THEN 1 ELSE 0 END) +
                 (CASE WHEN toLower(coalesce(e.agent, "")) CONTAINS toLower($hint) THEN 1 ELSE 0 END) AS score
            RETURN
                coalesce(e.id, elementId(e)) AS id,
                coalesce(e.description, "") AS description,
                coalesce(e.era, "") AS era,
                e.year AS year,
                coalesce(e.agent, "") AS agent,
                coalesce(e.source_book, "") AS source_book,
                score
            ORDER BY score DESC, year ASC
            LIMIT 1
            """,
            character=character,
            hint=hint,
        ).single()

        if not result:
            raise ValueError(f"No canonical event found for character='{character}' and hint='{hint}'")

        return CanonicalEvent(
            id=result["id"],
            description=result["description"],
            era=result["era"],
            year=result["year"],
            agent=result["agent"],
            source_book=result["source_book"],
        )

    def _events_in_gap(
        self,
        anchor_a: CanonicalEvent,
        anchor_b: CanonicalEvent,
        character: str,
    ) -> list[CanonicalEvent]:
        if not self.driver:
            return []

        with self.driver.session() as session:
            # Best-effort temporal query. We only enforce year bounds when both anchors have years.
            if anchor_a.year is not None and anchor_b.year is not None and anchor_a.era == anchor_b.era:
                rows = session.run(
                    """
                    MATCH (e:Event)
                    WHERE toLower(coalesce(e.agent, "")) CONTAINS toLower($character)
                      AND coalesce(e.era, "") = $era
                      AND e.year IS NOT NULL
                      AND e.year > $start_year
                      AND e.year < $end_year
                    RETURN coalesce(e.id, elementId(e)) AS id,
                           coalesce(e.description, "") AS description,
                           coalesce(e.era, "") AS era,
                           e.year AS year,
                           coalesce(e.agent, "") AS agent,
                           coalesce(e.source_book, "") AS source_book
                    ORDER BY e.year ASC
                    LIMIT 50
                    """,
                    character=character,
                    era=anchor_a.era,
                    start_year=min(anchor_a.year, anchor_b.year),
                    end_year=max(anchor_a.year, anchor_b.year),
                )
            else:
                rows = session.run(
                    """
                    MATCH (e:Event)
                    WHERE toLower(coalesce(e.agent, "")) CONTAINS toLower($character)
                      AND coalesce(e.id, elementId(e)) <> $a_id
                      AND coalesce(e.id, elementId(e)) <> $b_id
                    RETURN coalesce(e.id, elementId(e)) AS id,
                           coalesce(e.description, "") AS description,
                           coalesce(e.era, "") AS era,
                           e.year AS year,
                           coalesce(e.agent, "") AS agent,
                           coalesce(e.source_book, "") AS source_book
                    ORDER BY e.year ASC
                    LIMIT 20
                    """,
                    character=character,
                    a_id=anchor_a.id,
                    b_id=anchor_b.id,
                )

            return [
                CanonicalEvent(
                    id=r["id"],
                    description=r["description"],
                    era=r["era"],
                    year=r["year"],
                    agent=r["agent"],
                    source_book=r["source_book"],
                )
                for r in rows
            ]

    def _hard_rules_text(self, character: str, eras: list[str]) -> str:
        if not self.world_bible:
            return "No world bible loaded. Preserve Tolkien-consistent behavior and chronology."

        terms = [character.lower()] + [e.lower() for e in eras if e]
        selected: list[str] = []
        for category_rules in self.world_bible.rules.values():
            for rule in category_rules:
                blob = f"{rule.title} {rule.description} {' '.join(rule.related_entities)}".lower()
                if any(t and t in blob for t in terms):
                    selected.append(f"- [{rule.category.value}] {rule.title}: {rule.description}")

        if not selected:
            # fallback: short global subset
            for category_rules in self.world_bible.rules.values():
                for rule in category_rules[:1]:
                    selected.append(f"- [{rule.category.value}] {rule.title}: {rule.description}")
                    if len(selected) >= 8:
                        break
                if len(selected) >= 8:
                    break

        return "\n".join(selected[:12])

    @staticmethod
    def _event_text(e: CanonicalEvent) -> str:
        when = f" ({e.era} {e.year})" if e.year is not None else (f" ({e.era})" if e.era else "")
        return f"{e.description}{when}".strip()

    @staticmethod
    def _blocked_events_text(events: list[CanonicalEvent]) -> str:
        if not events:
            return "- None explicitly known in graph for this interval"
        return "\n".join(f"- {OutlinerEngine._event_text(e)}" for e in events[:20])

    @staticmethod
    def _parse_chapter_row(row: dict, default_number: int) -> ChapterOutline:
        return ChapterOutline(
            number=int(row.get("number", default_number)),
            title=str(row.get("title", f"Chapter {default_number}")),
            beat=str(row.get("beat", "")),
            characters=list(row.get("characters", [])),
            setting=str(row.get("setting", "")),
            canonical_constraint=str(row.get("canonical_constraint", "")),
            plot_thread_opens=row.get("plot_thread_opens"),
            plot_thread_closes=row.get("plot_thread_closes"),
        )

    def _extract_json(self, text: str) -> dict:
        parsed = self.llm.extract_json(text)
        if isinstance(parsed, dict):
            return parsed

        match = re.search(r"\{[\s\S]*\}", text or "")
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {"chapters": []}
