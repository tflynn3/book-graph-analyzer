"""Data models for story generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class GenerationStatus(Enum):
    """Status of a generated piece."""
    DRAFT = "draft"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    FLAGGED = "flagged"  # Needs human review


@dataclass
class GenerationConfig:
    """Configuration for generation."""
    model: str = "meta-llama/Llama-3.1-70B-Instruct"
    temperature: float = 0.8
    max_tokens: int = 1500
    
    # Scoring weights
    weight_lore: float = 0.30
    weight_style: float = 0.20
    weight_narrative: float = 0.35
    weight_consistency: float = 0.15
    
    # Thresholds
    min_quality_score: float = 0.6  # Below this = flagged for review
    max_critique_iterations: int = 3
    
    # Context
    context_window_scenes: int = 3  # How many previous scenes to include


@dataclass
class SceneScores:
    """Quality scores for a generated scene."""
    # Individual dimensions
    lore_score: float = 0.0       # World bible compliance
    style_score: float = 0.0      # Tolkien style match
    narrative_score: float = 0.0  # Engagement, pacing, dialogue
    consistency_score: float = 0.0  # Character voice, timeline
    
    # Narrative sub-scores (from LLM judge)
    engagement: float = 0.0
    pacing: float = 0.0
    dialogue: float = 0.0
    imagery: float = 0.0
    emotional_weight: float = 0.0
    
    # Overall
    overall: float = 0.0
    
    def compute_overall(self, config: GenerationConfig) -> float:
        """Compute weighted overall score."""
        self.overall = (
            self.lore_score * config.weight_lore +
            self.style_score * config.weight_style +
            self.narrative_score * config.weight_narrative +
            self.consistency_score * config.weight_consistency
        )
        return self.overall
    
    def to_dict(self) -> dict:
        return {
            "lore_score": self.lore_score,
            "style_score": self.style_score,
            "narrative_score": self.narrative_score,
            "consistency_score": self.consistency_score,
            "engagement": self.engagement,
            "pacing": self.pacing,
            "dialogue": self.dialogue,
            "imagery": self.imagery,
            "emotional_weight": self.emotional_weight,
            "overall": self.overall,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneScores":
        return cls(
            lore_score=float(data.get("lore_score", 0.0) or 0.0),
            style_score=float(data.get("style_score", 0.0) or 0.0),
            narrative_score=float(data.get("narrative_score", 0.0) or 0.0),
            consistency_score=float(data.get("consistency_score", 0.0) or 0.0),
            engagement=float(data.get("engagement", 0.0) or 0.0),
            pacing=float(data.get("pacing", 0.0) or 0.0),
            dialogue=float(data.get("dialogue", 0.0) or 0.0),
            imagery=float(data.get("imagery", 0.0) or 0.0),
            emotional_weight=float(data.get("emotional_weight", 0.0) or 0.0),
            overall=float(data.get("overall", 0.0) or 0.0),
        )


@dataclass
class Scene:
    """A generated scene."""
    id: str
    number: int  # Position in chapter
    
    # Content
    text: str
    summary: str = ""
    
    # Entities involved (names, resolved to Neo4j IDs later)
    characters: list[str] = field(default_factory=list)
    places: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    events_depicted: list[str] = field(default_factory=list)
    
    # Scores
    scores: SceneScores = field(default_factory=SceneScores)
    
    # Status
    status: GenerationStatus = GenerationStatus.DRAFT
    critique_notes: list[str] = field(default_factory=list)
    revision_count: int = 0
    
    # Meta
    word_count: int = 0
    generated_at: datetime = field(default_factory=datetime.now)
    model_used: str = ""
    generation_prompt: str = ""
    context_snapshot: Optional["AssembledContext"] = None
    
    def __post_init__(self):
        if not self.word_count:
            self.word_count = len(self.text.split())
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "number": self.number,
            "text": self.text,
            "summary": self.summary,
            "characters": self.characters,
            "places": self.places,
            "objects": self.objects,
            "events_depicted": self.events_depicted,
            "scores": self.scores.to_dict(),
            "status": self.status.value,
            "critique_notes": self.critique_notes,
            "revision_count": self.revision_count,
            "word_count": self.word_count,
            "generated_at": self.generated_at.isoformat(),
            "model_used": self.model_used,
            "context_snapshot": (
                self.context_snapshot.to_dict() if self.context_snapshot else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        from .context import AssembledContext

        generated_at_raw = data.get("generated_at")
        generated_at = datetime.now()
        if generated_at_raw:
            try:
                generated_at = datetime.fromisoformat(generated_at_raw)
            except ValueError:
                pass

        status_raw = data.get("status", GenerationStatus.DRAFT.value)
        try:
            status = GenerationStatus(status_raw)
        except ValueError:
            status = GenerationStatus.DRAFT

        context_snapshot_raw = data.get("context_snapshot")
        context_snapshot = None
        if isinstance(context_snapshot_raw, dict):
            context_snapshot = AssembledContext.from_dict(context_snapshot_raw)

        return cls(
            id=str(data.get("id", "")),
            number=int(data.get("number", 0) or 0),
            text=str(data.get("text", "")),
            summary=str(data.get("summary", "")),
            characters=list(data.get("characters", [])),
            places=list(data.get("places", [])),
            objects=list(data.get("objects", [])),
            events_depicted=list(data.get("events_depicted", [])),
            scores=SceneScores.from_dict(data.get("scores", {}) or {}),
            status=status,
            critique_notes=list(data.get("critique_notes", [])),
            revision_count=int(data.get("revision_count", 0) or 0),
            word_count=int(data.get("word_count", 0) or 0),
            generated_at=generated_at,
            model_used=str(data.get("model_used", "")),
            generation_prompt=str(data.get("generation_prompt", "")),
            context_snapshot=context_snapshot,
        )


@dataclass
class Chapter:
    """A chapter containing scenes."""
    id: str
    number: int
    title: str = ""
    summary: str = ""
    
    scenes: list[Scene] = field(default_factory=list)
    
    # Planning
    outline: str = ""  # Beat-by-beat outline
    canonical_constraint: str = ""
    plot_thread_opens: str = ""
    plot_thread_closes: str = ""
    target_scenes: int = 5
    
    def add_scene(self, scene: Scene) -> None:
        scene.number = len(self.scenes) + 1
        self.scenes.append(scene)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "number": self.number,
            "title": self.title,
            "summary": self.summary,
            "outline": self.outline,
            "canonical_constraint": self.canonical_constraint,
            "plot_thread_opens": self.plot_thread_opens,
            "plot_thread_closes": self.plot_thread_closes,
            "scenes": [s.to_dict() for s in self.scenes],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chapter":
        chapter = cls(
            id=str(data.get("id", "")),
            number=int(data.get("number", 0) or 0),
            title=str(data.get("title", "")),
            summary=str(data.get("summary", "")),
            outline=str(data.get("outline", "")),
            canonical_constraint=str(data.get("canonical_constraint", "")),
            plot_thread_opens=str(data.get("plot_thread_opens", "")),
            plot_thread_closes=str(data.get("plot_thread_closes", "")),
            target_scenes=int(data.get("target_scenes", 5) or 5),
        )
        chapter.scenes = [Scene.from_dict(s) for s in data.get("scenes", [])]
        return chapter


@dataclass
class Story:
    """A generated story."""
    id: str
    title: str
    
    # Source material
    corpus_name: str = ""  # Which corpus this is based on
    
    # Content
    chapters: list[Chapter] = field(default_factory=list)
    
    # Planning
    premise: str = ""
    outline: str = ""
    
    # Meta
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_chapter(self, chapter: Chapter) -> None:
        chapter.number = len(self.chapters) + 1
        self.chapters.append(chapter)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "corpus_name": self.corpus_name,
            "premise": self.premise,
            "outline": self.outline,
            "chapters": [c.to_dict() for c in self.chapters],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Story":
        created_at_raw = data.get("created_at")
        updated_at_raw = data.get("updated_at")

        created_at = datetime.now()
        updated_at = datetime.now()

        if created_at_raw:
            try:
                created_at = datetime.fromisoformat(created_at_raw)
            except ValueError:
                pass
        if updated_at_raw:
            try:
                updated_at = datetime.fromisoformat(updated_at_raw)
            except ValueError:
                pass

        story = cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "Untitled Story")),
            corpus_name=str(data.get("corpus_name", "")),
            premise=str(data.get("premise", "")),
            outline=str(data.get("outline", "")),
            created_at=created_at,
            updated_at=updated_at,
        )
        story.chapters = [Chapter.from_dict(c) for c in data.get("chapters", [])]
        return story
