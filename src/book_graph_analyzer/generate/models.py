"""Data models for story generation."""

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
        }


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
            "scenes": [s.to_dict() for s in self.scenes],
        }


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
