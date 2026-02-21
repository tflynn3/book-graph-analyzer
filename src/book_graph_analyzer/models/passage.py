"""Passage model for text storage."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class Passage(BaseModel):
    """A unit of source text (typically a sentence)."""

    id: str
    text: str
    book: str
    chapter: str
    chapter_num: int
    paragraph_num: int
    sentence_num: int
    char_offset: int
    sentiment: str | None = None
    scene_type: str | None = None  # dialogue, action, description

    # -----------------------------------------------------------------------
    # Author period — when Tolkien wrote/revised this text
    # Used for conflict resolution (later texts supersede earlier for retcons)
    # -----------------------------------------------------------------------
    author_period: Optional[str] = None  # 'early' | 'middle' | 'late' (AuthorPeriod)
    source_compilation: Optional[str] = None  # e.g. 'Unfinished Tales', 'HoME Vol. 5'

    # -----------------------------------------------------------------------
    # Story-time frame (when this passage *occurs* in the narrative)
    # -----------------------------------------------------------------------
    story_era: Optional[str] = None    # e.g. 'Third Age'
    story_year: Optional[int] = None   # approximate year within story_era

    # -----------------------------------------------------------------------
    # Temporal depth (oldest era *referenced* within the passage)
    # -----------------------------------------------------------------------
    temporal_depth_era: Optional[str] = None         # e.g. 'Before Time'
    temporal_depth_years_back: Optional[float] = None  # years before story_year

    # -----------------------------------------------------------------------
    # Multi-era reference count
    # -----------------------------------------------------------------------
    era_reference_count: int = 0  # how many distinct eras are referenced?

    # -----------------------------------------------------------------------
    # Scene type and register (Tolkien-specific)
    # -----------------------------------------------------------------------
    tolkien_register: Optional[str] = None  # 'high', 'archaic', 'colloquial', 'narrative'

    # -----------------------------------------------------------------------
    # Structural / speaker info
    # -----------------------------------------------------------------------
    pov_character_id: Optional[str] = None
    is_dialogue: bool = False
    speaker_ids: List[str] = Field(default_factory=list)

    # -----------------------------------------------------------------------
    # Style metrics
    # -----------------------------------------------------------------------
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    passive_ratio: float = 0.0
    dialogue_density: float = 0.0
    archaic_word_count: int = 0

    def short_location(self) -> str:
        """Return a short location string."""
        return f"{self.book} / Ch.{self.chapter_num} / P{self.paragraph_num} / S{self.sentence_num}"

    def temporal_summary(self) -> str:
        """Return a human-readable summary of this passage's temporal properties."""
        parts = []
        if self.story_era:
            year_str = f" {self.story_year}" if self.story_year else ""
            parts.append(f"set in: {self.story_era}{year_str}")
        if self.temporal_depth_era:
            parts.append(f"reaches back to: {self.temporal_depth_era}")
        if self.era_reference_count:
            parts.append(f"refs {self.era_reference_count} era(s)")
        return " | ".join(parts) if parts else "no temporal data"
