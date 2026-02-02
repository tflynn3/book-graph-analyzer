"""Passage model for text storage."""

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

    def short_location(self) -> str:
        """Return a short location string."""
        return f"{self.book} / Ch.{self.chapter_num} / P{self.paragraph_num} / S{self.sentence_num}"
