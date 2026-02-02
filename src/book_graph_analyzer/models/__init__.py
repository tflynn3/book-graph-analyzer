"""Data models for entities and relationships."""

from book_graph_analyzer.models.entities import Character, Place, Object, Event, Concept
from book_graph_analyzer.models.passage import Passage

__all__ = ["Character", "Place", "Object", "Event", "Concept", "Passage"]
