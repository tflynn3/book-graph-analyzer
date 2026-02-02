"""
Corpus Management Module

Phase 6: Handle multiple books, cross-book entity resolution,
and unified queries across an author's complete works.
"""

from .manager import CorpusManager, BookInfo
from .resolver import CrossBookResolver

__all__ = [
    "CorpusManager",
    "BookInfo",
    "CrossBookResolver",
]
