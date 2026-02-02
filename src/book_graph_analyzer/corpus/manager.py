"""
Corpus Manager

Manages multiple books in a unified corpus, tracking what's been
processed and enabling cross-book queries.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


@dataclass
class BookInfo:
    """Information about a processed book."""
    id: str
    title: str
    author: str
    file_path: str
    
    # Processing status
    processed_at: Optional[str] = None
    
    # Stats
    total_words: int = 0
    total_passages: int = 0
    entity_count: int = 0
    relationship_count: int = 0
    dialogue_lines: int = 0
    character_profiles: int = 0
    
    # Style summary
    avg_sentence_length: float = 0.0
    flesch_kincaid_grade: float = 0.0
    
    # Optional metadata
    publication_year: Optional[int] = None
    series: Optional[str] = None
    series_order: Optional[int] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "BookInfo":
        return cls(**d)


@dataclass
class CorpusInfo:
    """Information about a corpus of books."""
    name: str
    author: str
    books: list[BookInfo] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Aggregate stats
    total_words: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['books'] = [b.to_dict() if hasattr(b, 'to_dict') else b for b in self.books]
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> "CorpusInfo":
        books = [BookInfo.from_dict(b) if isinstance(b, dict) else b for b in d.get('books', [])]
        d['books'] = books
        return cls(**d)


class CorpusManager:
    """
    Manages a corpus of books by the same author.
    
    Tracks processed books, maintains cross-book entity resolution,
    and provides unified query interface.
    
    Usage:
        manager = CorpusManager("tolkien_corpus")
        manager.add_book("The Hobbit", "data/texts/the_hobbit.txt")
        manager.add_book("Fellowship", "data/texts/fellowship.txt")
        manager.process_all()
    """
    
    def __init__(
        self,
        corpus_name: str,
        author: str = "Unknown",
        data_dir: str = "data/corpus",
    ):
        """
        Initialize corpus manager.
        
        Args:
            corpus_name: Name for this corpus
            author: Author name
            data_dir: Directory for corpus data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.corpus_file = self.data_dir / f"{corpus_name}.json"
        
        # Load or create corpus info
        if self.corpus_file.exists():
            self.corpus = self._load_corpus()
        else:
            self.corpus = CorpusInfo(name=corpus_name, author=author)
    
    def _load_corpus(self) -> CorpusInfo:
        """Load corpus info from file."""
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            return CorpusInfo.from_dict(json.load(f))
    
    def _save_corpus(self) -> None:
        """Save corpus info to file."""
        self.corpus.updated_at = datetime.now().isoformat()
        with open(self.corpus_file, 'w', encoding='utf-8') as f:
            json.dump(self.corpus.to_dict(), f, indent=2)
    
    def add_book(
        self,
        title: str,
        file_path: str,
        series: Optional[str] = None,
        series_order: Optional[int] = None,
        publication_year: Optional[int] = None,
    ) -> BookInfo:
        """
        Add a book to the corpus.
        
        Args:
            title: Book title
            file_path: Path to the text file
            series: Optional series name
            series_order: Optional order in series
            publication_year: Optional publication year
            
        Returns:
            BookInfo for the added book
        """
        book_id = title.lower().replace(" ", "_").replace("'", "")
        
        # Check if already exists
        for book in self.corpus.books:
            if book.id == book_id:
                return book  # Already added
        
        book = BookInfo(
            id=book_id,
            title=title,
            author=self.corpus.author,
            file_path=str(file_path),
            series=series,
            series_order=series_order,
            publication_year=publication_year,
        )
        
        self.corpus.books.append(book)
        self._save_corpus()
        
        return book
    
    def get_book(self, book_id: str) -> Optional[BookInfo]:
        """Get book by ID."""
        for book in self.corpus.books:
            if book.id == book_id:
                return book
        return None
    
    def list_books(self) -> list[BookInfo]:
        """List all books in corpus."""
        return self.corpus.books
    
    def update_book_stats(
        self,
        book_id: str,
        total_words: int = 0,
        total_passages: int = 0,
        entity_count: int = 0,
        relationship_count: int = 0,
        dialogue_lines: int = 0,
        character_profiles: int = 0,
        avg_sentence_length: float = 0.0,
        flesch_kincaid_grade: float = 0.0,
    ) -> None:
        """Update stats for a processed book."""
        book = self.get_book(book_id)
        if not book:
            return
        
        book.processed_at = datetime.now().isoformat()
        book.total_words = total_words
        book.total_passages = total_passages
        book.entity_count = entity_count
        book.relationship_count = relationship_count
        book.dialogue_lines = dialogue_lines
        book.character_profiles = character_profiles
        book.avg_sentence_length = avg_sentence_length
        book.flesch_kincaid_grade = flesch_kincaid_grade
        
        # Update corpus totals
        self._update_corpus_totals()
        self._save_corpus()
    
    def _update_corpus_totals(self) -> None:
        """Recalculate corpus totals from all books."""
        self.corpus.total_words = sum(b.total_words for b in self.corpus.books)
        self.corpus.total_entities = sum(b.entity_count for b in self.corpus.books)
        self.corpus.total_relationships = sum(b.relationship_count for b in self.corpus.books)
    
    def get_unprocessed_books(self) -> list[BookInfo]:
        """Get books that haven't been processed yet."""
        return [b for b in self.corpus.books if not b.processed_at]
    
    def get_processed_books(self) -> list[BookInfo]:
        """Get books that have been processed."""
        return [b for b in self.corpus.books if b.processed_at]
    
    def corpus_summary(self) -> str:
        """Generate corpus summary."""
        lines = [
            f"=== Corpus: {self.corpus.name} ===",
            f"Author: {self.corpus.author}",
            f"Books: {len(self.corpus.books)}",
            f"",
            f"[Aggregate Stats]",
            f"  Total words: {self.corpus.total_words:,}",
            f"  Total entities: {self.corpus.total_entities:,}",
            f"  Total relationships: {self.corpus.total_relationships:,}",
            f"",
            f"[Books]",
        ]
        
        for book in self.corpus.books:
            status = "OK" if book.processed_at else "pending"
            lines.append(f"  [{status}] {book.title}")
            if book.processed_at:
                lines.append(f"       {book.total_words:,} words, {book.entity_count} entities")
        
        return "\n".join(lines)
