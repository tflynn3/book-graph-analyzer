"""Text ingestion and processing."""

from book_graph_analyzer.ingest.loader import load_book
from book_graph_analyzer.ingest.splitter import split_into_passages

__all__ = ["load_book", "split_into_passages"]
