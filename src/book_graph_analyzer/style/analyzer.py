"""
Style Analyzer

Main entry point for style analysis. Processes text and generates
an AuthorStyleFingerprint.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
import json

import spacy
from spacy.tokens import Doc

from .metrics import (
    SentenceMetrics, 
    VocabularyProfile,
    calculate_sentence_metrics,
    calculate_vocabulary_profile,
    calculate_readability,
)
from .classifier import (
    PassageType,
    PassageClassification,
    classify_passage,
)
from .fingerprint import AuthorStyleFingerprint


@dataclass
class AnalysisProgress:
    """Progress tracking for style analysis."""
    phase: str
    current: int
    total: int
    message: str = ""


class StyleAnalyzer:
    """
    Analyzes text to extract stylometric features.
    
    Usage:
        analyzer = StyleAnalyzer()
        fingerprint = analyzer.analyze_text(text, author_name="Tolkien")
        # or
        fingerprint = analyzer.analyze_file("path/to/book.txt")
    """
    
    def __init__(
        self, 
        spacy_model: str = "en_core_web_sm",
        progress_callback: Optional[Callable[[AnalysisProgress], None]] = None,
    ):
        """
        Initialize the style analyzer.
        
        Args:
            spacy_model: spaCy model to use for NLP
            progress_callback: Optional callback for progress updates
        """
        self.progress_callback = progress_callback
        self._nlp = None
        self._spacy_model = spacy_model
    
    @property
    def nlp(self) -> spacy.Language:
        """Lazy-load spaCy model."""
        if self._nlp is None:
            self._report_progress("loading", 0, 1, "Loading spaCy model...")
            self._nlp = spacy.load(self._spacy_model)
        return self._nlp
    
    def _report_progress(self, phase: str, current: int, total: int, message: str = ""):
        """Report progress via callback if set."""
        if self.progress_callback:
            self.progress_callback(AnalysisProgress(phase, current, total, message))
    
    def analyze_text(
        self,
        text: str,
        author_name: str = "Unknown",
        source_name: str = "text",
        chunk_size: int = 100000,
    ) -> AuthorStyleFingerprint:
        """
        Analyze a text and return its style fingerprint.
        
        Args:
            text: The full text to analyze
            author_name: Name of the author
            source_name: Name of the source text
            chunk_size: Process text in chunks of this many chars (for memory)
            
        Returns:
            AuthorStyleFingerprint containing all stylometric features
        """
        self._report_progress("parsing", 0, 1, "Parsing text with spaCy...")
        
        # For very long texts, process in chunks
        if len(text) > chunk_size:
            return self._analyze_long_text(text, author_name, source_name, chunk_size)
        
        # Parse with spaCy
        doc = self.nlp(text)
        
        # Analyze
        return self._analyze_doc(doc, author_name, [source_name])
    
    def _analyze_long_text(
        self,
        text: str,
        author_name: str,
        source_name: str,
        chunk_size: int,
    ) -> AuthorStyleFingerprint:
        """Analyze a long text in chunks and aggregate results."""
        
        # Split into chunks at paragraph boundaries
        chunks = self._split_into_chunks(text, chunk_size)
        total_chunks = len(chunks)
        
        all_sentence_metrics = []
        all_classifications = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            self._report_progress(
                "parsing", i + 1, total_chunks, 
                f"Processing chunk {i + 1}/{total_chunks}..."
            )
            
            doc = self.nlp(chunk)
            
            # Collect sentence metrics
            for sent in doc.sents:
                if len(sent.text.strip()) > 0:
                    metrics = calculate_sentence_metrics(sent, self.nlp)
                    all_sentence_metrics.append(metrics)
            
            # Classify passages (use paragraphs as passages)
            paragraphs = chunk.split('\n\n')
            for para in paragraphs:
                if len(para.strip()) > 20:
                    classification = classify_passage(para)
                    all_classifications.append(classification)
        
        # Calculate vocabulary profile on full text
        self._report_progress("vocabulary", 0, 1, "Analyzing vocabulary...")
        full_doc = self.nlp(text[:500000])  # Cap for memory
        vocabulary_profile = calculate_vocabulary_profile(full_doc, self.nlp)
        
        # Calculate readability
        readability = calculate_readability(full_doc)
        
        # Create fingerprint
        return AuthorStyleFingerprint.from_analysis(
            sentence_metrics=all_sentence_metrics,
            vocabulary_profile=vocabulary_profile,
            passage_classifications=all_classifications,
            readability=readability,
            author_name=author_name,
            source_texts=[source_name],
        )
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> list[str]:
        """Split text into chunks at paragraph boundaries."""
        chunks = []
        current_chunk = []
        current_size = 0
        
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += para_size
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _analyze_doc(
        self,
        doc: Doc,
        author_name: str,
        source_texts: list[str],
    ) -> AuthorStyleFingerprint:
        """Analyze a spaCy Doc and return fingerprint."""
        
        # Sentence-level metrics
        self._report_progress("sentences", 0, 1, "Analyzing sentence structure...")
        sentence_metrics = []
        sentences = list(doc.sents)
        
        for i, sent in enumerate(sentences):
            if len(sent.text.strip()) > 0:
                metrics = calculate_sentence_metrics(sent, self.nlp)
                sentence_metrics.append(metrics)
            
            if i % 100 == 0:
                self._report_progress("sentences", i, len(sentences))
        
        # Vocabulary analysis
        self._report_progress("vocabulary", 0, 1, "Analyzing vocabulary...")
        vocabulary_profile = calculate_vocabulary_profile(doc, self.nlp)
        
        # Passage classification
        self._report_progress("classification", 0, 1, "Classifying passages...")
        classifications = []
        
        # Use paragraphs as passages
        text = doc.text
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            if len(para.strip()) > 20:
                classification = classify_passage(para)
                classifications.append(classification)
        
        # Readability
        self._report_progress("readability", 0, 1, "Calculating readability...")
        readability = calculate_readability(doc)
        
        # Create fingerprint
        self._report_progress("fingerprint", 0, 1, "Creating fingerprint...")
        fingerprint = AuthorStyleFingerprint.from_analysis(
            sentence_metrics=sentence_metrics,
            vocabulary_profile=vocabulary_profile,
            passage_classifications=classifications,
            readability=readability,
            author_name=author_name,
            source_texts=source_texts,
        )
        
        self._report_progress("complete", 1, 1, "Analysis complete!")
        return fingerprint
    
    def analyze_file(
        self,
        file_path: str | Path,
        author_name: str = "Unknown",
        encoding: str = "utf-8",
    ) -> AuthorStyleFingerprint:
        """
        Analyze a text file and return its style fingerprint.
        
        Args:
            file_path: Path to the text file
            author_name: Name of the author
            encoding: File encoding
            
        Returns:
            AuthorStyleFingerprint
        """
        path = Path(file_path)
        
        self._report_progress("loading", 0, 1, f"Loading {path.name}...")
        
        with open(path, 'r', encoding=encoding) as f:
            text = f.read()
        
        return self.analyze_text(text, author_name, path.name)
    
    def analyze_files(
        self,
        file_paths: list[str | Path],
        author_name: str = "Unknown",
        encoding: str = "utf-8",
    ) -> AuthorStyleFingerprint:
        """
        Analyze multiple text files and return an aggregated fingerprint.
        
        Args:
            file_paths: List of paths to text files
            author_name: Name of the author
            encoding: File encoding
            
        Returns:
            AuthorStyleFingerprint aggregated from all files
        """
        all_sentence_metrics = []
        all_classifications = []
        source_texts = []
        full_text_parts = []
        
        for i, fp in enumerate(file_paths):
            path = Path(fp)
            source_texts.append(path.name)
            
            self._report_progress(
                "files", i + 1, len(file_paths),
                f"Processing {path.name}..."
            )
            
            with open(path, 'r', encoding=encoding) as f:
                text = f.read()
            
            full_text_parts.append(text)
            
            # Parse and analyze
            doc = self.nlp(text)
            
            for sent in doc.sents:
                if len(sent.text.strip()) > 0:
                    metrics = calculate_sentence_metrics(sent, self.nlp)
                    all_sentence_metrics.append(metrics)
            
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                if len(para.strip()) > 20:
                    classification = classify_passage(para)
                    all_classifications.append(classification)
        
        # Calculate vocabulary on combined text (capped for memory)
        self._report_progress("vocabulary", 0, 1, "Analyzing combined vocabulary...")
        combined_text = '\n\n'.join(full_text_parts)[:500000]
        full_doc = self.nlp(combined_text)
        vocabulary_profile = calculate_vocabulary_profile(full_doc, self.nlp)
        readability = calculate_readability(full_doc)
        
        return AuthorStyleFingerprint.from_analysis(
            sentence_metrics=all_sentence_metrics,
            vocabulary_profile=vocabulary_profile,
            passage_classifications=all_classifications,
            readability=readability,
            author_name=author_name,
            source_texts=source_texts,
        )
    
    def compare(
        self,
        fingerprint1: AuthorStyleFingerprint,
        fingerprint2: AuthorStyleFingerprint,
    ) -> dict:
        """
        Compare two fingerprints and return detailed comparison.
        
        Args:
            fingerprint1: First fingerprint
            fingerprint2: Second fingerprint
            
        Returns:
            Dict with comparison metrics
        """
        delta = fingerprint1.burrows_delta(fingerprint2)
        similarity = fingerprint1.similarity_score(fingerprint2)
        
        comparison = {
            "author1": fingerprint1.author_name,
            "author2": fingerprint2.author_name,
            "burrows_delta": delta,
            "similarity_score": similarity,
            "interpretation": self._interpret_delta(delta),
            "details": {
                "sentence_length": {
                    "author1_mean": fingerprint1.sentence_length_dist.mean if fingerprint1.sentence_length_dist else 0,
                    "author2_mean": fingerprint2.sentence_length_dist.mean if fingerprint2.sentence_length_dist else 0,
                },
                "readability": {
                    "author1_fk_grade": fingerprint1.flesch_kincaid_grade,
                    "author2_fk_grade": fingerprint2.flesch_kincaid_grade,
                },
                "dialogue_ratio": {
                    "author1": fingerprint1.dialogue_ratio,
                    "author2": fingerprint2.dialogue_ratio,
                },
            }
        }
        
        return comparison
    
    def _interpret_delta(self, delta: float) -> str:
        """Interpret Burrows' Delta value."""
        if delta == float('inf'):
            return "Cannot compare (insufficient data)"
        elif delta < 0.5:
            return "Very similar styles (possibly same author)"
        elif delta < 1.0:
            return "Similar styles"
        elif delta < 1.5:
            return "Different but related styles"
        else:
            return "Very different styles"
    
    def save_fingerprint(
        self,
        fingerprint: AuthorStyleFingerprint,
        output_path: str | Path,
    ):
        """Save fingerprint to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(fingerprint.to_json())
    
    def load_fingerprint(self, input_path: str | Path) -> AuthorStyleFingerprint:
        """Load fingerprint from JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            return AuthorStyleFingerprint.from_json(f.read())
