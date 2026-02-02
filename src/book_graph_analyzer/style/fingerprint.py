"""
Author Style Fingerprint

Aggregate stylometric features into a comprehensive author profile
that can be used for comparison and generation guidance.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import statistics
from collections import Counter

from .metrics import Distribution, VocabularyProfile, SentenceMetrics, FUNCTION_WORDS
from .classifier import PassageType, PassageClassification


@dataclass
class AuthorStyleFingerprint:
    """
    Comprehensive author style profile.
    
    Contains all the stylometric features needed to:
    1. Compare this author's style to others (via Burrows' Delta)
    2. Characterize the author's voice for generation
    3. Validate generated content against the author's style
    """
    
    # Metadata
    author_name: str = "Unknown"
    source_texts: list[str] = field(default_factory=list)
    total_word_count: int = 0
    total_sentence_count: int = 0
    
    # === Sentence-Level Distributions ===
    sentence_length_dist: Optional[Distribution] = None
    word_length_dist: Optional[Distribution] = None
    clause_depth_dist: Optional[Distribution] = None
    punctuation_density_dist: Optional[Distribution] = None
    
    # === Ratios and Percentages ===
    dialogue_ratio: float = 0.0          # % of sentences with dialogue
    passive_voice_ratio: float = 0.0     # % of passive voice sentences
    question_ratio: float = 0.0          # % of questions
    exclamation_ratio: float = 0.0       # % of exclamations
    
    # POS ratios (per sentence average)
    noun_ratio: float = 0.0
    verb_ratio: float = 0.0
    adjective_ratio: float = 0.0
    adverb_ratio: float = 0.0
    
    # === Vocabulary Profile ===
    vocabulary_profile: Optional[VocabularyProfile] = None
    
    # === Function Word Signature (for Burrows' Delta) ===
    # Z-scores for top function words
    function_word_zscores: dict[str, float] = field(default_factory=dict)
    # Raw frequencies for reference
    function_word_frequencies: dict[str, float] = field(default_factory=dict)
    
    # === Passage Type Distribution ===
    passage_type_distribution: dict[str, float] = field(default_factory=dict)
    
    # === Readability Metrics ===
    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0
    gunning_fog: float = 0.0
    
    # === Special Features ===
    archaism_density: float = 0.0  # archaisms per 1000 words
    invented_word_density: float = 0.0  # potential invented words per 1000 words
    
    @classmethod
    def from_analysis(
        cls,
        sentence_metrics: list[SentenceMetrics],
        vocabulary_profile: VocabularyProfile,
        passage_classifications: list[PassageClassification],
        readability: dict[str, float],
        author_name: str = "Unknown",
        source_texts: list[str] = None,
    ) -> "AuthorStyleFingerprint":
        """
        Create a fingerprint from analyzed data.
        
        Args:
            sentence_metrics: List of metrics for each sentence
            vocabulary_profile: Vocabulary analysis for the full text
            passage_classifications: Classification for each passage
            readability: Readability metrics dict
            author_name: Author name for identification
            source_texts: List of source text filenames
            
        Returns:
            AuthorStyleFingerprint aggregated from the data
        """
        fp = cls()
        fp.author_name = author_name
        fp.source_texts = source_texts or []
        
        if not sentence_metrics:
            return fp
        
        # Aggregate sentence-level metrics
        fp.total_sentence_count = len(sentence_metrics)
        fp.total_word_count = sum(s.word_count for s in sentence_metrics)
        
        # Distributions
        fp.sentence_length_dist = Distribution.from_values(
            [s.word_count for s in sentence_metrics]
        )
        fp.word_length_dist = Distribution.from_values(
            [s.avg_word_length for s in sentence_metrics]
        )
        fp.clause_depth_dist = Distribution.from_values(
            [s.clause_depth for s in sentence_metrics]
        )
        fp.punctuation_density_dist = Distribution.from_values(
            [s.punctuation_density for s in sentence_metrics]
        )
        
        # Ratios
        fp.dialogue_ratio = sum(1 for s in sentence_metrics if s.has_dialogue) / len(sentence_metrics)
        fp.passive_voice_ratio = sum(1 for s in sentence_metrics if s.voice == "passive") / len(sentence_metrics)
        fp.question_ratio = sum(1 for s in sentence_metrics if s.is_question) / len(sentence_metrics)
        fp.exclamation_ratio = sum(1 for s in sentence_metrics if s.is_exclamation) / len(sentence_metrics)
        
        # POS ratios (average per sentence)
        total_words = sum(s.word_count for s in sentence_metrics) or 1
        fp.noun_ratio = sum(s.noun_count for s in sentence_metrics) / total_words
        fp.verb_ratio = sum(s.verb_count for s in sentence_metrics) / total_words
        fp.adjective_ratio = sum(s.adj_count for s in sentence_metrics) / total_words
        fp.adverb_ratio = sum(s.adv_count for s in sentence_metrics) / total_words
        
        # Vocabulary profile
        fp.vocabulary_profile = vocabulary_profile
        
        # Function word frequencies from vocabulary profile
        if vocabulary_profile:
            fp.function_word_frequencies = vocabulary_profile.function_word_frequencies.copy()
            
            # Calculate z-scores relative to a baseline
            # (For now, use the author's own frequencies; cross-author comparison
            # would use a corpus-wide baseline)
            freqs = list(vocabulary_profile.function_word_frequencies.values())
            if freqs:
                mean_freq = statistics.mean(freqs)
                std_freq = statistics.stdev(freqs) if len(freqs) > 1 else 1
                std_freq = std_freq if std_freq > 0 else 1
                
                for word, freq in vocabulary_profile.function_word_frequencies.items():
                    fp.function_word_zscores[word] = (freq - mean_freq) / std_freq
        
        # Passage type distribution
        if passage_classifications:
            type_counts = Counter(pc.primary_type.value for pc in passage_classifications)
            total_passages = len(passage_classifications)
            fp.passage_type_distribution = {
                pt: count / total_passages 
                for pt, count in type_counts.items()
            }
        
        # Readability
        fp.flesch_reading_ease = readability.get("flesch_reading_ease", 0)
        fp.flesch_kincaid_grade = readability.get("flesch_kincaid_grade", 0)
        fp.gunning_fog = readability.get("gunning_fog", 0)
        
        # Special features
        if vocabulary_profile and vocabulary_profile.total_words > 0:
            fp.archaism_density = (vocabulary_profile.archaism_count / vocabulary_profile.total_words) * 1000
            fp.invented_word_density = (len(vocabulary_profile.unknown_words) / vocabulary_profile.total_words) * 1000
        
        return fp
    
    def burrows_delta(self, other: "AuthorStyleFingerprint") -> float:
        """
        Calculate Burrows' Delta distance to another fingerprint.
        
        Lower values indicate more similar styles.
        
        Args:
            other: Another AuthorStyleFingerprint to compare
            
        Returns:
            Delta value (average absolute z-score difference)
        """
        if not self.function_word_zscores or not other.function_word_zscores:
            return float('inf')
        
        # Use common function words
        common_words = set(self.function_word_zscores.keys()) & set(other.function_word_zscores.keys())
        
        if not common_words:
            return float('inf')
        
        # Calculate average absolute difference
        total_diff = sum(
            abs(self.function_word_zscores[w] - other.function_word_zscores[w])
            for w in common_words
        )
        
        return total_diff / len(common_words)
    
    def similarity_score(self, other: "AuthorStyleFingerprint") -> float:
        """
        Calculate overall similarity score (0-1, higher = more similar).
        
        Combines multiple metrics:
        - Burrows' Delta (function words)
        - Sentence length distribution
        - Readability metrics
        - Passage type distribution
        """
        scores = []
        
        # 1. Burrows' Delta (inverted and normalized)
        delta = self.burrows_delta(other)
        if delta != float('inf'):
            # Typical delta range is 0-2, map to 0-1 similarity
            delta_sim = max(0, 1 - delta / 2)
            scores.append(delta_sim * 2)  # Weight x2
        
        # 2. Sentence length similarity
        if self.sentence_length_dist and other.sentence_length_dist:
            mean_diff = abs(self.sentence_length_dist.mean - other.sentence_length_dist.mean)
            # Typical mean sentence length 10-30, normalize
            sent_sim = max(0, 1 - mean_diff / 20)
            scores.append(sent_sim)
        
        # 3. Readability similarity
        fk_diff = abs(self.flesch_kincaid_grade - other.flesch_kincaid_grade)
        # Typical grade range 4-16, normalize
        read_sim = max(0, 1 - fk_diff / 12)
        scores.append(read_sim)
        
        # 4. Passage type distribution similarity (cosine-like)
        if self.passage_type_distribution and other.passage_type_distribution:
            all_types = set(self.passage_type_distribution.keys()) | set(other.passage_type_distribution.keys())
            dot_product = sum(
                self.passage_type_distribution.get(t, 0) * other.passage_type_distribution.get(t, 0)
                for t in all_types
            )
            scores.append(dot_product)  # Already 0-1 for normalized distributions
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        d = {}
        
        # Simple fields
        d['author_name'] = self.author_name
        d['source_texts'] = self.source_texts
        d['total_word_count'] = self.total_word_count
        d['total_sentence_count'] = self.total_sentence_count
        
        # Distribution fields
        for key in ['sentence_length_dist', 'word_length_dist', 'clause_depth_dist', 'punctuation_density_dist']:
            val = getattr(self, key)
            if val is not None:
                d[key] = asdict(val)
            else:
                d[key] = None
        
        # Ratio fields
        d['dialogue_ratio'] = self.dialogue_ratio
        d['passive_voice_ratio'] = self.passive_voice_ratio
        d['question_ratio'] = self.question_ratio
        d['exclamation_ratio'] = self.exclamation_ratio
        d['noun_ratio'] = self.noun_ratio
        d['verb_ratio'] = self.verb_ratio
        d['adjective_ratio'] = self.adjective_ratio
        d['adverb_ratio'] = self.adverb_ratio
        
        # Vocabulary profile
        if self.vocabulary_profile:
            d['vocabulary_profile'] = asdict(self.vocabulary_profile)
        else:
            d['vocabulary_profile'] = None
        
        # Dict fields
        d['function_word_zscores'] = self.function_word_zscores
        d['function_word_frequencies'] = self.function_word_frequencies
        d['passage_type_distribution'] = self.passage_type_distribution
        
        # Readability
        d['flesch_reading_ease'] = self.flesch_reading_ease
        d['flesch_kincaid_grade'] = self.flesch_kincaid_grade
        d['gunning_fog'] = self.gunning_fog
        
        # Special features
        d['archaism_density'] = self.archaism_density
        d['invented_word_density'] = self.invented_word_density
        
        return d
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, d: dict) -> "AuthorStyleFingerprint":
        """Create from dictionary."""
        fp = cls()
        
        # Simple fields
        for key in ['author_name', 'total_word_count', 'total_sentence_count',
                    'dialogue_ratio', 'passive_voice_ratio', 'question_ratio',
                    'exclamation_ratio', 'noun_ratio', 'verb_ratio',
                    'adjective_ratio', 'adverb_ratio', 'flesch_reading_ease',
                    'flesch_kincaid_grade', 'gunning_fog', 'archaism_density',
                    'invented_word_density']:
            if key in d:
                setattr(fp, key, d[key])
        
        # Lists
        fp.source_texts = d.get('source_texts', [])
        
        # Dicts
        fp.function_word_zscores = d.get('function_word_zscores', {})
        fp.function_word_frequencies = d.get('function_word_frequencies', {})
        fp.passage_type_distribution = d.get('passage_type_distribution', {})
        
        # Distributions
        for key in ['sentence_length_dist', 'word_length_dist', 'clause_depth_dist', 'punctuation_density_dist']:
            if d.get(key):
                setattr(fp, key, Distribution(**d[key]))
        
        # Vocabulary profile
        if d.get('vocabulary_profile'):
            fp.vocabulary_profile = VocabularyProfile(**d['vocabulary_profile'])
        
        return fp
    
    @classmethod
    def from_json(cls, json_str: str) -> "AuthorStyleFingerprint":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def summary(self) -> str:
        """Generate a human-readable summary of the fingerprint."""
        lines = [
            f"=== Author Style Fingerprint: {self.author_name} ===",
            f"",
            f"[Corpus Statistics]",
            f"   Total words: {self.total_word_count:,}",
            f"   Total sentences: {self.total_sentence_count:,}",
            f"   Source texts: {len(self.source_texts)}",
            f"",
        ]
        
        if self.sentence_length_dist:
            lines.extend([
                f"[Sentence Structure]",
                f"   Avg sentence length: {self.sentence_length_dist.mean:.1f} words",
                f"   Sentence length range: {self.sentence_length_dist.min:.0f} - {self.sentence_length_dist.max:.0f}",
                f"   Avg clause depth: {self.clause_depth_dist.mean:.2f}" if self.clause_depth_dist else "",
                f"",
            ])
        
        lines.extend([
            f"[Style Ratios]",
            f"   Dialogue: {self.dialogue_ratio*100:.1f}%",
            f"   Passive voice: {self.passive_voice_ratio*100:.1f}%",
            f"   Questions: {self.question_ratio*100:.1f}%",
            f"   Exclamations: {self.exclamation_ratio*100:.1f}%",
            f"",
            f"[Readability]",
            f"   Flesch Reading Ease: {self.flesch_reading_ease:.1f}",
            f"   Flesch-Kincaid Grade: {self.flesch_kincaid_grade:.1f}",
            f"   Gunning Fog Index: {self.gunning_fog:.1f}",
            f"",
        ])
        
        if self.vocabulary_profile:
            vp = self.vocabulary_profile
            lines.extend([
                f"[Vocabulary]",
                f"   Unique words: {vp.unique_words:,}",
                f"   Type-token ratio: {vp.type_token_ratio:.3f}",
                f"   Avg word length: {vp.avg_word_length:.2f} chars",
                f"   Hapax legomena: {vp.hapax_count:,} ({vp.hapax_ratio*100:.1f}%)",
                f"",
            ])
            
            if vp.archaisms_found:
                lines.extend([
                    f"[Archaic Language]",
                    f"   Archaisms found: {', '.join(vp.archaisms_found[:10])}{'...' if len(vp.archaisms_found) > 10 else ''}",
                    f"   Archaism density: {self.archaism_density:.2f} per 1000 words",
                    f"",
                ])
        
        if self.passage_type_distribution:
            lines.extend([
                f"[Scene Types]",
            ])
            for ptype, ratio in sorted(self.passage_type_distribution.items(), key=lambda x: -x[1]):
                lines.append(f"   {ptype}: {ratio*100:.1f}%")
        
        return "\n".join(lines)
