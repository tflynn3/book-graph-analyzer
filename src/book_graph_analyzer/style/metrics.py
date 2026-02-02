"""
Stylometric Metrics

Sentence-level and vocabulary-level metrics for author style analysis.
Based on established stylometry research (Burrows' Delta, etc.)
"""

from dataclasses import dataclass, field
from typing import Optional
import re
import statistics
from collections import Counter

import spacy
from spacy.tokens import Doc, Span


# Top 100 English function words (style markers, not content)
FUNCTION_WORDS = [
    "the", "of", "and", "to", "a", "in", "that", "is", "was", "he",
    "for", "it", "with", "as", "his", "on", "be", "at", "by", "i",
    "this", "had", "not", "are", "but", "from", "or", "have", "an", "they",
    "which", "one", "you", "were", "all", "her", "she", "there", "would", "their",
    "we", "him", "been", "has", "when", "who", "will", "no", "more", "if",
    "out", "so", "up", "said", "what", "its", "about", "than", "into", "them",
    "can", "only", "other", "time", "new", "some", "could", "these", "two", "may",
    "first", "then", "do", "any", "like", "my", "now", "over", "such", "our",
    "man", "me", "even", "most", "made", "after", "also", "did", "many", "off",
    "before", "must", "well", "back", "through", "years", "much", "where", "your", "way"
]

# Common archaisms and archaic-style words (Tolkien-relevant)
ARCHAISMS = [
    "thee", "thou", "thy", "thine", "ye", "hath", "doth", "art", "wast", "wert",
    "wherefore", "hither", "thither", "whither", "hence", "thence", "whence",
    "ere", "nay", "aye", "yea", "behold", "lo", "alas", "forsooth", "prithee",
    "methinks", "mayhap", "perchance", "betwixt", "amongst", "whilst", "oft",
    "twas", "tis", "twere", "twould", "neath", "oer", "eer",
    "verily", "hark", "hearken", "tarry", "smite", "smote", "smitten",
    "slew", "slain", "wrought", "begat", "begotten", "dwelt", "spake",
]


@dataclass
class Distribution:
    """Statistical distribution of a metric."""
    mean: float
    std: float
    min: float
    max: float
    median: float
    p25: float  # 25th percentile
    p75: float  # 75th percentile
    count: int
    
    @classmethod
    def from_values(cls, values: list[float]) -> "Distribution":
        """Create distribution from a list of values."""
        if not values:
            return cls(0, 0, 0, 0, 0, 0, 0, 0)
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        return cls(
            mean=statistics.mean(values),
            std=statistics.stdev(values) if n > 1 else 0,
            min=min(values),
            max=max(values),
            median=statistics.median(values),
            p25=sorted_vals[n // 4] if n >= 4 else sorted_vals[0],
            p75=sorted_vals[3 * n // 4] if n >= 4 else sorted_vals[-1],
            count=n
        )


@dataclass
class SentenceMetrics:
    """Metrics for a single sentence."""
    text: str
    word_count: int
    char_count: int
    avg_word_length: float
    clause_depth: int
    voice: str  # "active", "passive", or "mixed"
    has_dialogue: bool
    is_question: bool
    is_exclamation: bool
    punctuation_count: int
    punctuation_density: float  # punctuation per word
    comma_count: int
    semicolon_count: int
    
    # POS distribution
    noun_count: int
    verb_count: int
    adj_count: int
    adv_count: int
    

@dataclass
class VocabularyProfile:
    """Vocabulary analysis for a text."""
    total_words: int
    unique_words: int
    type_token_ratio: float  # unique / total (lexical diversity)
    
    avg_word_length: float
    word_length_distribution: dict[int, int]  # length -> count
    
    hapax_count: int  # words appearing only once
    hapax_ratio: float  # hapax / unique
    
    function_word_frequencies: dict[str, float]  # word -> relative frequency
    
    archaism_count: int
    archaisms_found: list[str]
    
    # Unknown words (potential invented words or proper nouns)
    unknown_words: list[str]
    unknown_word_ratio: float


def calculate_sentence_metrics(sent: Span, nlp: spacy.Language) -> SentenceMetrics:
    """
    Calculate stylometric metrics for a single sentence.
    
    Args:
        sent: spaCy Span representing a sentence
        nlp: spaCy language model (for additional processing if needed)
        
    Returns:
        SentenceMetrics with all calculated values
    """
    text = sent.text.strip()
    
    # Basic counts
    words = [token for token in sent if not token.is_punct and not token.is_space]
    word_count = len(words)
    char_count = sum(len(token.text) for token in words)
    avg_word_length = char_count / word_count if word_count > 0 else 0
    
    # Punctuation analysis
    punct_tokens = [token for token in sent if token.is_punct]
    punctuation_count = len(punct_tokens)
    punctuation_density = punctuation_count / word_count if word_count > 0 else 0
    comma_count = sum(1 for t in punct_tokens if t.text == ",")
    semicolon_count = sum(1 for t in punct_tokens if t.text == ";")
    
    # Sentence type
    is_question = text.endswith("?")
    is_exclamation = text.endswith("!")
    
    # Dialogue detection (simple heuristic)
    has_dialogue = bool(re.search(r'["\u201c\u201d]', text) or 
                       re.search(r"(?:said|asked|replied|cried|whispered|shouted)", text.lower()))
    
    # Clause depth (approximate via dependency parsing)
    clause_depth = _calculate_clause_depth(sent)
    
    # Voice detection
    voice = _detect_voice(sent)
    
    # POS counts
    noun_count = sum(1 for t in sent if t.pos_ in ("NOUN", "PROPN"))
    verb_count = sum(1 for t in sent if t.pos_ == "VERB")
    adj_count = sum(1 for t in sent if t.pos_ == "ADJ")
    adv_count = sum(1 for t in sent if t.pos_ == "ADV")
    
    return SentenceMetrics(
        text=text,
        word_count=word_count,
        char_count=char_count,
        avg_word_length=avg_word_length,
        clause_depth=clause_depth,
        voice=voice,
        has_dialogue=has_dialogue,
        is_question=is_question,
        is_exclamation=is_exclamation,
        punctuation_count=punctuation_count,
        punctuation_density=punctuation_density,
        comma_count=comma_count,
        semicolon_count=semicolon_count,
        noun_count=noun_count,
        verb_count=verb_count,
        adj_count=adj_count,
        adv_count=adv_count,
    )


def _calculate_clause_depth(sent: Span) -> int:
    """
    Calculate clause depth based on dependency parsing.
    Counts maximum depth of subordinate clause markers.
    """
    max_depth = 0
    
    for token in sent:
        # Count depth by walking up the dependency tree
        depth = 0
        current = token
        while current.head != current:
            if current.dep_ in ("ccomp", "xcomp", "advcl", "relcl", "acl"):
                depth += 1
            current = current.head
        max_depth = max(max_depth, depth)
    
    return max_depth


def _detect_voice(sent: Span) -> str:
    """
    Detect whether sentence uses active, passive, or mixed voice.
    """
    passive_count = 0
    active_count = 0
    
    for token in sent:
        # Passive: auxiliary + past participle with passive subject
        if token.dep_ == "nsubjpass":
            passive_count += 1
        elif token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            active_count += 1
    
    if passive_count > 0 and active_count > 0:
        return "mixed"
    elif passive_count > 0:
        return "passive"
    else:
        return "active"


def calculate_vocabulary_profile(
    doc: Doc, 
    nlp: spacy.Language,
    unknown_threshold: int = 2
) -> VocabularyProfile:
    """
    Calculate vocabulary profile for an entire document.
    
    Args:
        doc: spaCy Doc object
        nlp: spaCy language model
        unknown_threshold: words appearing fewer times than this and not in vocab
                          are considered potentially invented
        
    Returns:
        VocabularyProfile with vocabulary statistics
    """
    # Get all words (excluding punctuation and spaces)
    words = [token.text.lower() for token in doc 
             if not token.is_punct and not token.is_space]
    
    total_words = len(words)
    word_counts = Counter(words)
    unique_words = len(word_counts)
    
    # Type-token ratio
    type_token_ratio = unique_words / total_words if total_words > 0 else 0
    
    # Word length analysis
    word_lengths = [len(w) for w in words]
    avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
    word_length_distribution = Counter(word_lengths)
    
    # Hapax legomena (words appearing only once)
    hapax = [w for w, c in word_counts.items() if c == 1]
    hapax_count = len(hapax)
    hapax_ratio = hapax_count / unique_words if unique_words > 0 else 0
    
    # Function word frequencies (normalized)
    function_word_frequencies = {}
    for fw in FUNCTION_WORDS:
        count = word_counts.get(fw, 0)
        function_word_frequencies[fw] = count / total_words if total_words > 0 else 0
    
    # Archaisms
    archaisms_found = [w for w in ARCHAISMS if w in word_counts]
    archaism_count = sum(word_counts.get(w, 0) for w in archaisms_found)
    
    # Unknown/invented words
    # Words not in spaCy vocab, not proper nouns, appearing rarely
    unknown_words = []
    for token in doc:
        if (not token.is_punct and not token.is_space and 
            not token.is_oov == False and  # is_oov = out of vocabulary
            token.pos_ != "PROPN" and
            word_counts.get(token.text.lower(), 0) <= unknown_threshold and
            len(token.text) > 2):
            if token.text.lower() not in unknown_words:
                unknown_words.append(token.text.lower())
    
    unknown_word_ratio = len(unknown_words) / unique_words if unique_words > 0 else 0
    
    return VocabularyProfile(
        total_words=total_words,
        unique_words=unique_words,
        type_token_ratio=type_token_ratio,
        avg_word_length=avg_word_length,
        word_length_distribution=dict(word_length_distribution),
        hapax_count=hapax_count,
        hapax_ratio=hapax_ratio,
        function_word_frequencies=function_word_frequencies,
        archaism_count=archaism_count,
        archaisms_found=archaisms_found,
        unknown_words=unknown_words[:100],  # Cap at 100 for practicality
        unknown_word_ratio=unknown_word_ratio,
    )


def calculate_readability(doc: Doc) -> dict[str, float]:
    """
    Calculate readability metrics for a document.
    
    Returns dict with:
    - flesch_reading_ease: Higher = easier (0-100 typical range)
    - flesch_kincaid_grade: US grade level
    - gunning_fog: Years of education needed
    """
    sentences = list(doc.sents)
    words = [token for token in doc if not token.is_punct and not token.is_space]
    
    if not sentences or not words:
        return {
            "flesch_reading_ease": 0,
            "flesch_kincaid_grade": 0,
            "gunning_fog": 0,
        }
    
    total_sentences = len(sentences)
    total_words = len(words)
    total_syllables = sum(_count_syllables(token.text) for token in words)
    
    # Complex words (3+ syllables)
    complex_words = sum(1 for token in words if _count_syllables(token.text) >= 3)
    
    # Average sentence length
    asl = total_words / total_sentences
    
    # Average syllables per word
    asw = total_syllables / total_words
    
    # Flesch Reading Ease
    # 206.835 - 1.015 * ASL - 84.6 * ASW
    flesch_reading_ease = 206.835 - (1.015 * asl) - (84.6 * asw)
    
    # Flesch-Kincaid Grade Level
    # 0.39 * ASL + 11.8 * ASW - 15.59
    flesch_kincaid_grade = (0.39 * asl) + (11.8 * asw) - 15.59
    
    # Gunning Fog Index
    # 0.4 * (ASL + percent complex words)
    percent_complex = (complex_words / total_words) * 100
    gunning_fog = 0.4 * (asl + percent_complex)
    
    return {
        "flesch_reading_ease": round(flesch_reading_ease, 2),
        "flesch_kincaid_grade": round(flesch_kincaid_grade, 2),
        "gunning_fog": round(gunning_fog, 2),
    }


def _count_syllables(word: str) -> int:
    """
    Estimate syllable count for a word.
    Uses a simple vowel-counting heuristic.
    """
    word = word.lower()
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    
    # Adjust for silent e
    if word.endswith("e") and count > 1:
        count -= 1
    
    # Adjust for -le endings
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        count += 1
    
    return max(1, count)
