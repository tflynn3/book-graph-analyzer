"""
Character Voice Profile

Captures the distinctive speech patterns of a character.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import statistics
from collections import Counter


@dataclass
class CharacterVoiceProfile:
    """
    Voice profile for a single character.
    
    Captures how a character speaks: vocabulary, formality,
    sentence patterns, and distinctive phrases.
    """
    
    # Identity
    character_name: str
    character_id: Optional[str] = None  # Canonical entity ID
    
    # Corpus stats
    total_lines: int = 0
    total_words: int = 0
    total_chars: int = 0
    
    # Utterance metrics
    avg_utterance_length: float = 0.0      # Words per line
    utterance_length_std: float = 0.0
    min_utterance_length: int = 0
    max_utterance_length: int = 0
    
    # Dialogue type distribution
    question_ratio: float = 0.0            # % of lines that are questions
    exclamation_ratio: float = 0.0         # % of lines that are exclamations
    statement_ratio: float = 0.0           # % of lines that are statements
    
    # Vocabulary metrics
    unique_words: int = 0
    type_token_ratio: float = 0.0          # Lexical diversity
    avg_word_length: float = 0.0
    
    # Formality indicators
    contraction_ratio: float = 0.0         # Use of contractions (informal)
    first_person_ratio: float = 0.0        # "I", "me", "my" usage
    second_person_ratio: float = 0.0       # "you", "your" usage
    
    # Distinctive features
    top_words: list[tuple[str, int]] = field(default_factory=list)  # Most used words
    distinctive_words: list[str] = field(default_factory=list)       # Words unique to this character
    signature_phrases: list[str] = field(default_factory=list)       # Repeated phrases
    
    # Archaic language (Tolkien-relevant)
    archaism_count: int = 0
    archaisms_used: list[str] = field(default_factory=list)
    
    # Sample quotes
    sample_quotes: list[str] = field(default_factory=list)  # Representative quotes
    
    @classmethod
    def from_dialogue_lines(
        cls,
        character_name: str,
        lines: list,  # List of DialogueLine objects
        character_id: Optional[str] = None,
        all_character_words: Optional[dict[str, Counter]] = None,  # For distinctiveness
    ) -> "CharacterVoiceProfile":
        """
        Build a voice profile from dialogue lines.
        
        Args:
            character_name: Name of the character
            lines: List of DialogueLine objects
            character_id: Optional canonical ID
            all_character_words: Word counts for all characters (to find distinctive words)
        """
        profile = cls(
            character_name=character_name,
            character_id=character_id,
        )
        
        if not lines:
            return profile
        
        # Basic counts
        profile.total_lines = len(lines)
        
        # Word analysis
        all_words = []
        utterance_lengths = []
        word_lengths = []
        
        questions = 0
        exclamations = 0
        statements = 0
        
        contractions = 0
        first_person = 0
        second_person = 0
        
        first_person_words = {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'll", "i'd"}
        second_person_words = {'you', 'your', 'yours', 'yourself', "you're", "you've", "you'll", "you'd"}
        contraction_patterns = ["n't", "'s", "'re", "'ve", "'ll", "'d", "'m"]
        
        archaisms_list = [
            "thee", "thou", "thy", "thine", "ye", "hath", "doth", "art", "wast",
            "wherefore", "hither", "thither", "whither", "hence", "thence",
            "ere", "nay", "aye", "yea", "behold", "lo", "alas", "forsooth",
            "methinks", "mayhap", "perchance", "betwixt", "amongst", "whilst",
            "verily", "hark", "hearken", "tarry", "prithee",
        ]
        archaisms_found = set()
        
        for line in lines:
            text = line.text
            profile.total_chars += len(text)
            
            # Tokenize simply
            words = text.lower().split()
            word_count = len(words)
            all_words.extend(words)
            utterance_lengths.append(word_count)
            
            # Word lengths
            word_lengths.extend(len(w.strip('.,!?"\'-')) for w in words)
            
            # Classify
            if line.is_question:
                questions += 1
            elif line.is_exclamation:
                exclamations += 1
            else:
                statements += 1
            
            # Formality indicators
            for word in words:
                word_lower = word.lower().strip('.,!?"\'')
                if word_lower in first_person_words:
                    first_person += 1
                if word_lower in second_person_words:
                    second_person += 1
                for pattern in contraction_patterns:
                    if pattern in word:
                        contractions += 1
                        break
                if word_lower in archaisms_list:
                    archaisms_found.add(word_lower)
        
        profile.total_words = len(all_words)
        
        # Utterance length stats
        if utterance_lengths:
            profile.avg_utterance_length = statistics.mean(utterance_lengths)
            profile.utterance_length_std = statistics.stdev(utterance_lengths) if len(utterance_lengths) > 1 else 0
            profile.min_utterance_length = min(utterance_lengths)
            profile.max_utterance_length = max(utterance_lengths)
        
        # Type ratios
        if profile.total_lines > 0:
            profile.question_ratio = questions / profile.total_lines
            profile.exclamation_ratio = exclamations / profile.total_lines
            profile.statement_ratio = statements / profile.total_lines
        
        # Vocabulary
        word_counts = Counter(all_words)
        profile.unique_words = len(word_counts)
        profile.type_token_ratio = profile.unique_words / profile.total_words if profile.total_words > 0 else 0
        profile.avg_word_length = statistics.mean(word_lengths) if word_lengths else 0
        
        # Formality ratios
        if profile.total_words > 0:
            profile.contraction_ratio = contractions / profile.total_words
            profile.first_person_ratio = first_person / profile.total_words
            profile.second_person_ratio = second_person / profile.total_words
        
        # Top words (filter out very common words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'shall', 'can', 'that', 'this',
                     'it', 'its', 'as', 'if', 'not', 'no', 'so', 'up', 'out', 'about'}
        
        filtered_counts = {w: c for w, c in word_counts.items() 
                         if w not in stop_words and len(w) > 2}
        profile.top_words = sorted(filtered_counts.items(), key=lambda x: -x[1])[:20]
        
        # Distinctive words (words this character uses more than others)
        if all_character_words:
            profile.distinctive_words = _find_distinctive_words(
                character_name, word_counts, all_character_words
            )
        
        # Archaisms
        profile.archaism_count = sum(word_counts.get(a, 0) for a in archaisms_found)
        profile.archaisms_used = list(archaisms_found)
        
        # Sample quotes (pick diverse ones)
        profile.sample_quotes = _select_sample_quotes(lines, max_quotes=5)
        
        # Signature phrases (repeated sequences)
        profile.signature_phrases = _find_signature_phrases(lines)
        
        return profile
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, d: dict) -> "CharacterVoiceProfile":
        """Create from dictionary."""
        # Convert top_words back to list of tuples
        if 'top_words' in d:
            d['top_words'] = [tuple(x) if isinstance(x, list) else x for x in d['top_words']]
        return cls(**d)
    
    @classmethod
    def from_json(cls, json_str: str) -> "CharacterVoiceProfile":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== Voice Profile: {self.character_name} ===",
            f"",
            f"[Corpus]",
            f"   Total lines: {self.total_lines}",
            f"   Total words: {self.total_words}",
            f"",
            f"[Speech Patterns]",
            f"   Avg utterance: {self.avg_utterance_length:.1f} words",
            f"   Range: {self.min_utterance_length} - {self.max_utterance_length} words",
            f"   Questions: {self.question_ratio*100:.1f}%",
            f"   Exclamations: {self.exclamation_ratio*100:.1f}%",
            f"",
            f"[Vocabulary]",
            f"   Unique words: {self.unique_words}",
            f"   Type-token ratio: {self.type_token_ratio:.3f}",
            f"   Contractions: {self.contraction_ratio*100:.1f}%",
            f"",
        ]
        
        if self.top_words:
            lines.append(f"[Top Words]")
            for word, count in self.top_words[:10]:
                lines.append(f"   {word}: {count}")
            lines.append("")
        
        if self.distinctive_words:
            lines.append(f"[Distinctive Words]")
            lines.append(f"   {', '.join(self.distinctive_words[:10])}")
            lines.append("")
        
        if self.archaisms_used:
            lines.append(f"[Archaic Language]")
            lines.append(f"   {', '.join(self.archaisms_used)}")
            lines.append("")
        
        if self.sample_quotes:
            lines.append(f"[Sample Quotes]")
            for quote in self.sample_quotes[:3]:
                # Truncate long quotes
                display = quote[:80] + "..." if len(quote) > 80 else quote
                lines.append(f'   "{display}"')
            lines.append("")
        
        return "\n".join(lines)


def _find_distinctive_words(
    character: str,
    char_words: Counter,
    all_char_words: dict[str, Counter],
    min_count: int = 2,
    top_n: int = 10,
) -> list[str]:
    """
    Find words this character uses more frequently than others.
    
    Uses TF-IDF-like scoring: high frequency for this character,
    low frequency for other characters.
    """
    distinctive = []
    
    # Calculate total words per character
    char_totals = {c: sum(words.values()) for c, words in all_char_words.items()}
    total_chars = len(all_char_words)
    
    for word, count in char_words.items():
        if count < min_count:
            continue
        
        # How many other characters use this word?
        other_usage = sum(1 for c, words in all_char_words.items() 
                        if c != character and words.get(word, 0) > 0)
        
        # Score: frequency in this character / (1 + other characters using it)
        char_freq = count / char_totals.get(character, 1)
        distinctiveness = char_freq / (1 + other_usage / total_chars)
        
        distinctive.append((word, distinctiveness, count))
    
    # Sort by distinctiveness score
    distinctive.sort(key=lambda x: -x[1])
    
    return [w for w, _, _ in distinctive[:top_n]]


def _select_sample_quotes(lines: list, max_quotes: int = 5) -> list[str]:
    """Select diverse sample quotes."""
    if not lines:
        return []
    
    quotes = []
    
    # Get one question, one exclamation, and some statements
    questions = [l for l in lines if l.is_question]
    exclamations = [l for l in lines if l.is_exclamation]
    statements = [l for l in lines if l.is_statement]
    
    # Pick medium-length quotes (not too short, not too long)
    def quality_score(line):
        length = len(line.text.split())
        # Prefer 5-20 words
        if 5 <= length <= 20:
            return 1.0
        elif 3 <= length <= 30:
            return 0.5
        else:
            return 0.1
    
    # Sort each category by quality
    questions.sort(key=lambda x: -quality_score(x))
    exclamations.sort(key=lambda x: -quality_score(x))
    statements.sort(key=lambda x: -quality_score(x))
    
    # Pick from each category
    if questions:
        quotes.append(questions[0].text)
    if exclamations:
        quotes.append(exclamations[0].text)
    
    # Fill rest with statements
    for stmt in statements:
        if len(quotes) >= max_quotes:
            break
        if stmt.text not in quotes:
            quotes.append(stmt.text)
    
    return quotes


def _find_signature_phrases(lines: list, min_occurrences: int = 2) -> list[str]:
    """Find phrases this character repeats."""
    # Look for 2-4 word sequences that appear multiple times
    ngram_counts = Counter()
    
    for line in lines:
        words = line.text.lower().split()
        
        # 2-grams
        for i in range(len(words) - 1):
            ngram = ' '.join(words[i:i+2])
            ngram_counts[ngram] += 1
        
        # 3-grams
        for i in range(len(words) - 2):
            ngram = ' '.join(words[i:i+3])
            ngram_counts[ngram] += 1
    
    # Filter to repeated phrases
    # Exclude very common phrases
    common_phrases = {'i am', 'you are', 'it is', 'do not', 'i do', 'i have',
                     'you have', 'there is', 'there are', 'what is', 'that is'}
    
    signatures = [
        phrase for phrase, count in ngram_counts.items()
        if count >= min_occurrences and phrase not in common_phrases
    ]
    
    # Sort by frequency
    signatures.sort(key=lambda x: -ngram_counts[x])
    
    return signatures[:5]
