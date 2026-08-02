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
    
    # Formality indicators (Issue #10 — VoiceProfile spec)
    formality_score: float = 0.0        # 0.0 (informal/dialect) to 1.0 (high formal/archaic)
    archaism_rate: float = 0.0          # Archaisms per 100 words
    rhetorical_density: float = 0.0     # Questions used rhetorically (not seeking info)
    imperative_ratio: float = 0.0       # Lines that are commands/imperatives
    
    # Audience-variant metrics
    formality_by_audience: dict = field(default_factory=dict)   # { 'hobbit': 0.58, 'elf': 0.84, ... }
    length_by_audience: dict = field(default_factory=dict)       # avg words per line by audience type
    register_by_audience: dict = field(default_factory=dict)     # dominant context_type per audience
    
    # Distinctive features
    top_words: list = field(default_factory=list)           # Most used words (list of [word, count])
    distinctive_words: list = field(default_factory=list)    # Words unique to this character
    signature_phrases: list = field(default_factory=list)    # Recurring exact phrases
    never_says: list = field(default_factory=list)           # Words/constructions this character never uses
    topic_distribution: dict = field(default_factory=dict)   # { 'history': 0.3, 'practical': 0.4, ... }
    
    # Archaic language (Tolkien-relevant)
    archaism_count: int = 0
    archaisms_used: list = field(default_factory=list)
    
    # Verse/song handling (Issue #10)
    verse_lines: int = 0                # Lines detected as verse/song
    prose_lines: int = 0                # Lines detected as prose dialogue
    
    # Sample quotes
    sample_quotes: list = field(default_factory=list)  # Representative quotes
    
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
        imperatives = 0
        verse_count = 0
        prose_count = 0
        rhetorical_questions = 0
        
        contractions = 0
        first_person = 0
        second_person = 0
        
        # Audience-variant accumulators: { audience_type: [words_per_line, ...] }
        audience_lengths: dict[str, list[float]] = {}
        audience_archaism_counts: dict[str, int] = {}
        audience_contraction_counts: dict[str, int] = {}
        audience_word_counts: dict[str, int] = {}
        # Context type per audience for dominant register detection
        audience_context_types: dict[str, list[str]] = {}
        
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
        
        # Rhetorical question patterns (short, or using certain structures)
        rhetorical_patterns = [
            "what does", "what does it", "who knows", "who can", "who would",
            "what is the use", "why should", "why would", "how could",
            "do you not", "have you not", "is it not", "was it not",
            "can you not", "could you not", "shall we not",
        ]
        
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
            
            # Classify line type
            is_verse = getattr(line, 'is_verse', False)
            is_imperative = getattr(line, 'is_imperative', False)
            
            if is_verse:
                verse_count += 1
            else:
                prose_count += 1
            
            if line.is_question:
                questions += 1
                # Check for rhetorical question
                text_lower = text.lower()
                if any(p in text_lower for p in rhetorical_patterns) or word_count <= 4:
                    rhetorical_questions += 1
            elif line.is_exclamation:
                exclamations += 1
            else:
                statements += 1
            
            if is_imperative:
                imperatives += 1
            
            # Per-word formality analysis
            line_archaisms = 0
            line_contractions = 0
            for word in words:
                word_lower = word.lower().strip('.,!?"\'')
                if word_lower in first_person_words:
                    first_person += 1
                if word_lower in second_person_words:
                    second_person += 1
                for pattern in contraction_patterns:
                    if pattern in word:
                        contractions += 1
                        line_contractions += 1
                        break
                if word_lower in archaisms_list:
                    archaisms_found.add(word_lower)
                    line_archaisms += 1
            
            # Audience-variant accumulation
            audience = getattr(line, 'audience_type', 'neutral')
            context = getattr(line, 'context_type', 'statement')
            
            if audience not in audience_lengths:
                audience_lengths[audience] = []
                audience_archaism_counts[audience] = 0
                audience_contraction_counts[audience] = 0
                audience_word_counts[audience] = 0
                audience_context_types[audience] = []
            
            audience_lengths[audience].append(float(word_count))
            audience_archaism_counts[audience] += line_archaisms
            audience_contraction_counts[audience] += line_contractions
            audience_word_counts[audience] += word_count
            audience_context_types[audience].append(context)
        
        profile.total_words = len(all_words)
        profile.verse_lines = verse_count
        profile.prose_lines = prose_count
        
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
            profile.imperative_ratio = imperatives / profile.total_lines
        
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
        
        # Archaisms
        profile.archaism_count = sum(word_counts.get(a, 0) for a in archaisms_found)
        profile.archaisms_used = list(archaisms_found)
        profile.archaism_rate = (profile.archaism_count / profile.total_words * 100) if profile.total_words > 0 else 0.0
        
        # Formality score (0=informal, 1=formal)
        # Based on: archaism rate (normalized), low contraction, high avg word length
        archaism_norm = min(profile.archaism_rate / 5.0, 1.0)    # 5 archaisms/100w = fully formal
        contraction_inv = max(0.0, 1.0 - profile.contraction_ratio * 10)  # 10% contractions = 0 formality
        word_len_norm = min((profile.avg_word_length - 3.0) / 3.0, 1.0) if profile.avg_word_length > 3 else 0.0
        profile.formality_score = (archaism_norm * 0.5 + contraction_inv * 0.3 + word_len_norm * 0.2)
        profile.formality_score = max(0.0, min(1.0, profile.formality_score))
        
        # Rhetorical density: rhetorical questions as fraction of all questions
        if questions > 0:
            profile.rhetorical_density = rhetorical_questions / questions
        
        # Audience-variant metrics
        for audience, lengths in audience_lengths.items():
            if lengths:
                profile.length_by_audience[audience] = statistics.mean(lengths)
                # Formality by audience: archaism / words for this audience
                aud_words = audience_word_counts.get(audience, 0)
                if aud_words > 0:
                    aud_arch = audience_archaism_counts.get(audience, 0)
                    aud_contr = audience_contraction_counts.get(audience, 0)
                    aud_arch_norm = min(aud_arch / aud_words * 100 / 5.0, 1.0)
                    aud_contr_inv = max(0.0, 1.0 - (aud_contr / aud_words) * 10)
                    profile.formality_by_audience[audience] = (
                        aud_arch_norm * 0.6 + aud_contr_inv * 0.4
                    )
                else:
                    profile.formality_by_audience[audience] = 0.5
                
                # Dominant context type for this audience
                ctx_list = audience_context_types.get(audience, [])
                if ctx_list:
                    ctx_counter = Counter(ctx_list)
                    profile.register_by_audience[audience] = ctx_counter.most_common(1)[0][0]
        
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
        
        # Never-says: words that appear in other characters but never in this one
        if all_character_words:
            profile.never_says = _find_never_says(
                character_name, word_counts, all_character_words
            )
        
        # Topic distribution
        profile.topic_distribution = _compute_topic_distribution(all_words)
        
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
        summary_lines = [
            f"=== Voice Profile: {self.character_name} ===",
            "",
            "[Corpus]",
            f"   Total lines: {self.total_lines} (prose: {self.prose_lines}, verse: {self.verse_lines})",
            f"   Total words: {self.total_words}",
            "",
            "[Speech Patterns]",
            f"   Avg utterance: {self.avg_utterance_length:.1f} words",
            f"   Range: {self.min_utterance_length} - {self.max_utterance_length} words",
            f"   Questions: {self.question_ratio*100:.1f}%  Exclamations: {self.exclamation_ratio*100:.1f}%  Imperatives: {self.imperative_ratio*100:.1f}%",
            f"   Rhetorical density: {self.rhetorical_density:.2f}",
            "",
            "[Formality & Register]",
            f"   Formality score: {self.formality_score:.2f}  (0=informal, 1=formal)",
            f"   Archaism rate: {self.archaism_rate:.2f}/100w",
            f"   Contractions: {self.contraction_ratio*100:.1f}%",
            "",
            "[Vocabulary]",
            f"   Unique words: {self.unique_words}",
            f"   Type-token ratio: {self.type_token_ratio:.3f}",
            "",
        ]
        
        if self.formality_by_audience:
            summary_lines.append("[Audience-Variant Formality]")
            for aud, score in sorted(self.formality_by_audience.items(), key=lambda x: -x[1]):
                avg_len = self.length_by_audience.get(aud, 0)
                reg = self.register_by_audience.get(aud, "-")
                summary_lines.append(f"   {aud:10s}: formality={score:.2f}  avg_len={avg_len:.1f}w  register={reg}")
            summary_lines.append("")
        
        if self.top_words:
            summary_lines.append("[Top Words]")
            for word, count in self.top_words[:10]:
                summary_lines.append(f"   {word}: {count}")
            summary_lines.append("")
        
        if self.distinctive_words:
            summary_lines.append("[Distinctive Words]")
            summary_lines.append(f"   {', '.join(self.distinctive_words[:10])}")
            summary_lines.append("")
        
        if self.never_says:
            summary_lines.append("[Never Says]")
            summary_lines.append(f"   {', '.join(self.never_says[:10])}")
            summary_lines.append("")
        
        if self.topic_distribution:
            summary_lines.append("[Topic Distribution]")
            for topic, ratio in self.topic_distribution.items():
                summary_lines.append(f"   {topic}: {ratio*100:.0f}%")
            summary_lines.append("")
        
        if self.archaisms_used:
            summary_lines.append("[Archaic Language]")
            summary_lines.append(f"   {', '.join(self.archaisms_used)}")
            summary_lines.append("")
        
        if self.sample_quotes:
            summary_lines.append("[Sample Quotes]")
            for quote in self.sample_quotes[:3]:
                # Truncate long quotes
                display = quote[:80] + "..." if len(quote) > 80 else quote
                summary_lines.append(f'   "{display}"')
            summary_lines.append("")
        
        return "\n".join(summary_lines)


def _find_never_says(
    character: str,
    char_words: Counter,
    all_char_words: dict[str, Counter],
    min_other_chars: int = 1,
    top_n: int = 10,
) -> list[str]:
    """
    Find common words that OTHER characters use, but this character never does.
    
    These are potential "never says" items — words that break character voice.
    """
    # Collect words used by at least min_other_chars other characters
    other_common: Counter = Counter()
    other_char_count: Counter = Counter()
    
    for other_char, words in all_char_words.items():
        if other_char == character:
            continue
        for word, count in words.items():
            if count >= 2 and len(word) > 3:
                other_common[word] += count
                other_char_count[word] += 1
    
    # Find words used by many others but not by this character
    never_used = []
    for word, spread in other_char_count.most_common(50):
        if spread >= min_other_chars and char_words.get(word, 0) == 0:
            # Filter out stop words
            if word not in {'the', 'and', 'that', 'this', 'with', 'have', 'from',
                           'they', 'their', 'what', 'when', 'will', 'been', 'all'}:
                never_used.append(word)
    
    return never_used[:top_n]


_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "history": [
        "age", "era", "ancient", "long", "ago", "once", "before", "remember",
        "elder", "days", "old", "time", "past", "tale", "legend", "story",
        "years", "centuries",
    ],
    "practical": [
        "food", "water", "road", "path", "fire", "camp", "night", "day",
        "journey", "walk", "ride", "horse", "door", "house", "village",
        "market", "eat", "drink", "sleep", "rest",
    ],
    "war": [
        "battle", "sword", "enemy", "fight", "army", "war", "orc", "goblin",
        "attack", "defend", "victory", "defeat", "blood", "weapon", "shield",
        "host", "forces",
    ],
    "wisdom": [
        "counsel", "wisdom", "advice", "think", "know", "understand", "learn",
        "truth", "knowledge", "secret", "hidden", "meaning", "purpose",
        "plan", "strategy", "way",
    ],
    "nature": [
        "tree", "forest", "mountain", "river", "valley", "stone", "sky",
        "wind", "rain", "flower", "leaf", "bird", "star", "sun", "moon",
        "earth", "water", "light", "dark",
    ],
    "fellowship": [
        "friend", "companion", "together", "trust", "hope", "heart",
        "love", "care", "help", "aid", "follow", "loyal", "faithful",
        "master", "sir",
    ],
}


def _compute_topic_distribution(all_words: list[str]) -> dict[str, float]:
    """
    Compute topic distribution from a list of words.
    
    Returns normalized distribution over topic buckets.
    """
    word_set = set(all_words)
    topic_scores: dict[str, int] = {}
    
    for topic, keywords in _TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in word_set)
        if score > 0:
            topic_scores[topic] = score
    
    total = sum(topic_scores.values())
    if total == 0:
        return {}
    
    return {t: round(s / total, 3) for t, s in sorted(topic_scores.items(), key=lambda x: -x[1])}


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
