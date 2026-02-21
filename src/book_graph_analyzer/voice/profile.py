"""
Character Voice Profile

Captures the distinctive speech patterns of a character.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import re
import statistics
from collections import Counter


# ---------------------------------------------------------------------------
# Topic keyword clusters for topic_distribution
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "history": [
        "age", "era", "ancient", "old", "years", "kingdom", "legend", "tale",
        "past", "long ago", "remember", "history", "lore", "tradition", "ages",
        "days of", "yore", "elder", "forebear", "ancestor",
    ],
    "war": [
        "battle", "fight", "sword", "enemy", "army", "war", "warrior", "shield",
        "blade", "weapon", "attack", "defeat", "victory", "conquer", "siege",
        "foe", "host", "orc", "darkness", "evil", "sauron", "ring",
    ],
    "practical": [
        "food", "water", "road", "path", "camp", "fire", "rest", "sleep",
        "eat", "drink", "walk", "journey", "carry", "pack", "need", "get",
        "go", "come", "here", "there", "do", "make", "take",
    ],
    "wisdom": [
        "wisdom", "counsel", "hope", "fate", "doom", "purpose", "must",
        "ought", "understand", "know", "think", "believe", "truth", "power",
        "choice", "decide", "heart", "courage", "will", "destiny",
    ],
    "nature": [
        "forest", "river", "mountain", "sea", "tree", "stone", "sky", "land",
        "earth", "water", "wind", "star", "sun", "moon", "light", "shadow",
        "dark", "night", "day", "vale", "plain",
    ],
    "friendship": [
        "friend", "companion", "fellowship", "trust", "together", "loyal",
        "love", "care", "kin", "folk", "people", "home", "family", "master",
        "sir", "dear", "good",
    ],
}

# Words that would be anachronistic in a medieval/high-fantasy register
MODERN_ANACHRONISMS: list[str] = [
    "okay", "ok", "yeah", "yep", "nope", "cool", "awesome", "guys",
    "basically", "literally", "actually", "totally", "like", "whatever",
    "seriously", "definitely", "absolutely", "random", "stressed", "vibe",
    "chill", "dude", "bro", "hello", "hi", "bye", "thanks", "alright",
    "sure", "fine", "wanna", "gonna", "gotta", "kinda", "sorta",
]

# Common imperative verb starts (infinitive / base form)
IMPERATIVE_STARTS: list[str] = [
    "go", "come", "run", "stop", "listen", "look", "wait", "stand",
    "do not", "don't", "never", "beware", "fly", "flee", "seek", "find",
    "follow", "stay", "leave", "hold", "keep", "turn", "move", "speak",
    "tell", "say", "fear", "trust", "grant", "forgive", "remember",
    "bring", "take", "give", "get", "make", "let", "help", "save",
    "throw", "cast", "pass", "hear", "heed", "see", "read", "open",
    "close", "guard", "protect", "hide", "rise", "fall", "walk", "halt",
]

# Rhetorical question markers
RHETORICAL_PATTERNS: list[str] = [
    r"^why\s+(would|should|must|do|does|did|shall|will)\b",
    r"^how\s+(dare|could|can|shall|would|should)\b",
    r"\bis it not\b.*\?",
    r"\bare (?:we|you|they) not\b.*\?",
    r"\bwas (?:it|he|she|there) not\b.*\?",
    r"^what\s+(good|use|point|purpose)\b",
    r"^who\s+(among|would|could|shall|dares|dare)\b",
    r"^do you not\b",
    r"^does (?:he|she|it|one) not\b",
]

_RHETORICAL_RE = [re.compile(p, re.IGNORECASE) for p in RHETORICAL_PATTERNS]


def _is_rhetorical(text: str) -> bool:
    """Heuristically decide if a question is rhetorical."""
    stripped = text.strip()
    if not stripped.endswith("?"):
        return False
    for pattern in _RHETORICAL_RE:
        if pattern.search(stripped):
            return True
    return False


def _is_imperative(text: str) -> bool:
    """Heuristically decide if a line is imperative."""
    lowered = text.lower().strip()
    for imp in IMPERATIVE_STARTS:
        if (
            lowered.startswith(imp + " ")
            or lowered.startswith(imp + ",")
            or lowered.startswith(imp + "!")
            or lowered.startswith(imp + "?")
            or lowered == imp
        ):
            return True
    return False


def _compute_topic_distribution(all_words: list[str]) -> dict[str, float]:
    """Assign a word-overlap topic distribution. Returns proportions summing to 1."""
    word_set = set(all_words)
    scores: dict[str, float] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        hit = sum(1 for kw in keywords if kw in word_set)
        scores[topic] = float(hit)
    total = sum(scores.values())
    if total == 0:
        # Uniform distribution
        n = len(TOPIC_KEYWORDS)
        return {t: round(1.0 / n, 4) for t in TOPIC_KEYWORDS}
    return {t: round(v / total, 4) for t, v in scores.items()}


def _compute_formality_score(
    archaism_rate: float,
    contraction_ratio: float,
    avg_word_length: float,
    first_person_ratio: float,
) -> float:
    """Compute 0.0–1.0 formality score from four signals."""
    # archaism: rate up to 0.05 = full formal contribution
    arch = min(1.0, archaism_rate / 0.05)
    # contractions: 0.15+ → fully informal
    contr = 1.0 - min(1.0, contraction_ratio / 0.15)
    # word length: 3 chars informal, 7+ chars formal
    wl = min(1.0, max(0.0, (avg_word_length - 3.0) / 4.0))
    # first-person: 0.08+ → fully informal
    fp = 1.0 - min(1.0, first_person_ratio / 0.08)
    score = 0.25 * arch + 0.25 * contr + 0.25 * wl + 0.25 * fp
    return round(max(0.0, min(1.0, score)), 4)


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

    # ------------------------------------------------------------------
    # Issue #10: Audience-variant metrics
    # ------------------------------------------------------------------
    formality_by_audience: dict[str, float] = field(default_factory=dict)
    # e.g. {'hobbit': 0.58, 'elf': 0.84}
    length_by_audience: dict[str, float] = field(default_factory=dict)
    # avg utterance length by audience type
    register_by_audience: dict[str, str] = field(default_factory=dict)
    # dominant register per audience type

    # Computed formality score (0.0 informal → 1.0 formal)
    formality_score: float = 0.0
    archaism_rate: float = 0.0
    rhetorical_density: float = 0.0   # Questions used for effect, not info-seeking
    imperative_ratio: float = 0.0     # Lines that issue commands

    # Fingerprint extras
    never_says: list[str] = field(default_factory=list)
    # Anachronistic / wrong-register words this character avoids
    topic_distribution: dict[str, float] = field(default_factory=dict)
    # e.g. {'history': 0.3, 'practical': 0.4}

    @classmethod
    def from_dialogue_lines(
        cls,
        character_name: str,
        lines: list,  # List of DialogueLine objects
        character_id: Optional[str] = None,
        all_character_words: Optional[dict[str, Counter]] = None,
        audience_lines: Optional[list[tuple]] = None,
        # List of (DialogueLine, audience_type, context_type) for audience metrics
    ) -> "CharacterVoiceProfile":
        """
        Build a voice profile from dialogue lines.

        Args:
            character_name: Name of the character
            lines: List of DialogueLine objects
            character_id: Optional canonical ID
            all_character_words: Word counts for all characters (to find distinctive words)
            audience_lines: Pre-classified (line, audience_type, context_type) triples
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
        rhetorical_questions = 0
        imperatives = 0

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
                if _is_rhetorical(text):
                    rhetorical_questions += 1
            elif line.is_exclamation:
                exclamations += 1
            else:
                statements += 1

            # Imperatives
            if _is_imperative(text):
                imperatives += 1

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
            profile.imperative_ratio = round(imperatives / profile.total_lines, 4)

        # Rhetorical density: fraction of ALL lines that are rhetorical questions
        if profile.total_lines > 0:
            profile.rhetorical_density = round(rhetorical_questions / profile.total_lines, 4)

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
        profile.archaism_rate = round(
            profile.archaism_count / profile.total_words if profile.total_words > 0 else 0.0,
            4,
        )

        # Formality score (composite)
        profile.formality_score = _compute_formality_score(
            profile.archaism_rate,
            profile.contraction_ratio,
            profile.avg_word_length,
            profile.first_person_ratio,
        )

        # Topic distribution
        profile.topic_distribution = _compute_topic_distribution(all_words)

        # never_says: modern/anachronistic words this character does NOT use
        used_words_lower = {w.lower().strip('.,!?"\'') for w in all_words}
        profile.never_says = [w for w in MODERN_ANACHRONISMS if w not in used_words_lower]

        # Sample quotes (pick diverse ones)
        profile.sample_quotes = _select_sample_quotes(lines, max_quotes=5)

        # Signature phrases (repeated sequences)
        profile.signature_phrases = _find_signature_phrases(lines)

        # Audience-variant metrics
        if audience_lines:
            profile.formality_by_audience, profile.length_by_audience, profile.register_by_audience = (
                _compute_audience_metrics(audience_lines)
            )

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
        lines_out = [
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
            f"   Imperatives: {self.imperative_ratio*100:.1f}%",
            f"   Rhetorical density: {self.rhetorical_density*100:.1f}%",
            f"",
            f"[Formality]",
            f"   Formality score: {self.formality_score:.3f}",
            f"   Archaism rate: {self.archaism_rate*100:.1f}%",
            f"   Contractions: {self.contraction_ratio*100:.1f}%",
            f"",
            f"[Vocabulary]",
            f"   Unique words: {self.unique_words}",
            f"   Type-token ratio: {self.type_token_ratio:.3f}",
            f"",
        ]

        if self.top_words:
            lines_out.append(f"[Top Words]")
            for word, count in self.top_words[:10]:
                lines_out.append(f"   {word}: {count}")
            lines_out.append("")

        if self.distinctive_words:
            lines_out.append(f"[Distinctive Words]")
            lines_out.append(f"   {', '.join(self.distinctive_words[:10])}")
            lines_out.append("")

        if self.archaisms_used:
            lines_out.append(f"[Archaic Language]")
            lines_out.append(f"   {', '.join(self.archaisms_used)}")
            lines_out.append("")

        if self.topic_distribution:
            lines_out.append(f"[Topic Distribution]")
            for topic, weight in sorted(self.topic_distribution.items(), key=lambda x: -x[1]):
                lines_out.append(f"   {topic}: {weight:.2f}")
            lines_out.append("")

        if self.formality_by_audience:
            lines_out.append(f"[Formality by Audience]")
            for aud, score in sorted(self.formality_by_audience.items()):
                lines_out.append(f"   {aud}: {score:.3f}")
            lines_out.append("")

        if self.sample_quotes:
            lines_out.append(f"[Sample Quotes]")
            for quote in self.sample_quotes[:3]:
                display = quote[:80] + "..." if len(quote) > 80 else quote
                lines_out.append(f'   "{display}"')
            lines_out.append("")

        return "\n".join(lines_out)


# ---------------------------------------------------------------------------
# Audience-variant metric computation
# ---------------------------------------------------------------------------

def _compute_audience_metrics(
    audience_lines: list[tuple],
) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    """
    Compute per-audience formality, avg length, and dominant register.

    Args:
        audience_lines: list of (DialogueLine, audience_type, context_type) triples

    Returns:
        (formality_by_audience, length_by_audience, register_by_audience)
    """
    # Group by audience
    by_audience: dict[str, list] = {}
    for line, aud_type, ctx_type in audience_lines:
        if aud_type not in by_audience:
            by_audience[aud_type] = []
        by_audience[aud_type].append((line, ctx_type))

    formality_by: dict[str, float] = {}
    length_by: dict[str, float] = {}
    register_by: dict[str, str] = {}

    for aud_type, items in by_audience.items():
        lines_for_aud = [item[0] for item in items]
        ctx_types = [item[1] for item in items]

        # Avg length
        lengths = [len(l.text.split()) for l in lines_for_aud]
        length_by[aud_type] = round(statistics.mean(lengths), 2) if lengths else 0.0

        # Formality approximation per audience
        all_words = []
        for l in lines_for_aud:
            all_words.extend(l.text.lower().split())

        archaisms_list = {
            "thee", "thou", "thy", "thine", "ye", "hath", "doth", "art", "wast",
            "wherefore", "hither", "thither", "whither", "hence", "thence",
            "ere", "nay", "aye", "yea", "behold", "lo", "alas", "forsooth",
            "methinks", "mayhap", "perchance", "betwixt", "amongst", "whilst",
            "verily", "hark", "hearken", "tarry", "prithee",
        }
        contraction_patterns = ["n't", "'s", "'re", "'ve", "'ll", "'d", "'m"]

        arch_count = sum(1 for w in all_words if w.strip('.,!?"\'') in archaisms_list)
        contr_count = sum(
            1 for w in all_words
            if any(p in w for p in contraction_patterns)
        )
        fp_words = {'i', 'me', 'my', 'mine', 'myself', "i'm", "i've", "i'll", "i'd"}
        fp_count = sum(1 for w in all_words if w.strip('.,!?"\'') in fp_words)
        word_lengths = [len(w.strip('.,!?"\'')) for w in all_words if w.strip('.,!?"\'')]
        avg_wl = statistics.mean(word_lengths) if word_lengths else 4.0
        total = len(all_words) if all_words else 1

        archaism_rate = arch_count / total
        contraction_ratio = contr_count / total
        first_person_ratio = fp_count / total
        formality_by[aud_type] = _compute_formality_score(
            archaism_rate, contraction_ratio, avg_wl, first_person_ratio
        )

        # Register: most common context type → register label
        ctx_counter = Counter(ctx_types)
        dominant_ctx = ctx_counter.most_common(1)[0][0] if ctx_counter else "neutral"
        # Map context to register label
        ctx_to_register = {
            "command": "authoritative",
            "warning": "urgent",
            "comfort": "intimate",
            "explanation": "didactic",
            "farewell": "ceremonial",
            "crisis": "terse",
        }
        register_by[aud_type] = ctx_to_register.get(dominant_ctx, "neutral")

    return formality_by, length_by, register_by


# ---------------------------------------------------------------------------
# Helper functions (unchanged from original, plus new ones)
# ---------------------------------------------------------------------------

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
        if 5 <= length <= 20:
            return 1.0
        elif 3 <= length <= 30:
            return 0.5
        else:
            return 0.1

    questions.sort(key=lambda x: -quality_score(x))
    exclamations.sort(key=lambda x: -quality_score(x))
    statements.sort(key=lambda x: -quality_score(x))

    if questions:
        quotes.append(questions[0].text)
    if exclamations:
        quotes.append(exclamations[0].text)

    for stmt in statements:
        if len(quotes) >= max_quotes:
            break
        if stmt.text not in quotes:
            quotes.append(stmt.text)

    return quotes


def _find_signature_phrases(lines: list, min_occurrences: int = 2) -> list[str]:
    """Find phrases this character repeats."""
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

    common_phrases = {'i am', 'you are', 'it is', 'do not', 'i do', 'i have',
                     'you have', 'there is', 'there are', 'what is', 'that is'}

    signatures = [
        phrase for phrase, count in ngram_counts.items()
        if count >= min_occurrences and phrase not in common_phrases
    ]

    signatures.sort(key=lambda x: -ngram_counts[x])

    return signatures[:5]
