"""
Dialogue Extraction

Extract quoted speech from text and attribute to speakers.
"""

from dataclasses import dataclass, field
from typing import Optional
import re

import spacy

from .audience import classify_dialogue_line


# Speech verbs for attribution detection
SPEECH_VERBS = {
    # Common
    "said", "says", "say", "saying",
    "asked", "asks", "ask", "asking",
    "replied", "replies", "reply", "replying",
    "answered", "answers", "answer", "answering",
    
    # Manner of speaking
    "whispered", "whispers", "whisper",
    "shouted", "shouts", "shout",
    "cried", "cries", "cry",
    "called", "calls", "call",
    "yelled", "yells", "yell",
    "screamed", "screams", "scream",
    "muttered", "mutters", "mutter",
    "murmured", "murmurs", "murmur",
    "growled", "growls", "growl",
    "hissed", "hisses", "hiss",
    "snarled", "snarls", "snarl",
    "snapped", "snaps", "snap",
    "groaned", "groans", "groan",
    "moaned", "moans", "moan",
    "sighed", "sighs", "sigh",
    "laughed", "laughs", "laugh",
    "chuckled", "chuckles", "chuckle",
    "giggled", "giggles", "giggle",
    "sobbed", "sobs", "sob",
    "wailed", "wails", "wail",
    
    # Declarative
    "declared", "declares", "declare",
    "announced", "announces", "announce",
    "proclaimed", "proclaims", "proclaim",
    "exclaimed", "exclaims", "exclaim",
    "stated", "states", "state",
    
    # Questioning
    "inquired", "inquires", "inquire",
    "queried", "queries", "query",
    "demanded", "demands", "demand",
    "wondered", "wonders", "wonder",
    
    # Persuasive
    "begged", "begs", "beg",
    "pleaded", "pleads", "plead",
    "urged", "urges", "urge",
    "insisted", "insists", "insist",
    "suggested", "suggests", "suggest",
    "proposed", "proposes", "propose",
    
    # Informative
    "explained", "explains", "explain",
    "told", "tells", "tell",
    "informed", "informs", "inform",
    "added", "adds", "add",
    "continued", "continues", "continue",
    "went on",
    
    # Archaic (Tolkien-relevant)
    "spake", "quoth", "cried out",
}
_SPEECH_VERB_PATTERN = "|".join(sorted((re.escape(verb) for verb in SPEECH_VERBS), key=len, reverse=True))
_NAME_TOKEN_PATTERN = (
    r"[A-Z\u00C0-\u00D6\u00D8-\u00DE]"
    r"[a-z\u00E0-\u00F6\u00F8-\u00FF]+"
    r"(?:['\u2019-][A-Z\u00C0-\u00D6\u00D8-\u00DE]?"
    r"[a-z\u00E0-\u00F6\u00F8-\u00FF]+)*"
)
_NAME_PATTERN = rf"{_NAME_TOKEN_PATTERN}(?:\s+{_NAME_TOKEN_PATTERN})?"
_NON_SPEAKER_WORDS = {
    # Pronouns
    "he", "she", "it", "they", "we", "i", "you",
    "him", "her", "them", "us", "me",
    # Determiners/articles
    "the", "a", "an", "this", "that", "these", "those",
    # Conjunctions/prepositions
    "but", "and", "or", "then", "so", "yet", "for",
    "to", "from", "with", "at", "by", "in", "on", "of", "after", "before",
    # Common adverbs
    "there", "here", "where", "when", "what", "how", "why",
    "now", "then", "just", "still", "even", "also",
    # Other common words
    "one", "all", "some", "any", "no", "not", "only",
    "old", "young", "little", "other", "first", "last",
    "yes", "no", "well", "aye", "nay",
}
_SPEECH_INTRO_VERBS = {
    "saying",
    "speaking",
    "calling",
    "crying",
    "shouting",
    "whispering",
    "muttering",
    "murmuring",
    "answering",
    "replying",
    "asking",
    "telling",
    "declaring",
    "announcing",
    "explaining",
    "continuing",
    "adding",
}
_SPEECH_INTRO_PATTERN = "|".join(sorted((re.escape(verb) for verb in _SPEECH_INTRO_VERBS), key=len, reverse=True))
_LEXICAL_QUOTE_CONTEXT_PATTERNS = (
    r"\b(?:called|named|known as|referred to as|refer to as)\s*$",
    r"\b(?:word|words|term|phrase|phrases|title|nickname)\s*$",
)
_CONTINUATION_GAP_PATTERN = re.compile(
    rf"""
    ^[\s,;:.!?'"`\-()]*
    (?:
        (?:(?:he|she|they|we|i)\s+)?
        (?:(?i:{_SPEECH_VERB_PATTERN})|(?i:{_SPEECH_INTRO_PATTERN}))
        (?:\s+(?:again|softly|quietly|grimly|gently|then|aloud|briefly))*
        (?:\s+(?:he|she|they|we|i))?
    )?
    [\s,;:.!?'"`\-()]*$
    """,
    re.VERBOSE,
)


@dataclass
class DialogueLine:
    """A single line of dialogue."""
    text: str                           # The quoted text (without quotes)
    speaker: Optional[str] = None       # Attributed speaker name
    speaker_id: Optional[str] = None    # Canonical entity ID if resolved
    speech_verb: Optional[str] = None   # The verb used (said, asked, etc.)
    
    # Context
    passage_id: Optional[str] = None    # Source passage ID
    position: int = 0                   # Position in passage (0-indexed)
    quote_start: Optional[int] = None   # Start offset of the quoted span in the passage
    quote_end: Optional[int] = None     # End offset of the quoted span in the passage
    context_before: str = ""            # Text before the quote
    context_after: str = ""             # Text after the quote
    
    # Classification
    is_question: bool = False
    is_exclamation: bool = False
    is_statement: bool = True
    is_imperative: bool = False         # Starts with a verb (command/request)
    is_verse: bool = False              # Detected as song/poetry
    
    # Audience & context type (Issue #10)
    audience_type: str = "neutral"      # 'hobbit'|'elf'|'man'|'dwarf'|'enemy'|'neutral'|'self'|'prayer'
    context_type: str = "statement"     # 'crisis'|'explanation'|'command'|'comfort'|'warning'|'farewell'|'statement'
    audience_confidence: float = 0.0    # Confidence in audience classification
    
    # Confidence
    attribution_confidence: float = 0.0  # How confident we are about the speaker


@dataclass
class DialogueExtraction:
    """Result of dialogue extraction from a text."""
    source_text: str
    passage_id: Optional[str] = None
    dialogue_lines: list[DialogueLine] = field(default_factory=list)
    
    # Stats
    total_dialogue_chars: int = 0
    total_text_chars: int = 0
    dialogue_ratio: float = 0.0
    
    @property
    def speaker_counts(self) -> dict[str, int]:
        """Count lines per speaker."""
        counts = {}
        for line in self.dialogue_lines:
            speaker = line.speaker or "UNKNOWN"
            counts[speaker] = counts.get(speaker, 0) + 1
        return counts


def extract_dialogue(
    text: str,
    passage_id: Optional[str] = None,
    nlp: Optional[spacy.Language] = None,
) -> DialogueExtraction:
    """
    Extract dialogue lines from a text passage.
    
    Args:
        text: The text to extract dialogue from
        passage_id: Optional ID for the passage
        nlp: Optional spaCy model for NER-based attribution
        
    Returns:
        DialogueExtraction with all found dialogue lines
    """
    result = DialogueExtraction(
        source_text=text,
        passage_id=passage_id,
        total_text_chars=len(text),
    )
    
    # Find all quoted text. These Tolkien sources use both double- and single-
    # quoted dialogue, but straight single quotes also appear as apostrophes, so
    # we scan them with a boundary-aware parser instead of a loose regex.
    all_quotes = _find_double_quoted_spans(text) + _find_single_quoted_spans(text)
    
    # Sort by position
    all_quotes.sort(key=lambda x: x['start'])
    
    previous_line: DialogueLine | None = None
    previous_quote_end: int | None = None

    # Process each quote
    for i, quote in enumerate(all_quotes):
        quote_text = quote['text']
        start = quote['start']
        end = quote['end']
        
        # Get context
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        context_before = text[context_start:start].strip()
        context_after = text[end:context_end].strip()

        if not _is_probable_dialogue_quote(quote_text, context_before, context_after):
            previous_quote_end = end
            continue
        
        # Try to attribute speaker
        speaker, speech_verb, confidence = _attribute_speaker(
            context_before, context_after, text, nlp
        )
        if speaker is None and previous_line and previous_quote_end is not None:
            carried = _carry_forward_speaker(
                full_text=text,
                quote_start=start,
                previous_quote_end=previous_quote_end,
                previous_line=previous_line,
            )
            if carried is not None:
                speaker, speech_verb, confidence = carried
        
        # Classify the dialogue
        is_question = quote_text.rstrip().endswith('?')
        is_exclamation = quote_text.rstrip().endswith('!')
        is_statement = not is_question and not is_exclamation
        
        # Detect imperative (starts with a base-form verb, no subject pronoun)
        is_imperative = _detect_imperative(quote_text)
        
        # Detect verse/song (short lines, often rhythmic, may have internal newlines)
        is_verse = _detect_verse(quote_text)
        
        # Audience & context classification
        ctx_before_window = context_before[-100:] if len(context_before) > 100 else context_before
        ctx_after_window = context_after[:100] if len(context_after) > 100 else context_after
        audience_cls = classify_dialogue_line(
            quote_text, ctx_before_window, ctx_after_window, speaker
        )
        
        line = DialogueLine(
            text=quote_text,
            speaker=speaker,
            speech_verb=speech_verb,
            passage_id=passage_id,
            position=i,
            quote_start=start,
            quote_end=end,
            context_before=context_before[-50:] if len(context_before) > 50 else context_before,
            context_after=context_after[:50] if len(context_after) > 50 else context_after,
            is_question=is_question,
            is_exclamation=is_exclamation,
            is_statement=is_statement,
            is_imperative=is_imperative,
            is_verse=is_verse,
            audience_type=audience_cls.audience_type,
            context_type=audience_cls.context_type,
            audience_confidence=audience_cls.confidence,
            attribution_confidence=confidence,
        )
        
        result.dialogue_lines.append(line)
        result.total_dialogue_chars += len(quote_text)
        previous_line = line
        previous_quote_end = end
    
    # Calculate dialogue ratio
    if result.total_text_chars > 0:
        result.dialogue_ratio = result.total_dialogue_chars / result.total_text_chars
    
    return result


def _find_double_quoted_spans(text: str) -> list[dict[str, int | str]]:
    quote_patterns = [
        r'"([^"]+)"',                    # Standard double quotes
        r'\u201c([^\u201d]+)\u201d',     # Curly double quotes
        # Mangled encoding patterns (double-encoded UTF-8)
        r'\xe2\x80\x9c([^\xe2]+)\xe2\x80\x9d',  # â€œ...â€
        r'â€œ([^â]+)â€',                # Same but as decoded characters
    ]

    spans: list[dict[str, int | str]] = []
    for pattern in quote_patterns:
        for match in re.finditer(pattern, text):
            quote_text = match.group(1).strip()
            if len(quote_text) <= 1:
                continue
            spans.append({
                "text": quote_text,
                "start": match.start(),
                "end": match.end(),
            })
    return spans


def _find_single_quoted_spans(text: str) -> list[dict[str, int | str]]:
    """Extract single-quoted spans while ignoring apostrophes inside words."""
    spans: list[dict[str, int | str]] = []
    i = 0
    while i < len(text):
        if text[i] != "'":
            i += 1
            continue

        prev_char = text[i - 1] if i > 0 else ""
        next_char = text[i + 1] if i + 1 < len(text) else ""
        if _is_word_apostrophe(prev_char, next_char) or not _is_single_quote_open(prev_char, next_char):
            i += 1
            continue

        j = i + 1
        while j < len(text):
            if text[j] != "'":
                j += 1
                continue

            prev_inner = text[j - 1] if j > 0 else ""
            next_inner = text[j + 1] if j + 1 < len(text) else ""
            if _is_word_apostrophe(prev_inner, next_inner):
                j += 1
                continue
            if not _is_single_quote_close(prev_inner, next_inner):
                j += 1
                continue

            quote_text = text[i + 1:j].strip()
            if len(quote_text) > 1:
                spans.append({
                    "text": quote_text,
                    "start": i,
                    "end": j + 1,
                })
            i = j + 1
            break
        else:
            i += 1

    return spans


def _attribute_speaker(
    context_before: str,
    context_after: str,
    full_text: str,
    nlp: Optional[spacy.Language] = None,
) -> tuple[Optional[str], Optional[str], float]:
    """
    Try to determine who is speaking.
    
    Returns:
        (speaker_name, speech_verb, confidence)
    """
    speaker = None
    speech_verb = None
    confidence = 0.0
    
    # Pattern 1: "..." said NAME
    # Look in context_after for speech verb + name
    after_match = re.search(
        rf"^[,.]?\s*((?i:{_SPEECH_VERB_PATTERN}))\s+({_NAME_PATTERN})",
        context_after,
    )
    if after_match:
        speech_verb = after_match.group(1).lower()
        speaker = _sanitize_speaker_candidate(after_match.group(2))
        if speaker:
            confidence = 0.9
            return speaker, speech_verb, confidence

    # Pattern 1b: "..." NAME said to ADDRESSEE
    after_named_speaker_match = re.search(
        rf"^[,.]?\s*({_NAME_PATTERN})\s+((?i:{_SPEECH_VERB_PATTERN}))(?:\s+(?:to|at)\s+{_NAME_PATTERN})?",
        context_after,
    )
    if after_named_speaker_match:
        speaker = _sanitize_speaker_candidate(after_named_speaker_match.group(1))
        speech_verb = after_named_speaker_match.group(2).lower()
        if speaker:
            confidence = 0.86
            return speaker, speech_verb, confidence
    
    # Pattern 2: NAME said, "..."
    # Look in context_before for name + speech verb
    before_match = re.search(
        rf"({_NAME_PATTERN})\s+((?i:{_SPEECH_VERB_PATTERN}))[:,;.]?\s*$",
        context_before,
    )
    if before_match:
        speaker = _sanitize_speaker_candidate(before_match.group(1))
        speech_verb = before_match.group(2).lower()
        if speaker:
            confidence = 0.9
            return speaker, speech_verb, confidence

    # Pattern 2a: NAME said to ADDRESSEE, "..."
    before_named_speaker_with_addressee = re.search(
        rf"({_NAME_PATTERN})\s+((?i:{_SPEECH_VERB_PATTERN}))(?:\s+(?:to|at)\s+{_NAME_PATTERN})[:,;.]?\s*$",
        context_before,
    )
    if before_named_speaker_with_addressee:
        speaker = _sanitize_speaker_candidate(before_named_speaker_with_addressee.group(1))
        speech_verb = before_named_speaker_with_addressee.group(2).lower()
        if speaker:
            confidence = 0.86
            return speaker, speech_verb, confidence

    # Pattern 2b: "..." [said NAME.] "..."
    before_inverted_match = re.search(
        rf"((?i:{_SPEECH_VERB_PATTERN}))\s+({_NAME_PATTERN})(?:\s+[a-z][a-z'-]*){{0,3}}[:,;.]?\s*$",
        context_before,
    )
    if before_inverted_match:
        speech_verb = before_inverted_match.group(1).lower()
        speaker = _sanitize_speaker_candidate(before_inverted_match.group(2))
        if speaker:
            confidence = 0.85
            return speaker, speech_verb, confidence

    # Pattern 2c: NAME arose, saying: "..."
    lead_in_match = re.search(
        rf"({_NAME_PATTERN})(?:\s+[a-z][a-z'-]*){{0,3}},?\s+((?i:{_SPEECH_INTRO_PATTERN}))[:;,]?\s*$",
        context_before,
    )
    if lead_in_match:
        speaker = _sanitize_speaker_candidate(lead_in_match.group(1))
        speech_verb = lead_in_match.group(2).lower()
        if speaker:
            confidence = 0.78
            return speaker, speech_verb, confidence
    
    # Pattern 3: said NAME (without quote immediately before/after)
    after_verb_match = re.search(
        rf"((?i:{_SPEECH_VERB_PATTERN}))\s+({_NAME_PATTERN})",
        context_after,
    )
    if after_verb_match:
        speech_verb = after_verb_match.group(1).lower()
        speaker = _sanitize_speaker_candidate(after_verb_match.group(2))
        if speaker:
            confidence = 0.7
            return speaker, speech_verb, confidence

    return None, None, 0.0


def _is_word_apostrophe(prev_char: str, next_char: str) -> bool:
    return prev_char.isalpha() and next_char.isalpha()


def _is_single_quote_open(prev_char: str, next_char: str) -> bool:
    if not next_char or next_char.isspace():
        return False
    if prev_char and prev_char.isalpha():
        return False
    return not prev_char or prev_char.isspace() or prev_char in "([{-:;,.!?\n"


def _is_single_quote_close(prev_char: str, next_char: str) -> bool:
    if not prev_char or prev_char.isspace():
        return False
    if next_char and next_char.isalpha():
        return False
    return not next_char or next_char.isspace() or next_char in ")]}-:;,.!?\n"


def _is_probable_dialogue_quote(
    quote_text: str,
    context_before: str,
    context_after: str,
) -> bool:
    """Filter out scare quotes and other quoted fragments that are not dialogue."""
    stripped = quote_text.strip()
    if not stripped:
        return False

    nearby_before = context_before[-60:].lower()
    nearby_after = context_after[:60].lower()
    if any(re.search(pattern, nearby_before) for pattern in _LEXICAL_QUOTE_CONTEXT_PATTERNS):
        return False

    words = re.findall(
        r"[A-Za-z\u00C0-\u00D6\u00D8-\u00DE\u00E0-\u00F6\u00F8-\u00FF]+(?:['\u2019-]"
        r"[A-Za-z\u00C0-\u00D6\u00D8-\u00DE\u00E0-\u00F6\u00F8-\u00FF]+)*",
        stripped,
    )
    if not words:
        return False

    has_end_punctuation = stripped[-1] in "?!.,;:"
    has_speech_cue = bool(
        re.search(rf"(?i)\b(?:{_SPEECH_VERB_PATTERN}|{_SPEECH_INTRO_PATTERN})\b", nearby_before)
        or re.search(rf"(?i)\b(?:{_SPEECH_VERB_PATTERN}|{_SPEECH_INTRO_PATTERN})\b", nearby_after)
        or context_before.rstrip().endswith(":")
    )
    lowered = stripped.lower()
    has_dialogue_pronoun = bool(
        re.search(r"\b(?:i|you|we|my|your|our|me|us|thou|thee|thy|shall|must|cannot|can't|won't|don't|am|are|is)\b", lowered)
    )

    if len(words) <= 2 and not has_end_punctuation:
        return False
    if len(words) <= 3 and not (has_speech_cue or has_end_punctuation):
        return False
    if not has_speech_cue and not has_end_punctuation and len(words) < 5 and not has_dialogue_pronoun:
        return False

    return True


def _carry_forward_speaker(
    *,
    full_text: str,
    quote_start: int,
    previous_quote_end: int,
    previous_line: DialogueLine,
) -> tuple[str, Optional[str], float] | None:
    """Reuse the previous speaker only across tight same-speaker quote continuations."""
    if not previous_line.speaker:
        return None
    if previous_line.attribution_confidence < 0.7:
        return None

    gap = full_text[previous_quote_end:quote_start]
    if len(gap) > 80:
        return None
    if re.search(_NAME_PATTERN, gap):
        return None
    if not _CONTINUATION_GAP_PATTERN.match(gap):
        return None

    return previous_line.speaker, previous_line.speech_verb, 0.65


def _sanitize_speaker_candidate(candidate: str | None) -> str | None:
    """Discard obvious non-speaker tokens from loose quote attribution."""
    if not candidate:
        return None

    tokens = re.findall(
        r"[A-Za-z\u00C0-\u00D6\u00D8-\u00DE\u00E0-\u00F6\u00F8-\u00FF]+(?:['\u2019-]"
        r"[A-Za-z\u00C0-\u00D6\u00D8-\u00DE\u00E0-\u00F6\u00F8-\u00FF]+)*",
        candidate.strip(),
    )
    if not tokens:
        return None

    normalized = " ".join(tokens)
    lowered = [token.lower() for token in tokens]
    if any(token in _NON_SPEAKER_WORDS for token in lowered):
        return None

    if len(tokens) == 1:
        token = tokens[0]
        if len(token) < 3:
            return None
        if not re.search(r"[aeiouy\u00E0-\u00F6\u00F8-\u00FF]", token.lower()):
            return None

    return normalized


def _detect_imperative(text: str) -> bool:
    """
    Heuristically detect imperative sentences.
    
    Imperatives typically start with a base-form verb with no subject.
    """
    # Common imperative starters
    imperative_starters = {
        # Motion
        "go", "come", "stay", "stop", "run", "flee", "follow", "lead",
        "move", "walk", "ride", "fly", "climb", "fall",
        # Action
        "take", "bring", "give", "hold", "keep", "put", "set", "get",
        "make", "do", "let", "leave", "return", "stand", "sit", "rise",
        # Perception
        "look", "see", "watch", "hear", "listen", "read", "speak",
        # Archaic
        "heed", "behold", "hearken", "hark", "tarry", "stay",
        # Negative
        "do not", "do not", "never", "fear not", "worry not",
    }
    tokens = text.lower().split()
    first_word = tokens[0].strip('.,!?"\'-') if tokens else ""
    first_two_raw = tokens[:2] if len(tokens) >= 2 else tokens
    first_two = " ".join(w.strip('.,!?"\'-') for w in first_two_raw)
    
    return first_word in imperative_starters or first_two in imperative_starters


def _detect_verse(text: str) -> bool:
    """
    Heuristically detect verse/songs.
    
    Verse often has:
    - Internal newlines
    - Short, rhythmic lines
    - May start with 'O' or have unusual capitalization mid-sentence
    """
    # Check for internal newlines (multi-line speech)
    if '\n' in text:
        return True
    
    # Check for verse markers
    verse_patterns = [
        r'\bO\s+[A-Z]',             # "O Elbereth" type invocations
        r'\b(sing|song|sang|sung)\b',  # References to singing
        r'[,;]\s+\w+\s+\w+[,;]',    # Comma-delimited short phrases (rhythmic)
    ]
    for pattern in verse_patterns:
        if re.search(pattern, text):
            return True
    
    return False


def extract_dialogue_from_passages(
    passages: list,  # List of Passage objects or dicts
    nlp: Optional[spacy.Language] = None,
) -> list[DialogueExtraction]:
    """
    Extract dialogue from multiple passages.
    
    Args:
        passages: List of passage objects (need .text and .id attributes)
        nlp: Optional spaCy model
        
    Returns:
        List of DialogueExtraction objects
    """
    results = []
    
    for passage in passages:
        # Handle both objects and dicts
        if hasattr(passage, 'text'):
            text = passage.text
            pid = getattr(passage, 'id', None) or str(id(passage))
        elif isinstance(passage, dict):
            text = passage.get('text', '')
            pid = passage.get('id', str(id(passage)))
        else:
            text = str(passage)
            pid = str(id(passage))
        
        extraction = extract_dialogue(text, passage_id=pid, nlp=nlp)
        if extraction.dialogue_lines:  # Only include if there's dialogue
            results.append(extraction)
    
    return results


def merge_dialogue_extractions(
    extractions: list[DialogueExtraction]
) -> dict[str, list[DialogueLine]]:
    """
    Merge dialogue from multiple extractions, grouped by speaker.
    
    Returns:
        Dict mapping speaker name to list of their dialogue lines
    """
    by_speaker = {}
    
    for extraction in extractions:
        for line in extraction.dialogue_lines:
            speaker = line.speaker or "UNKNOWN"
            if speaker not in by_speaker:
                by_speaker[speaker] = []
            by_speaker[speaker].append(line)
    
    return by_speaker
