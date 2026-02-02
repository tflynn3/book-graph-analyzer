"""
Dialogue Extraction

Extract quoted speech from text and attribute to speakers.
"""

from dataclasses import dataclass, field
from typing import Optional
import re

import spacy
from spacy.tokens import Doc


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
    context_before: str = ""            # Text before the quote
    context_after: str = ""             # Text after the quote
    
    # Classification
    is_question: bool = False
    is_exclamation: bool = False
    is_statement: bool = True
    
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
    
    # Find all quoted text
    # Pattern handles "...", '...', and "..." (curly quotes)
    quote_patterns = [
        r'"([^"]+)"',                    # Standard double quotes
        r'\u201c([^\u201d]+)\u201d',     # Curly double quotes
        r"'([^']+)'",                    # Single quotes (be careful - apostrophes)
        r'\u2018([^\u2019]+)\u2019',     # Curly single quotes
        # Mangled encoding patterns (double-encoded UTF-8)
        r'\xe2\x80\x9c([^\xe2]+)\xe2\x80\x9d',  # â€œ...â€
        r'â€œ([^â]+)â€',                # Same but as decoded characters
    ]
    
    # Combine patterns, prefer double quotes
    # Use the most common pattern first: "..."
    all_quotes = []
    
    for pattern in quote_patterns[:2]:  # Focus on double quotes
        for match in re.finditer(pattern, text):
            quote_text = match.group(1).strip()
            if len(quote_text) > 1:  # Skip single characters
                all_quotes.append({
                    'text': quote_text,
                    'start': match.start(),
                    'end': match.end(),
                })
    
    # Sort by position
    all_quotes.sort(key=lambda x: x['start'])
    
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
        
        # Try to attribute speaker
        speaker, speech_verb, confidence = _attribute_speaker(
            context_before, context_after, text, nlp
        )
        
        # Classify the dialogue
        is_question = quote_text.rstrip().endswith('?')
        is_exclamation = quote_text.rstrip().endswith('!')
        is_statement = not is_question and not is_exclamation
        
        line = DialogueLine(
            text=quote_text,
            speaker=speaker,
            speech_verb=speech_verb,
            passage_id=passage_id,
            position=i,
            context_before=context_before[-50:] if len(context_before) > 50 else context_before,
            context_after=context_after[:50] if len(context_after) > 50 else context_after,
            is_question=is_question,
            is_exclamation=is_exclamation,
            is_statement=is_statement,
            attribution_confidence=confidence,
        )
        
        result.dialogue_lines.append(line)
        result.total_dialogue_chars += len(quote_text)
    
    # Calculate dialogue ratio
    if result.total_text_chars > 0:
        result.dialogue_ratio = result.total_dialogue_chars / result.total_text_chars
    
    return result


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
        r'^[,.]?\s*(' + '|'.join(SPEECH_VERBS) + r')\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        context_after,
        re.IGNORECASE
    )
    if after_match:
        speech_verb = after_match.group(1).lower()
        speaker = after_match.group(2)
        confidence = 0.9
        return speaker, speech_verb, confidence
    
    # Pattern 2: NAME said, "..."
    # Look in context_before for name + speech verb
    before_match = re.search(
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(' + '|'.join(SPEECH_VERBS) + r')[,.]?\s*$',
        context_before,
        re.IGNORECASE
    )
    if before_match:
        speaker = before_match.group(1)
        speech_verb = before_match.group(2).lower()
        confidence = 0.9
        return speaker, speech_verb, confidence
    
    # Pattern 3: said NAME (without quote immediately before/after)
    after_verb_match = re.search(
        r'(' + '|'.join(SPEECH_VERBS) + r')\s+([A-Z][a-z]+)',
        context_after,
        re.IGNORECASE
    )
    if after_verb_match:
        speech_verb = after_verb_match.group(1).lower()
        speaker = after_verb_match.group(2)
        confidence = 0.7
        return speaker, speech_verb, confidence
    
    # Pattern 4: Look for any capitalized name near the quote
    # Lower confidence since it might be wrong
    name_pattern = r'\b([A-Z][a-z]+)\b'
    
    # Common words to filter out
    non_names = {
        # Pronouns
        'he', 'she', 'it', 'they', 'we', 'i', 'you',
        'him', 'her', 'them', 'us', 'me',
        # Determiners/articles
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        # Conjunctions/prepositions
        'but', 'and', 'or', 'then', 'so', 'yet', 'for',
        'to', 'from', 'with', 'at', 'by', 'in', 'on', 'of', 'after', 'before',
        # Common adverbs
        'there', 'here', 'where', 'when', 'what', 'how', 'why',
        'now', 'then', 'just', 'still', 'even', 'also',
        # Other common words
        'one', 'all', 'some', 'any', 'no', 'not', 'only',
        # Descriptions that might be capitalized at sentence start
        'old', 'young', 'little', 'other', 'first', 'last',
    }
    
    # Check after first
    after_names = re.findall(name_pattern, context_after[:30])
    if after_names:
        filtered = [n for n in after_names if n.lower() not in non_names and len(n) > 1]
        if filtered:
            speaker = filtered[0]
            confidence = 0.4
            return speaker, speech_verb, confidence
    
    # Check before
    before_names = re.findall(name_pattern, context_before[-30:])
    if before_names:
        filtered = [n for n in before_names if n.lower() not in non_names and len(n) > 1]
        if filtered:
            speaker = filtered[-1]  # Take the last one (closest to quote)
            confidence = 0.3
            return speaker, speech_verb, confidence
    
    return None, None, 0.0


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
