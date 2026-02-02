"""
Passage Type Classification

Classify passages into scene types: dialogue, action, description, etc.
Uses rule-based heuristics with optional LLM enhancement.
"""

from enum import Enum
from dataclasses import dataclass
import re
from typing import Optional

from spacy.tokens import Doc


class PassageType(Enum):
    """Types of narrative passages."""
    DIALOGUE = "dialogue"           # Conversation between characters
    ACTION = "action"               # Physical action, events happening
    DESCRIPTION = "description"     # Setting, appearance, atmosphere
    TRAVEL = "travel"               # Movement, journey passages
    BATTLE = "battle"               # Combat, conflict
    EXPOSITION = "exposition"       # Background info, explanation
    REFLECTION = "reflection"       # Internal thoughts, contemplation
    UNKNOWN = "unknown"


@dataclass
class PassageClassification:
    """Result of passage classification."""
    primary_type: PassageType
    confidence: float  # 0.0 to 1.0
    secondary_type: Optional[PassageType] = None
    
    # Feature scores that led to classification
    dialogue_score: float = 0.0
    action_score: float = 0.0
    description_score: float = 0.0
    travel_score: float = 0.0
    battle_score: float = 0.0


# Keyword patterns for classification
DIALOGUE_PATTERNS = [
    r'["\u201c][^"\u201d]+["\u201d]',  # Quoted text
    r'\bsaid\b', r'\basked\b', r'\breplied\b', r'\banswered\b',
    r'\bcried\b', r'\bshouted\b', r'\bwhispered\b', r'\bmuttered\b',
    r'\bexclaimed\b', r'\bdeclared\b', r'\bdemanded\b', r'\bbegged\b',
]

ACTION_VERBS = [
    "ran", "jumped", "grabbed", "threw", "caught", "hit", "struck",
    "pulled", "pushed", "climbed", "fell", "leaped", "rushed", "seized",
    "turned", "moved", "sprang", "dashed", "plunged", "dove", "swung",
]

DESCRIPTION_INDICATORS = [
    r'\bwas\s+\w+ing\b',  # Progressive descriptions
    r'\bseemed\b', r'\bappeared\b', r'\blooked\b',
    r'\bthe\s+\w+\s+was\b',  # "The X was Y" pattern
    r'\bthere\s+was\b', r'\bthere\s+were\b',
]

TRAVEL_KEYWORDS = [
    "walked", "went", "traveled", "journeyed", "rode", "marched",
    "path", "road", "trail", "way", "distance", "miles", "leagues",
    "east", "west", "north", "south", "onwards", "forward", "onward",
    "mountain", "forest", "river", "valley", "hill", "plain",
]

BATTLE_KEYWORDS = [
    "sword", "blade", "arrow", "spear", "shield", "armor", "helm",
    "fought", "battle", "war", "attack", "defend", "enemy", "foe",
    "slew", "killed", "wounded", "blood", "death", "fell", "charge",
    "army", "soldiers", "warriors", "host", "orcs", "goblins",
]

REFLECTION_INDICATORS = [
    r'\bthought\b', r'\bwondered\b', r'\bremembered\b', r'\bfelt\b',
    r'\brealized\b', r'\bknew\b', r'\bunderstood\b', r'\bbelieved\b',
    r'\bhoped\b', r'\bfeared\b', r'\bdreamed\b', r'\bimagined\b',
]


def classify_passage(
    text: str,
    doc: Optional[Doc] = None,
    use_llm: bool = False,
) -> PassageClassification:
    """
    Classify a passage into a narrative type.
    
    Args:
        text: The passage text
        doc: Optional pre-processed spaCy Doc
        use_llm: Whether to use LLM for enhanced classification
        
    Returns:
        PassageClassification with type and confidence
    """
    text_lower = text.lower()
    
    # Calculate scores for each type
    scores = {
        PassageType.DIALOGUE: _score_dialogue(text, text_lower),
        PassageType.ACTION: _score_action(text_lower),
        PassageType.DESCRIPTION: _score_description(text, text_lower),
        PassageType.TRAVEL: _score_travel(text_lower),
        PassageType.BATTLE: _score_battle(text_lower),
        PassageType.REFLECTION: _score_reflection(text_lower),
    }
    
    # Get primary and secondary types
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary_type, primary_score = sorted_types[0]
    secondary_type, secondary_score = sorted_types[1] if len(sorted_types) > 1 else (None, 0)
    
    # Calculate confidence (normalized)
    total_score = sum(scores.values())
    confidence = primary_score / total_score if total_score > 0 else 0
    
    # If confidence is very low, mark as unknown
    if primary_score < 0.5:
        primary_type = PassageType.UNKNOWN
        confidence = 0.0
    
    return PassageClassification(
        primary_type=primary_type,
        confidence=min(1.0, confidence),
        secondary_type=secondary_type if secondary_score > 0.3 else None,
        dialogue_score=scores[PassageType.DIALOGUE],
        action_score=scores[PassageType.ACTION],
        description_score=scores[PassageType.DESCRIPTION],
        travel_score=scores[PassageType.TRAVEL],
        battle_score=scores[PassageType.BATTLE],
    )


def _score_dialogue(text: str, text_lower: str) -> float:
    """Score how dialogue-like a passage is."""
    score = 0.0
    
    # Check for quoted text
    quotes = re.findall(r'["\u201c][^"\u201d]+["\u201d]', text)
    if quotes:
        # Calculate proportion of text that is quoted
        quoted_chars = sum(len(q) for q in quotes)
        quote_ratio = quoted_chars / len(text) if text else 0
        score += quote_ratio * 3.0  # Strong indicator
    
    # Check for speech verbs
    for pattern in DIALOGUE_PATTERNS[1:]:  # Skip the quote pattern
        if re.search(pattern, text_lower):
            score += 0.3
    
    return score


def _score_action(text_lower: str) -> float:
    """Score how action-oriented a passage is."""
    score = 0.0
    
    # Count action verbs
    for verb in ACTION_VERBS:
        if verb in text_lower:
            score += 0.4
    
    # Short sentences often indicate action
    sentences = text_lower.split('.')
    short_sentences = sum(1 for s in sentences if len(s.split()) < 10 and len(s.split()) > 3)
    if short_sentences > len(sentences) / 2:
        score += 0.5
    
    return min(score, 3.0)  # Cap the score


def _score_description(text: str, text_lower: str) -> float:
    """Score how descriptive a passage is."""
    score = 0.0
    
    # Check for descriptive patterns
    for pattern in DESCRIPTION_INDICATORS:
        matches = re.findall(pattern, text_lower)
        score += len(matches) * 0.3
    
    # Adjective density (approximate)
    words = text_lower.split()
    # Common descriptive adjectives
    desc_adjs = ["great", "dark", "long", "small", "old", "new", "high", "deep",
                 "wide", "narrow", "bright", "dim", "cold", "warm", "vast"]
    adj_count = sum(1 for w in words if w in desc_adjs)
    score += adj_count * 0.2
    
    # Longer sentences often indicate description
    if len(words) > 30:
        score += 0.3
    
    return score


def _score_travel(text_lower: str) -> float:
    """Score how travel-related a passage is."""
    score = 0.0
    
    for keyword in TRAVEL_KEYWORDS:
        if keyword in text_lower:
            score += 0.35
    
    # Direction words are strong indicators
    directions = ["east", "west", "north", "south", "onwards", "forward"]
    for d in directions:
        if d in text_lower:
            score += 0.3
    
    return min(score, 3.0)


def _score_battle(text_lower: str) -> float:
    """Score how battle-related a passage is."""
    score = 0.0
    
    for keyword in BATTLE_KEYWORDS:
        if keyword in text_lower:
            score += 0.4
    
    return min(score, 3.5)


def _score_reflection(text_lower: str) -> float:
    """Score how reflective/contemplative a passage is."""
    score = 0.0
    
    for pattern in REFLECTION_INDICATORS:
        if re.search(pattern, text_lower):
            score += 0.35
    
    # First person pronouns can indicate reflection
    first_person = ["i", "me", "my", "myself"]
    for pronoun in first_person:
        if f" {pronoun} " in f" {text_lower} ":
            score += 0.2
    
    return score


def classify_passages_batch(
    passages: list[str],
    docs: Optional[list[Doc]] = None,
) -> list[PassageClassification]:
    """
    Classify multiple passages.
    
    Args:
        passages: List of passage texts
        docs: Optional list of pre-processed spaCy Docs
        
    Returns:
        List of PassageClassification objects
    """
    results = []
    for i, text in enumerate(passages):
        doc = docs[i] if docs else None
        results.append(classify_passage(text, doc))
    return results
