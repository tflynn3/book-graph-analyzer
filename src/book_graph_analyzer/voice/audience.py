"""
Audience & Context Classification for Dialogue

Classifies who a character is speaking to (audience_type) and
what kind of speech act it is (context_type).
"""

import re
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Audience type classification
# ---------------------------------------------------------------------------

# Tolkien-specific character/race name clusters
_AUDIENCE_KEYWORDS: dict[str, list[str]] = {
    "hobbit": [
        "hobbit", "halfling", "shire", "bilbo", "frodo", "samwise", "sam",
        "merry", "meriadoc", "pippin", "peregrin", "took", "baggins",
        "bag end", "brandybuck", "gamgee",
    ],
    "elf": [
        "elf", "elves", "elven", "elvish", "elrond", "galadriel", "legolas",
        "arwen", "celeborn", "glorfindel", "erestor", "haldir", "rivendell",
        "imladris", "lothlórien", "lothlorien", "lindon", "silvan", "noldor",
        "sindar", "eldar", "firstborn",
    ],
    "man": [
        "man", "men", "human", "mortal", "aragorn", "strider", "boromir",
        "faramir", "théoden", "theoden", "éomer", "eomer", "éowyn", "eowyn",
        "gondor", "rohan", "númenor", "numenor", "dúnedain", "dunedain",
        "ranger",
    ],
    "dwarf": [
        "dwarf", "dwarves", "dwarven", "dwarfish", "gimli", "glóin", "gloin",
        "balin", "dwalin", "thorin", "erebor", "moria", "khazad",
        "durin", "longbeard",
    ],
    "enemy": [
        "orc", "orcs", "orcish", "goblin", "goblinlike", "balrog", "sauron",
        "morgoth", "mordor", "nazgûl", "nazgul", "wraith", "ringwraith",
        "uruk", "troll", "warg", "enemy", "enemies", "foe", "foes",
        "shadow", "dark lord",
    ],
    "self": [
        "himself", "herself", "myself", "itself",
    ],
    "prayer": [
        "ilúvatar", "iluvatar", "eru", "valar", "manwë", "manwe", "varda",
        "elbereth", "stars", "o lord", "o great",
    ],
}

# Context-type detection patterns (order matters — first match wins)
# More specific patterns come first to avoid false positives.
_CONTEXT_PATTERNS: list[tuple[str, list[str]]] = [
    ("farewell", [
        r"\b(farewell|goodbye|good-bye|namárië|until we meet|parting|depart|go well|godspeed|fare thee well|fare well)\b",
    ]),
    ("comfort", [
        r"\b(fear not|do not fear|be not afraid|all shall be well|do not worry)\b",
        r"\b(peace|rest|comfort|solace|burden|grief|sorrow|weep|tears|courage|endure)\b",
    ]),
    ("warning", [
        r"\b(beware|peril|careful|heed|caution|lest)\b",
        r"\b(danger|shadow|doom|trap|ambush|dark lord|wraith)\b",
    ]),
    ("crisis", [
        r"\b(run|flee|hurry|quick|danger|attack|help|aid|escape|save)\b",
        r"\b(no time|urgent|at once|immediately)\b",
    ]),
    ("command", [
        r"^(go|come|stop|stay|take|bring|follow|hold|stand|rise|sit|listen|look|hear|move)\b",
        r"\b(do not|never|you must|thou shalt|thou must)\b",
    ]),
    ("explanation", [
        r"\b(because|for|thus|therefore|hence|this is|the reason|let me explain|you see|know that|understand)\b",
        r"\b(long ago|once|history|legend|tale|story|as you know|recall|remember)\b",
    ]),
]


AUDIENCE_TYPES = frozenset({"hobbit", "elf", "man", "dwarf", "enemy", "neutral", "self", "prayer"})
CONTEXT_TYPES = frozenset({"crisis", "explanation", "command", "comfort", "warning", "farewell", "statement"})


def classify_audience_type(
    dialogue_text: str,
    context_before: str,
    context_after: str,
    speaker: Optional[str] = None,
) -> str:
    """
    Classify the audience type for a dialogue line.

    Returns one of: 'hobbit' | 'elf' | 'man' | 'dwarf' | 'enemy' |
                    'neutral' | 'self' | 'prayer'
    """
    # Check for self-referential speech
    if re.search(r"\b(himself|herself|myself|itself|to himself|to herself)\b",
                 context_before + " " + context_after, re.IGNORECASE):
        return "self"

    # Combine context windows for audience detection
    combined = (context_before + " " + context_after).lower()

    # Score each audience type by keyword hits
    scores: dict[str, int] = {}
    for audience, keywords in _AUDIENCE_KEYWORDS.items():
        if audience == "self":
            continue
        count = sum(1 for kw in keywords if kw in combined)
        if count > 0:
            scores[audience] = count

    if scores:
        return max(scores, key=lambda k: scores[k])

    return "neutral"


def classify_context_type(dialogue_text: str) -> str:
    """
    Classify the context type (speech act) of a dialogue line.

    Returns one of: 'crisis' | 'explanation' | 'command' | 'comfort' |
                    'warning' | 'farewell' | 'statement'
    """
    text_lower = dialogue_text.lower().strip()

    for context_type, patterns in _CONTEXT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return context_type

    return "statement"


@dataclass
class AudienceClassification:
    """Result of audience + context classification for a dialogue line."""
    audience_type: str   # One of AUDIENCE_TYPES
    context_type: str    # One of CONTEXT_TYPES
    confidence: float    # 0.0 - 1.0


def classify_dialogue_line(
    dialogue_text: str,
    context_before: str,
    context_after: str,
    speaker: Optional[str] = None,
) -> AudienceClassification:
    """
    Classify both audience type and context type for a dialogue line.
    """
    audience = classify_audience_type(
        dialogue_text, context_before, context_after, speaker
    )
    context = classify_context_type(dialogue_text)

    # Confidence: higher if we found keyword matches
    confidence = 0.8 if audience != "neutral" else 0.3

    return AudienceClassification(
        audience_type=audience,
        context_type=context,
        confidence=confidence,
    )
