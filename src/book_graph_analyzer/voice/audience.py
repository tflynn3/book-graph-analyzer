"""
Audience Classification

Heuristic classifier for audience type and conversational context
of a dialogue line.  No LLM required — keyword/pattern based.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dialogue import DialogueLine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIENCE_TYPES = [
    "hobbit",
    "elf",
    "man",
    "dwarf",
    "enemy",
    "neutral",
    "self",
    "prayer",
]

CONTEXT_TYPES = [
    "crisis",
    "explanation",
    "command",
    "comfort",
    "warning",
    "farewell",
]

# Keywords / names strongly associated with each audience type
_AUDIENCE_KEYWORDS: dict[str, list[str]] = {
    "hobbit": [
        "bilbo", "frodo", "sam", "samwise", "merry", "pippin", "peregrin",
        "hobbit", "hobbits", "halfling", "halflings", "shire", "baggins",
        "gamgee", "took", "brandybuck",
    ],
    "elf": [
        "gandalf",  # Gandalf is Maia but often classed with elves in address
        "legolas", "elrond", "galadriel", "arwen", "glorfindel", "celeborn",
        "elf", "elves", "elvish", "eldar", "firstborn", "rivendell", "lothlórien",
        "lothlórien", "lorien",
    ],
    "man": [
        "aragorn", "boromir", "faramir", "theoden", "eomer", "eowyn",
        "man", "men", "dunedain", "gondor", "rohan", "mortal", "mortals",
        "king", "captain", "lord", "ranger",
    ],
    "dwarf": [
        "gimli", "gloin", "balin", "oin", "dwalin", "thorin", "dori", "nori",
        "dwarf", "dwarves", "dwarfs", "khazad", "erebor", "moria",
    ],
    "enemy": [
        "sauron", "saruman", "morgoth", "balrog", "nazgul", "orc", "orcs",
        "wraith", "dark lord", "enemy", "foe", "villain", "traitor",
    ],
    "self": [
        "myself", "mine own", "i ask myself", "i wonder", "i know not",
    ],
    "prayer": [
        "elbereth", "varda", "iluvatar", "eru", "manwe", "the valar",
        "bless", "grant me", "hear my", "o lord", "o ancient",
    ],
}

# Compiled lower-case keyword lookup per audience type
_AUDIENCE_RE: dict[str, re.Pattern] = {
    aud: re.compile(
        r"\b(" + "|".join(re.escape(kw) for kw in kws) + r")\b",
        re.IGNORECASE,
    )
    for aud, kws in _AUDIENCE_KEYWORDS.items()
}

# ---------------------------------------------------------------------------
# Context classification patterns
# ---------------------------------------------------------------------------

_IMPERATIVE_STARTS = [
    "go", "come", "run", "stop", "listen", "look", "wait", "stand",
    "do not", "don't", "never", "beware", "fly", "flee", "seek", "find",
    "follow", "stay", "leave", "hold", "keep", "turn", "move", "speak",
    "tell", "say", "fear", "trust", "grant", "forgive", "remember",
    "bring", "take", "give", "get", "make", "let", "help", "save",
    "throw", "cast", "pass", "hear", "heed", "see", "read", "open",
    "close", "guard", "protect", "hide", "rise", "fall", "walk", "halt",
]

_FAREWELL_PATTERNS = re.compile(
    r"\b(farewell|goodbye|good-bye|till we meet|safe travels?|may (your|the|your) "
    r"|namarie|namarié|go well|ride well|god speed|godspeed|adieu|until we meet)\b",
    re.IGNORECASE,
)

_COMFORT_PATTERNS = re.compile(
    r"\b(fear not|do not grieve|all will be well|there there|take heart|"
    r"courage|hope|you are safe|rest now|do not weep|be at peace|"
    r"worry not|peace|calm|be comforted)\b",
    re.IGNORECASE,
)

_WARNING_PATTERNS = re.compile(
    r"\b(beware|danger|watch out|take care|be wary|peril|threat|"
    r"do not go|do not trust|flee the|run from|escape)\b",
    re.IGNORECASE,
)

_CRISIS_PATTERNS = re.compile(
    r"\b(attack|they are upon|surrounded|no time|hurry|quickly|retreat|"
    r"help|emergency|fire|break|now or never|too late|fallen|lost|"
    r"we are lost|we shall not)\b",
    re.IGNORECASE,
)

_EXPLANATION_PATTERNS = re.compile(
    r"\b(because|therefore|thus|hence|in other words|that is to say|"
    r"you see|let me explain|the reason|it means|what this means|"
    r"understand|know that|be aware that|hear me)\b",
    re.IGNORECASE,
)


def _classify_context(text: str, is_question: bool, is_exclamation: bool) -> str:
    """Return a CONTEXT_TYPE label for a dialogue line.

    Priority order (highest to lowest):
        farewell → comfort → warning → crisis → command → explanation
    Comfort and warning are checked before command so that patterns like
    "Fear not" and "Beware" are correctly labelled rather than falling
    through to a generic imperative match.
    """
    lowered = text.lower().strip()

    # 1. Farewell — easiest wins
    if _FAREWELL_PATTERNS.search(lowered):
        return "farewell"

    # 2. Comfort — reassurance language ("fear not", "all will be well", …)
    #    Must come before command so "Fear not" isn't caught as imperative.
    if _COMFORT_PATTERNS.search(lowered):
        return "comfort"

    # 3. Warning — danger language ("beware", "peril", …)
    #    Must come before command so "Beware" isn't caught as imperative.
    if _WARNING_PATTERNS.search(lowered):
        return "warning"

    # 4. Crisis — urgent, exclamatory, panic
    if _CRISIS_PATTERNS.search(lowered) and (is_exclamation or is_question):
        return "crisis"

    # 5. Command — imperative structure (verbs not already caught above)
    for imp in _IMPERATIVE_STARTS:
        if (
            lowered.startswith(imp + " ")
            or lowered.startswith(imp + ",")
            or lowered.startswith(imp + "!")
            or lowered == imp
        ):
            return "command"

    # 6. Explanation — because / therefore / you see
    if _EXPLANATION_PATTERNS.search(lowered):
        return "explanation"

    # Fallback
    return "explanation"


def _classify_audience_from_text(
    text: str,
    addressee: str | None,
    passage_context: str,
) -> str:
    """Return an AUDIENCE_TYPE label."""
    # If we already have a named addressee, try to match it
    if addressee:
        addr_lower = addressee.lower()
        for aud_type, kws in _AUDIENCE_KEYWORDS.items():
            for kw in kws:
                if kw in addr_lower:
                    return aud_type

    # Scan passage context + text for known names
    combined = (text + " " + passage_context).lower()
    scores: dict[str, int] = {}
    for aud_type, pattern in _AUDIENCE_RE.items():
        hits = pattern.findall(combined)
        if hits:
            scores[aud_type] = len(hits)

    if scores:
        return max(scores, key=lambda k: scores[k])

    return "neutral"


def classify_audience(
    dialogue_line: "DialogueLine",
    passage_context: str = "",
) -> tuple[str, str]:
    """
    Classify the audience type and conversational context for a dialogue line.

    Args:
        dialogue_line: A DialogueLine object (must have .text, .is_question,
                       .is_exclamation; optionally .addressee).
        passage_context: Surrounding passage text for additional context.

    Returns:
        (audience_type, context_type) — both are strings from AUDIENCE_TYPES
        and CONTEXT_TYPES respectively.
    """
    addressee = getattr(dialogue_line, "addressee", None)
    text = dialogue_line.text

    audience_type = _classify_audience_from_text(text, addressee, passage_context)
    context_type = _classify_context(
        text,
        is_question=dialogue_line.is_question,
        is_exclamation=dialogue_line.is_exclamation,
    )

    return audience_type, context_type
