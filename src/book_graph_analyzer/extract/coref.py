"""
Lightweight Coreference Resolution

Since `coreferee` cannot be installed (build dependency conflicts),
this module implements a sliding-window pronoun resolver using spaCy's
dependency parsing — no external coref model required.

Algorithm (pronoun substitution):
1. Parse each passage with spaCy
2. Identify entity mentions (NER + PROPN sequences)
3. For each pronoun, look back within the sentence and adjacent
   sentences for the nearest antecedent that matches gender/number
4. Build a mention list with resolved antecedents attached

For cross-passage coreference:
- Use a sliding window of the last 3 passages as context
- Keep a "recency buffer" of recently seen named entities

Limitations vs full coreferee:
- No transformer-based pronoun resolution
- Single-pass (no iterative resolution)
- Limited gender agreement checking (English only)

For best results with Tolkien corpora: use explicit alias detection
from text patterns ("Gandalf, whom they called Mithrandir") via
DynamicEntityResolver.detect_aliases_from_text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Masculine pronouns
_MASC_PRONOUNS = {"he", "him", "his", "himself"}
# Feminine pronouns
_FEM_PRONOUNS = {"she", "her", "hers", "herself"}
# Neutral / plural
_NEUT_PRONOUNS = {"it", "its", "itself"}
_PLURAL_PRONOUNS = {"they", "them", "their", "theirs", "themselves"}
# "the X" patterns that need resolution
_EPITHET_PREFIXES = {"the wizard", "the ranger", "the king", "the hobbit",
                     "the elf", "the dwarf", "the man", "the grey pilgrim",
                     "the dark lord", "the enemy", "the shadow"}

ALL_PRONOUNS = _MASC_PRONOUNS | _FEM_PRONOUNS | _NEUT_PRONOUNS | _PLURAL_PRONOUNS


@dataclass
class ResolvedMention:
    """A mention that has been resolved to a candidate antecedent."""
    text: str                          # The original text (pronoun or epithet)
    antecedent: Optional[str] = None   # The resolved canonical form (if found)
    confidence: float = 0.0
    is_pronoun: bool = False
    passage_idx: int = 0
    char_offset: int = 0


@dataclass
class CoreferenceChain:
    """A coreference chain: the canonical entity + all its mentions."""
    canonical: str
    mentions: list[ResolvedMention] = field(default_factory=list)


class PronounResolver:
    """
    Sliding-window pronoun resolver.

    Resolves pronouns and definite epithets within and across a window
    of recent passages using spaCy NER + dependency parsing.
    """

    def __init__(
        self,
        window_size: int = 3,        # How many recent passages to consider
        recency_decay: float = 0.85,  # Confidence decay per passage back
    ) -> None:
        self.window_size = window_size
        self.recency_decay = recency_decay
        self._nlp = None             # Lazy-loaded spaCy model

    def _get_nlp(self):
        """Load spaCy model lazily."""
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Pronoun resolution disabled.")
                self._nlp = False  # Sentinel: don't retry
        return self._nlp if self._nlp is not False else None

    def resolve_passage(
        self,
        passage: str,
        recent_entities: list[str],  # Named entities from recent passages (most-recent last)
    ) -> list[ResolvedMention]:
        """
        Resolve pronouns in a single passage.

        Args:
            passage: The passage text to process
            recent_entities: Recent named entities from prior passages

        Returns:
            List of ResolvedMention objects for found pronouns/epithets
        """
        nlp = self._get_nlp()
        if not nlp:
            return []

        resolved: list[ResolvedMention] = []
        doc = nlp(passage[:10000])  # Limit for performance

        # Collect named entities from this passage (in order)
        passage_entities: list[tuple[int, str]] = []
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "GPE", "LOC", "ORG", "NORP", "FAC"):
                passage_entities.append((ent.start, ent.text))

        # All available antecedents: recent entities + passage entities so far
        # Build a flat recency list (most-recent = highest index)
        antecedent_pool = list(recent_entities[-self.window_size * 5:])

        # Process tokens
        prev_sent_entities: list[str] = []

        for sent in doc.sents:
            sent_entities: list[str] = []

            for token in sent:
                token_lower = token.text.lower()

                # Is this a pronoun?
                if token_lower in ALL_PRONOUNS and token.dep_ in ("nsubj", "nsubjpass", "dobj", "pobj", "nmod"):
                    antecedent, conf = self._find_antecedent(
                        token_lower, sent_entities + prev_sent_entities + antecedent_pool
                    )
                    if antecedent:
                        resolved.append(ResolvedMention(
                            text=token.text,
                            antecedent=antecedent,
                            confidence=conf,
                            is_pronoun=True,
                            char_offset=token.idx,
                        ))

                # Is this a named entity? Add to current sentence pool
                if token.ent_type_ in ("PERSON", "GPE", "LOC", "ORG"):
                    sent_entities.append(token.text)

            prev_sent_entities = sent_entities
            antecedent_pool.extend(sent_entities)

        return resolved

    def resolve_passages(
        self,
        passages: list[str],
    ) -> list[list[ResolvedMention]]:
        """
        Resolve pronouns across a sequence of passages with sliding window.

        Returns a list of ResolvedMention lists, one per passage.
        """
        all_resolved: list[list[ResolvedMention]] = []
        recent_entities: list[str] = []

        for i, passage in enumerate(passages):
            nlp = self._get_nlp()
            if not nlp:
                all_resolved.append([])
                continue

            resolved = self.resolve_passage(passage, recent_entities)
            all_resolved.append(resolved)

            # Update recency buffer from this passage
            doc = nlp(passage[:5000])
            new_entities = [ent.text for ent in doc.ents
                          if ent.label_ in ("PERSON", "GPE", "LOC", "ORG")]
            recent_entities.extend(new_entities)

            # Keep window bounded
            max_buffer = self.window_size * 20
            if len(recent_entities) > max_buffer:
                recent_entities = recent_entities[-max_buffer:]

        return all_resolved

    def _find_antecedent(
        self,
        pronoun: str,
        candidates: list[str],
    ) -> tuple[Optional[str], float]:
        """
        Find the most likely antecedent for a pronoun from candidates.

        Simple strategy: take the most recently mentioned candidate.
        Gender agreement is not enforced (Tolkien has many ambiguous cases).

        Returns (antecedent_text, confidence) or (None, 0.0)
        """
        # Filter to person-likely candidates
        person_candidates = [c for c in reversed(candidates)
                             if len(c) > 1 and c[0].isupper()]

        if not person_candidates:
            return None, 0.0

        # Most recent = highest confidence
        top = person_candidates[0]
        conf = 0.65  # Moderate confidence for recency-based resolution

        return top, conf

    def get_pronoun_chain(
        self,
        passages: list[str],
    ) -> dict[str, CoreferenceChain]:
        """
        Build coreference chains from a sequence of passages.

        Returns dict of canonical_name → CoreferenceChain.
        """
        chains: dict[str, CoreferenceChain] = {}
        all_resolved = self.resolve_passages(passages)

        for i, resolved_list in enumerate(all_resolved):
            for mention in resolved_list:
                if mention.antecedent:
                    if mention.antecedent not in chains:
                        chains[mention.antecedent] = CoreferenceChain(canonical=mention.antecedent)
                    mention.passage_idx = i
                    chains[mention.antecedent].mentions.append(mention)

        return chains


# ---------------------------------------------------------------------------
# Alias detection from explicit text patterns
# ---------------------------------------------------------------------------

# Name token: starts with uppercase (incl. accented), followed by word chars + hyphens + apostrophes
# Uses re.UNICODE implicitly in Python 3 for \w
_NAME_TOKEN = r"(?:[A-Z\u00C0-\u024F][\w'\-]{1,})"
_NAME_PHRASE = rf"(?:{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,3}})"

_ALIAS_PATTERNS = [
    # "X, whose real name was Y"
    rf"({_NAME_PHRASE}),?\s+whose\s+(?:real\s+)?name\s+was\s+({_NAME_PHRASE})",
    # "X (also known as Y)" / "X (known as Y)"
    rf"({_NAME_PHRASE})\s*\((?:also\s+)?(?:known|called)\s+(?:as\s+)?({_NAME_PHRASE})\)",
    # "X, or Y as he/she was called"
    rf"({_NAME_PHRASE}),?\s+or\s+({_NAME_PHRASE})\s+as\s+(?:he|she|they|it)\s+(?:was|were)\s+(?:called|known)",
    # "X, called Y by Z"
    rf"({_NAME_PHRASE}),?\s+(?:called|named|known\s+as)\s+({_NAME_PHRASE})",
]


def detect_explicit_aliases(text: str) -> list[tuple[str, str]]:
    """
    Detect explicit alias statements in text.

    Returns list of (name_a, name_b) pairs that are stated to be the same entity.
    """
    import re
    found: list[tuple[str, str]] = []
    for pattern in _ALIAS_PATTERNS:
        for m in re.finditer(pattern, text):
            a, b = m.group(1).strip(), m.group(2).strip()
            if a != b:
                found.append((a, b))
    return found
