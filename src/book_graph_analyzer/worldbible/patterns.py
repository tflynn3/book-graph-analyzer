"""Pattern matching for world-building extraction.

Uses keyword patterns and linguistic markers to identify
passages that contain world-building information.
"""

import re
from dataclasses import dataclass
from typing import Callable

from .models import WorldBibleCategory


@dataclass
class PatternMatch:
    """A matched pattern in text."""
    text: str
    category: WorldBibleCategory
    pattern_name: str
    start: int
    end: int
    confidence: float = 0.8


class PatternMatcher:
    """Matches patterns that indicate world-building content.
    
    Uses a combination of:
    - Keyword patterns (specific words/phrases)
    - Structural patterns (definitional statements, rules)
    - Entity co-occurrence (magic + constraint words)
    """
    
    # Keywords that suggest different categories
    CATEGORY_KEYWORDS = {
        WorldBibleCategory.MAGIC: [
            r"\b(magic|spell|enchant|curse|charm|wizard|sorcerer|witch)",
            r"\b(power|ring of power|staff|wand|incantation)",
            r"\b(invisible|vanish|appear|transform|heal)",
            r"\b(foresight|prophecy|vision|dream|foretold)",
        ],
        WorldBibleCategory.CULTURE: [
            r"\b(custom|tradition|ritual|ceremony|festival)",
            r"\b(honor|shame|duty|law|forbidden|taboo)",
            r"\b(marriage|death|birth|coming of age)",
            r"\b(greeting|farewell|hospitality|gift)",
            r"\b(elves?|dwarves?|hobbits?|men|orcs?)\s+(always|never|typically)",
        ],
        WorldBibleCategory.GEOGRAPHY: [
            r"\b(leagues?|miles?|days?' journey|travel)",
            r"\b(mountain|river|forest|sea|lake|valley|pass)",
            r"\b(east|west|north|south)\s+of",
            r"\b(road|path|way|route)\s+to",
            r"\b(border|frontier|realm|kingdom|land)",
        ],
        WorldBibleCategory.TECHNOLOGY: [
            r"\b(forge|smith|craft|make|build|construct)",
            r"\b(weapon|armor|sword|bow|shield)",
            r"\b(ship|boat|cart|wagon)",
            r"\b(mithril|steel|iron|gold|silver)",
        ],
        WorldBibleCategory.COSMOLOGY: [
            r"\b(god|valar|maiar|eru|iluvatar)",
            r"\b(creation|beginning|end|world|earth)",
            r"\b(immortal|mortal|death|afterlife|fate)",
            r"\b(light|dark|shadow|void|flame)",
            r"\b(sun|moon|stars?|heaven|sky)",
        ],
        WorldBibleCategory.HISTORY: [
            r"\b(age|year|era|time|long ago|ancient)",
            r"\b(war|battle|siege|victory|defeat)",
            r"\b(king|queen|lord|steward)\s+\w+\s+(reign|rule|fell)",
            r"\b(first|second|third)\s+age",
        ],
        WorldBibleCategory.CREATURES: [
            r"\b(dragon|spider|wolf|warg|eagle|bear)",
            r"\b(troll|goblin|orc|balrog|nazgul|wraith)",
            r"\b(ent|eagle|raven|thrush)",
            r"\b(creature|beast|monster)",
        ],
        WorldBibleCategory.LANGUAGE: [
            r"\b(tongue|language|speech|word|name)",
            r"\b(elvish|sindarin|quenya|dwarvish|khuzdul)",
            r"\b(rune|inscription|written|script)",
            r"\b(meaning|translate|call|named)",
        ],
    }
    
    # Patterns that indicate definitional/rule statements
    RULE_PATTERNS = [
        # "X cannot/can Y"
        (r"(\w+(?:\s+\w+)*)\s+(cannot|can|may|must|shall|will)\s+(not\s+)?([\w\s]+)", "capability"),
        # "X is/are Y" (definitional)
        (r"(the\s+)?(\w+(?:\s+\w+)*)\s+(is|are|was|were)\s+(always|never|immortal|mortal|[\w\s]+)", "definition"),
        # "Only X can Y"
        (r"only\s+([\w\s]+)\s+(can|may|could)\s+([\w\s]+)", "exclusivity"),
        # "No X can/may Y"
        (r"no\s+([\w\s]+)\s+(can|may|could)\s+([\w\s]+)", "prohibition"),
        # "If X then Y"
        (r"if\s+([\w\s,]+)\s+then\s+([\w\s,]+)", "conditional"),
        # "X always/never Y"
        (r"(\w+(?:\s+\w+)*)\s+(always|never)\s+([\w\s]+)", "always_never"),
        # "It is said/known that"
        (r"it\s+(is|was)\s+(said|known|believed|written)\s+that\s+([\w\s,]+)", "lore"),
    ]
    
    def __init__(self):
        """Initialize the pattern matcher."""
        self._category_patterns: dict[WorldBibleCategory, list[re.Pattern]] = {}
        self._rule_patterns: list[tuple[re.Pattern, str]] = []
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile all regex patterns."""
        for cat, patterns in self.CATEGORY_KEYWORDS.items():
            self._category_patterns[cat] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        for pattern, name in self.RULE_PATTERNS:
            self._rule_patterns.append((re.compile(pattern, re.IGNORECASE), name))
    
    def find_category_matches(self, text: str) -> list[PatternMatch]:
        """Find category keyword matches in text."""
        matches = []
        
        for cat, patterns in self._category_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    matches.append(PatternMatch(
                        text=match.group(0),
                        category=cat,
                        pattern_name="keyword",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.7,
                    ))
        
        return matches
    
    def find_rule_matches(self, text: str) -> list[PatternMatch]:
        """Find rule/definition patterns in text."""
        matches = []
        
        for pattern, name in self._rule_patterns:
            for match in pattern.finditer(text):
                # Determine category from surrounding context
                context = text[max(0, match.start() - 50):min(len(text), match.end() + 50)]
                category = self._infer_category(context)
                
                matches.append(PatternMatch(
                    text=match.group(0),
                    category=category,
                    pattern_name=name,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                ))
        
        return matches
    
    def _infer_category(self, text: str) -> WorldBibleCategory:
        """Infer the most likely category from context."""
        category_scores: dict[WorldBibleCategory, int] = {}
        
        for cat, patterns in self._category_patterns.items():
            score = 0
            for pattern in patterns:
                score += len(pattern.findall(text))
            if score > 0:
                category_scores[cat] = score
        
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return WorldBibleCategory.THEMES  # Default fallback
    
    def classify_passage(self, text: str) -> dict[WorldBibleCategory, float]:
        """Classify a passage by its world-building categories.
        
        Returns a dict of category -> confidence scores.
        """
        scores: dict[WorldBibleCategory, float] = {}
        
        # Check category keywords
        for cat, patterns in self._category_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = pattern.findall(text)
                score += len(matches) * 0.2
            
            if score > 0:
                scores[cat] = min(score, 1.0)
        
        # Boost score if rule patterns are found
        rule_matches = self.find_rule_matches(text)
        for match in rule_matches:
            if match.category in scores:
                scores[match.category] = min(scores[match.category] + 0.3, 1.0)
            else:
                scores[match.category] = 0.5
        
        return scores
    
    def is_world_building_passage(self, text: str, threshold: float = 0.3) -> bool:
        """Check if a passage likely contains world-building content."""
        scores = self.classify_passage(text)
        return any(score >= threshold for score in scores.values())
    
    def extract_all(self, text: str) -> list[PatternMatch]:
        """Extract all pattern matches from text."""
        matches = []
        matches.extend(self.find_category_matches(text))
        matches.extend(self.find_rule_matches(text))
        
        # Deduplicate overlapping matches
        return self._deduplicate_matches(matches)
    
    def _deduplicate_matches(self, matches: list[PatternMatch]) -> list[PatternMatch]:
        """Remove overlapping matches, keeping highest confidence."""
        if not matches:
            return []
        
        # Sort by start position, then by confidence (descending)
        sorted_matches = sorted(matches, key=lambda m: (m.start, -m.confidence))
        
        result = []
        for match in sorted_matches:
            # Check if overlaps with any accepted match
            overlaps = False
            for accepted in result:
                if not (match.end <= accepted.start or match.start >= accepted.end):
                    overlaps = True
                    break
            
            if not overlaps:
                result.append(match)
        
        return result
