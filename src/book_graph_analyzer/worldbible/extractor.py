"""World Bible Extractor.

Extracts world-building rules and patterns from text using:
1. Pattern matching for identifying relevant passages
2. LLM synthesis for inferring rules from multiple passages
3. Entity linking for connecting rules to characters/places
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
from collections import defaultdict

from ..config import get_settings
from ..llm import LLMClient
from ..ingest.loader import load_book
from ..ingest.splitter import split_into_passages
from .models import (
    WorldBible,
    WorldRule,
    WorldBibleCategory,
    CulturalProfile,
    MagicSystem,
    GeographyEntry,
    SourcePassage,
)
from .patterns import PatternMatcher, PatternMatch


@dataclass
class ExtractionConfig:
    """Configuration for world bible extraction."""
    use_llm: bool = True
    min_passages_for_rule: int = 2  # Minimum passages to support a rule
    confidence_threshold: float = 0.5
    max_passages_per_category: int = 100  # Limit for LLM synthesis
    

@dataclass
class CategorizedPassage:
    """A passage categorized by its world-building content."""
    text: str
    book: str
    location: str
    categories: dict[WorldBibleCategory, float]  # category -> confidence
    matches: list[PatternMatch] = field(default_factory=list)


class WorldBibleExtractor:
    """Extracts world bible content from text.
    
    Usage:
        extractor = WorldBibleExtractor()
        bible = extractor.extract_from_file("hobbit.txt", "Middle-earth")
        bible.save("middle_earth_bible.json")
    """
    
    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        """Initialize the extractor.
        
        Args:
            config: Extraction configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config or ExtractionConfig()
        self.progress = progress_callback or (lambda x: None)
        self.settings = get_settings()
        self.matcher = PatternMatcher()
    
    def extract_from_file(
        self,
        file_path: Path | str,
        world_name: str,
    ) -> WorldBible:
        """Extract world bible from a text file.
        
        Args:
            file_path: Path to the text file
            world_name: Name of the fictional world
            
        Returns:
            Populated WorldBible
        """
        file_path = Path(file_path)
        self.progress(f"Loading {file_path.name}...")
        
        text = load_book(file_path)
        return self.extract_from_text(text, world_name, source_name=file_path.name)
    
    def extract_from_text(
        self,
        text: str,
        world_name: str,
        source_name: str = "unknown",
    ) -> WorldBible:
        """Extract world bible from text.
        
        Args:
            text: The full text to analyze
            world_name: Name of the fictional world
            source_name: Name of the source text
            
        Returns:
            Populated WorldBible
        """
        bible = WorldBible(
            name=world_name,
            sources=[source_name],
            last_updated=datetime.now().isoformat(),
        )
        
        # Split into passages
        self.progress("Splitting into passages...")
        passages = split_into_passages(text, source_name)
        
        # Categorize passages
        self.progress("Categorizing passages...")
        categorized = self._categorize_passages(passages, source_name)
        
        # Group by category
        by_category: dict[WorldBibleCategory, list[CategorizedPassage]] = defaultdict(list)
        for cp in categorized:
            for cat in cp.categories:
                by_category[cat].append(cp)
        
        self.progress(f"Found passages for {len(by_category)} categories")
        
        # Extract rules for each category
        for cat, cat_passages in by_category.items():
            self.progress(f"Processing {cat.value}: {len(cat_passages)} passages")
            
            if self.config.use_llm and len(cat_passages) >= self.config.min_passages_for_rule:
                # Use LLM to synthesize rules
                rules = self._synthesize_rules_llm(cat, cat_passages[:self.config.max_passages_per_category])
                for rule in rules:
                    bible.add_rule(rule)
            else:
                # Fall back to pattern-based extraction
                rules = self._extract_rules_patterns(cat, cat_passages)
                for rule in rules:
                    bible.add_rule(rule)
        
        # Extract cultural profiles
        if WorldBibleCategory.CULTURE in by_category:
            self.progress("Extracting cultural profiles...")
            cultures = self._extract_cultures(by_category[WorldBibleCategory.CULTURE])
            for culture in cultures:
                bible.cultures[culture.id] = culture
        
        # Extract magic systems
        if WorldBibleCategory.MAGIC in by_category:
            self.progress("Extracting magic systems...")
            magic = self._extract_magic_systems(by_category[WorldBibleCategory.MAGIC])
            for system in magic:
                bible.magic_systems[system.id] = system
        
        # Extract geography
        if WorldBibleCategory.GEOGRAPHY in by_category:
            self.progress("Extracting geography...")
            geography = self._extract_geography(by_category[WorldBibleCategory.GEOGRAPHY])
            for entry in geography:
                bible.geography[entry.id] = entry
        
        return bible
    
    def _categorize_passages(
        self,
        passages: list,
        source_name: str,
    ) -> list[CategorizedPassage]:
        """Categorize passages by world-building content."""
        categorized = []
        
        for passage in passages:
            # Get category scores
            scores = self.matcher.classify_passage(passage.text)
            
            if scores:  # Has some world-building content
                matches = self.matcher.extract_all(passage.text)
                
                categorized.append(CategorizedPassage(
                    text=passage.text,
                    book=source_name,
                    location=f"Ch.{passage.chapter_num}, P{passage.paragraph_num}",
                    categories=scores,
                    matches=matches,
                ))
        
        return categorized
    
    def _synthesize_rules_llm(
        self,
        category: WorldBibleCategory,
        passages: list[CategorizedPassage],
    ) -> list[WorldRule]:
        """Use LLM to synthesize rules from passages."""
        if not passages:
            return []
        
        # Build context from passages
        passage_texts = [f"- {p.text[:300]}..." if len(p.text) > 300 else f"- {p.text}" 
                        for p in passages[:20]]  # Limit context size
        context = "\n".join(passage_texts)
        
        prompt = f"""Based on these passages from a fantasy novel, extract specific world-building rules about {category.value}.

For each rule you identify, provide:
1. A short title (5-10 words)
2. A clear description of the rule
3. Any constraints or exceptions mentioned

Passages:
{context}

Return your answer as a JSON array of objects with keys: "title", "description", "constraints" (array), "exceptions" (array).
Only include rules that are clearly supported by the text. Be specific, not generic.

JSON array:"""

        # Use unified LLM client
        llm = LLMClient()
        response_text = llm.generate(prompt, temperature=0.3, timeout=120.0)
        
        if not response_text:
            return self._extract_rules_patterns(category, passages)
        
        # Extract JSON from response
        rules_data = llm.extract_json(response_text)
        
        if not rules_data or not isinstance(rules_data, list):
            return self._extract_rules_patterns(category, passages)
        
        # Convert to WorldRule objects
        rules = []
        for i, item in enumerate(rules_data):
            if isinstance(item, dict) and "title" in item:
                rule_id = f"{category.value}_{i}"
                
                # Find supporting passages
                sources = []
                for p in passages[:5]:  # Limit source passages
                    sources.append(SourcePassage(
                        text=p.text[:200],
                        book=p.book,
                        location=p.location,
                    ))
                
                rules.append(WorldRule(
                    id=rule_id,
                    title=item.get("title", "Unknown"),
                    description=item.get("description", ""),
                    category=category,
                    source_passages=sources,
                    confidence=0.8,
                    constraints=item.get("constraints", []),
                    exceptions=item.get("exceptions", []),
                ))
        
        return rules
    
    def _extract_rules_patterns(
        self,
        category: WorldBibleCategory,
        passages: list[CategorizedPassage],
    ) -> list[WorldRule]:
        """Extract rules using pattern matching only."""
        rules = []
        rule_texts_seen = set()
        
        for i, passage in enumerate(passages):
            for match in passage.matches:
                if match.category == category and match.pattern_name in ("capability", "definition", "exclusivity", "prohibition", "always_never"):
                    # Avoid duplicates
                    if match.text.lower() in rule_texts_seen:
                        continue
                    rule_texts_seen.add(match.text.lower())
                    
                    rule_id = f"{category.value}_pattern_{len(rules)}"
                    
                    rules.append(WorldRule(
                        id=rule_id,
                        title=match.text[:50] + ("..." if len(match.text) > 50 else ""),
                        description=match.text,
                        category=category,
                        source_passages=[SourcePassage(
                            text=passage.text[:200],
                            book=passage.book,
                            location=passage.location,
                        )],
                        confidence=match.confidence,
                    ))
        
        return rules
    
    def _extract_cultures(
        self,
        passages: list[CategorizedPassage],
    ) -> list[CulturalProfile]:
        """Extract cultural profiles from passages."""
        # Group passages by mentioned peoples/races
        peoples_keywords = [
            ("hobbits", ["hobbit", "halfling", "shire-folk"]),
            ("elves", ["elf", "elves", "elvish", "eldar"]),
            ("dwarves", ["dwarf", "dwarves", "dwarvish"]),
            ("men", ["men", "mankind", "mortal men"]),
            ("orcs", ["orc", "orcs", "goblin", "goblins"]),
            ("wizards", ["wizard", "wizards", "istari"]),
        ]
        
        by_people: dict[str, list[CategorizedPassage]] = defaultdict(list)
        
        for passage in passages:
            text_lower = passage.text.lower()
            for people_id, keywords in peoples_keywords:
                if any(kw in text_lower for kw in keywords):
                    by_people[people_id].append(passage)
        
        cultures = []
        for people_id, people_passages in by_people.items():
            if len(people_passages) < 2:
                continue
            
            # Extract values/customs from passages
            values = []
            customs = []
            
            for p in people_passages:
                # Look for value/custom indicators
                if any(word in p.text.lower() for word in ["love", "honor", "value", "cherish"]):
                    # Extract what they value
                    pass  # Would need more sophisticated extraction
            
            cultures.append(CulturalProfile(
                id=people_id,
                name=people_id.title(),
                source_passages=[
                    SourcePassage(text=p.text[:200], book=p.book, location=p.location)
                    for p in people_passages[:10]
                ],
            ))
        
        return cultures
    
    def _extract_magic_systems(
        self,
        passages: list[CategorizedPassage],
    ) -> list[MagicSystem]:
        """Extract magic system descriptions."""
        # For now, create a single "general magic" system
        # More sophisticated extraction would identify distinct systems
        
        if len(passages) < 2:
            return []
        
        abilities = []
        limitations = []
        
        for p in passages:
            text_lower = p.text.lower()
            if "cannot" in text_lower or "forbidden" in text_lower:
                limitations.append(p.text[:100])
            elif any(word in text_lower for word in ["can", "able", "power"]):
                abilities.append(p.text[:100])
        
        return [MagicSystem(
            id="general_magic",
            name="Magic",
            abilities=abilities[:5],
            limitations=limitations[:5],
            source_passages=[
                SourcePassage(text=p.text[:200], book=p.book, location=p.location)
                for p in passages[:10]
            ],
        )]
    
    def _extract_geography(
        self,
        passages: list[CategorizedPassage],
    ) -> list[GeographyEntry]:
        """Extract geography entries."""
        # Look for named locations
        location_pattern = re.compile(
            r"(the\s+)?([\w\s]+)\s+(is|was|lies|stands?)\s+(in|on|near|by|at)\s+",
            re.IGNORECASE
        )
        
        locations: dict[str, list[CategorizedPassage]] = defaultdict(list)
        
        for p in passages:
            for match in location_pattern.finditer(p.text):
                location_name = match.group(2).strip()
                if len(location_name) > 2 and location_name[0].isupper():
                    locations[location_name].append(p)
        
        entries = []
        for name, loc_passages in locations.items():
            if len(loc_passages) < 1:
                continue
            
            entries.append(GeographyEntry(
                id=name.lower().replace(" ", "_"),
                name=name,
                source_passages=[
                    SourcePassage(text=p.text[:200], book=p.book, location=p.location)
                    for p in loc_passages[:5]
                ],
            ))
        
        return entries[:20]  # Limit number of entries
    
    def save_bible(self, bible: WorldBible, path: Path | str) -> None:
        """Save world bible to file."""
        path = Path(path)
        bible.save(path)
    
    def load_bible(self, path: Path | str) -> WorldBible:
        """Load world bible from file."""
        path = Path(path)
        return WorldBible.load(path)
