"""
World Bible Extractor

Extracts world-building rules and patterns from text using
keyword-based passage gathering and optional LLM synthesis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
import json
import re
import uuid

from .models import WorldBible, WorldRule, CulturalProfile, WorldBibleCategory, SourcePassage
from .categories import CATEGORY_KEYWORDS, CATEGORY_PROMPTS, CULTURE_EXTRACTION_PROMPT


@dataclass
class ExtractionConfig:
    """Configuration for world bible extraction."""
    use_llm: bool = False
    llm_model: str = "llama3.1:8b"
    min_passages_per_category: int = 5
    max_passages_per_category: int = 50
    min_keyword_matches: int = 1


class WorldBibleExtractor:
    """
    Extracts world-building information from text.
    
    Can operate in two modes:
    1. Keyword-based: Finds relevant passages and groups them
    2. LLM-assisted: Uses LLM to synthesize rules from passages
    
    Usage:
        extractor = WorldBibleExtractor()
        
        # From passages
        bible = extractor.extract_from_passages(passages, "Middle-earth")
        
        # From file
        bible = extractor.extract_from_file("hobbit.txt", "Middle-earth")
    """
    
    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the extractor.
        
        Args:
            config: Extraction configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config or ExtractionConfig()
        self.progress_callback = progress_callback
        self._llm_client = None
    
    def _report(self, message: str):
        """Report progress."""
        if self.progress_callback:
            self.progress_callback(message)
    
    def extract_from_text(
        self,
        text: str,
        world_name: str,
        book_title: str = "Unknown",
    ) -> WorldBible:
        """
        Extract world bible from text.
        
        Args:
            text: Full text to analyze
            world_name: Name for the world
            book_title: Book this text is from
            
        Returns:
            WorldBible with extracted rules
        """
        # Split into passages (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        
        self._report(f"Processing {len(paragraphs)} passages...")
        
        # Create passage objects
        passages = []
        for i, para in enumerate(paragraphs):
            passages.append({
                'id': f"{book_title}_{i}",
                'text': para,
                'book': book_title,
            })
        
        return self.extract_from_passages(passages, world_name, [book_title])
    
    def extract_from_passages(
        self,
        passages: list[dict],  # {'id': str, 'text': str, 'book': str}
        world_name: str,
        source_books: list[str] = None,
    ) -> WorldBible:
        """
        Extract world bible from passages.
        
        Args:
            passages: List of passage dicts with id, text, book
            world_name: Name for the world
            source_books: List of source book titles
            
        Returns:
            WorldBible with extracted rules
        """
        bible = WorldBible(name=world_name, source_books=source_books or [])
        
        # Find relevant passages for each category
        for category in WorldBibleCategory:
            self._report(f"Analyzing {category.value}...")
            
            relevant = self._find_relevant_passages(passages, category)
            
            if len(relevant) >= self.config.min_passages_per_category:
                # Limit passages
                if len(relevant) > self.config.max_passages_per_category:
                    relevant = relevant[:self.config.max_passages_per_category]
                
                if self.config.use_llm:
                    rules = self._extract_rules_llm(relevant, category)
                else:
                    rules = self._extract_rules_keyword(relevant, category)
                
                for rule in rules:
                    bible.add_rule(rule)
                
                self._report(f"  Found {len(rules)} rules for {category.value}")
        
        # Extract cultural profiles
        self._report("Extracting cultural profiles...")
        cultures = self._extract_cultures(passages)
        for culture in cultures:
            bible.add_culture(culture)
        
        return bible
    
    def extract_from_file(
        self,
        file_path: str | Path,
        world_name: str,
        encoding: str = "utf-8",
    ) -> WorldBible:
        """Extract from a text file."""
        path = Path(file_path)
        
        with open(path, 'r', encoding=encoding) as f:
            text = f.read()
        
        return self.extract_from_text(text, world_name, path.stem)
    
    def _find_relevant_passages(
        self,
        passages: list[dict],
        category: WorldBibleCategory,
    ) -> list[dict]:
        """Find passages relevant to a category using keywords."""
        keywords = CATEGORY_KEYWORDS.get(category, [])
        relevant = []
        
        for passage in passages:
            text_lower = passage['text'].lower()
            
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in text_lower)
            
            if matches >= self.config.min_keyword_matches:
                passage_copy = passage.copy()
                passage_copy['keyword_matches'] = matches
                relevant.append(passage_copy)
        
        # Sort by relevance
        relevant.sort(key=lambda p: -p['keyword_matches'])
        
        return relevant
    
    def _extract_rules_keyword(
        self,
        passages: list[dict],
        category: WorldBibleCategory,
    ) -> list[WorldRule]:
        """
        Extract rules using keyword-based pattern matching.
        
        This is a simpler approach that groups passages by topic
        without LLM synthesis.
        """
        rules = []
        
        # Group passages by key terms
        keywords = CATEGORY_KEYWORDS.get(category, [])
        grouped = {}
        
        for passage in passages:
            text_lower = passage['text'].lower()
            
            # Find most relevant keyword
            best_kw = None
            best_count = 0
            for kw in keywords:
                count = text_lower.count(kw)
                if count > best_count:
                    best_count = count
                    best_kw = kw
            
            if best_kw:
                if best_kw not in grouped:
                    grouped[best_kw] = []
                grouped[best_kw].append(passage)
        
        # Create a rule for each significant grouping
        for keyword, group_passages in grouped.items():
            if len(group_passages) >= 2:  # At least 2 passages mention it
                rule_id = f"{category.value}_{keyword}_{uuid.uuid4().hex[:6]}"
                
                rule = WorldRule(
                    id=rule_id,
                    category=category,
                    title=f"References to {keyword}",
                    description=f"Found {len(group_passages)} passages referencing '{keyword}' in the context of {category.value}.",
                    extraction_method="keyword",
                    confidence=min(0.3 + len(group_passages) * 0.1, 0.7),
                )
                
                for p in group_passages[:5]:  # Limit source passages
                    rule.add_source(p['id'], p['text'], p['book'])
                
                rules.append(rule)
        
        return rules
    
    def _extract_rules_llm(
        self,
        passages: list[dict],
        category: WorldBibleCategory,
    ) -> list[WorldRule]:
        """
        Extract rules using LLM synthesis.
        
        Sends passages to LLM and parses structured output.
        """
        try:
            import ollama
        except ImportError:
            self._report("  [Warning] ollama not available, falling back to keyword")
            return self._extract_rules_keyword(passages, category)
        
        # Build context from passages
        context = "\n\n---\n\n".join([
            f"[{p['book']} - {p['id']}]\n{p['text']}"
            for p in passages[:20]  # Limit context size
        ])
        
        prompt = CATEGORY_PROMPTS.get(category, "Extract world-building rules from these passages.")
        
        full_prompt = f"""{prompt}

PASSAGES:
{context}

Respond with a JSON array of objects, each with:
- "title": short title
- "description": full description  
- "sources": list of passage IDs that support this

JSON:"""

        try:
            response = ollama.generate(
                model=self.config.llm_model,
                prompt=full_prompt,
            )
            
            # Parse JSON from response
            response_text = response.get('response', '')
            rules = self._parse_llm_rules(response_text, category, passages)
            return rules
            
        except Exception as e:
            self._report(f"  [Warning] LLM extraction failed: {e}")
            return self._extract_rules_keyword(passages, category)
    
    def _parse_llm_rules(
        self,
        response: str,
        category: WorldBibleCategory,
        passages: list[dict],
    ) -> list[WorldRule]:
        """Parse LLM response into WorldRule objects."""
        rules = []
        
        # Try to extract JSON array
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            return rules
        
        try:
            data = json.loads(json_match.group())
            
            # Create passage lookup
            passage_map = {p['id']: p for p in passages}
            
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                rule_id = f"{category.value}_{uuid.uuid4().hex[:8]}"
                
                rule = WorldRule(
                    id=rule_id,
                    category=category,
                    title=item.get('title', 'Untitled'),
                    description=item.get('description', ''),
                    extraction_method="llm",
                    confidence=0.6,  # LLM-extracted rules get medium confidence
                )
                
                # Link to source passages
                for source_id in item.get('sources', []):
                    if source_id in passage_map:
                        p = passage_map[source_id]
                        rule.add_source(p['id'], p['text'], p['book'])
                
                rules.append(rule)
                
        except json.JSONDecodeError:
            pass
        
        return rules
    
    def _extract_cultures(self, passages: list[dict]) -> list[CulturalProfile]:
        """
        Extract cultural profiles from passages.
        
        Looks for mentions of different peoples/races and
        gathers information about each.
        """
        cultures = []
        
        # Common fantasy races/peoples to look for
        culture_names = [
            ("elves", "Elves"), ("elf", "Elves"),
            ("dwarves", "Dwarves"), ("dwarf", "Dwarves"),
            ("hobbits", "Hobbits"), ("hobbit", "Hobbits"),
            ("men", "Men"), ("human", "Men"),
            ("orcs", "Orcs"), ("orc", "Orcs"),
            ("goblins", "Goblins"), ("goblin", "Goblins"),
            ("wizards", "Wizards"), ("wizard", "Wizards"),
        ]
        
        found_cultures = {}
        
        for passage in passages:
            text_lower = passage['text'].lower()
            
            for keyword, culture_name in culture_names:
                if keyword in text_lower:
                    if culture_name not in found_cultures:
                        found_cultures[culture_name] = []
                    found_cultures[culture_name].append(passage)
        
        # Create profiles for cultures with enough mentions
        for culture_name, culture_passages in found_cultures.items():
            if len(culture_passages) >= 3:
                culture_id = culture_name.lower().replace(' ', '_')
                
                profile = CulturalProfile(
                    id=culture_id,
                    name=culture_name,
                )
                
                # Add source passages
                for p in culture_passages[:10]:
                    profile.source_passages.append(SourcePassage(
                        passage_id=p['id'],
                        text=p['text'][:300],
                        book=p['book'],
                    ))
                
                cultures.append(profile)
        
        return cultures
    
    def save_bible(self, bible: WorldBible, output_path: str | Path):
        """Save world bible to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(bible.to_json())
    
    def load_bible(self, input_path: str | Path) -> WorldBible:
        """Load world bible from JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            return WorldBible.from_json(f.read())
