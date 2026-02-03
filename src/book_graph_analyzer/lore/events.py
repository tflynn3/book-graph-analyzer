"""Event extraction for temporal reasoning.

Extracts structured events from text:
- Who did what (agent, action, patient)
- When it happened (era, year, relative time)
- Temporal ordering (before/after other events)
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from ..llm import LLMClient
from .temporal import Era


@dataclass
class Event:
    """A structured event extracted from text."""
    id: str
    description: str  # Short description: "Bilbo found the Ring"
    
    # Participants
    agent: Optional[str] = None  # Who did it (Bilbo)
    action: Optional[str] = None  # What they did (found)
    patient: Optional[str] = None  # What it was done to (the Ring)
    
    # Temporal info
    era: Optional[Era] = None
    year: Optional[int] = None  # Year within era (e.g., 2941 TA)
    year_text: Optional[str] = None  # Original text ("Third Age 2941")
    
    # Source
    source_text: str = ""
    source_book: str = ""
    source_location: str = ""
    
    # Confidence
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "agent": self.agent,
            "action": self.action,
            "patient": self.patient,
            "era": self.era.value if self.era else None,
            "year": self.year,
            "year_text": self.year_text,
            "source_text": self.source_text,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            id=d["id"],
            description=d["description"],
            agent=d.get("agent"),
            action=d.get("action"),
            patient=d.get("patient"),
            era=Era(d["era"]) if d.get("era") else None,
            year=d.get("year"),
            year_text=d.get("year_text"),
            source_text=d.get("source_text", ""),
            confidence=d.get("confidence", 1.0),
        )


@dataclass
class EventRelation:
    """Temporal relationship between events."""
    event1_id: str
    relation: str  # "before", "after", "during", "causes"
    event2_id: str
    confidence: float = 1.0
    source_text: str = ""


@dataclass
class EventGraph:
    """Graph of events and their temporal relationships."""
    events: dict[str, Event] = field(default_factory=dict)
    relations: list[EventRelation] = field(default_factory=list)
    
    # Index for quick lookup
    _by_agent: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    _by_patient: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    _by_action: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    
    def add_event(self, event: Event) -> None:
        """Add an event to the graph."""
        self.events[event.id] = event
        
        if event.agent:
            self._by_agent[event.agent.lower()].append(event.id)
        if event.patient:
            self._by_patient[event.patient.lower()].append(event.id)
        if event.action:
            self._by_action[event.action.lower()].append(event.id)
    
    def add_relation(self, relation: EventRelation) -> None:
        """Add a temporal relation."""
        self.relations.append(relation)
    
    def find_events(
        self,
        agent: Optional[str] = None,
        action: Optional[str] = None,
        patient: Optional[str] = None,
    ) -> list[Event]:
        """Find events matching criteria with fuzzy matching."""
        results = []
        
        for event in self.events.values():
            matches = True
            
            if agent:
                agent_lower = agent.lower()
                event_agent = (event.agent or "").lower()
                if agent_lower not in event_agent and event_agent not in agent_lower:
                    matches = False
            
            if action and matches:
                action_lower = action.lower().rstrip('ed').rstrip('s')  # Normalize verb
                event_action = (event.action or "").lower().rstrip('ed').rstrip('s')
                if action_lower != event_action and action_lower not in event_action:
                    matches = False
            
            if patient and matches:
                patient_lower = patient.lower()
                event_patient = (event.patient or "").lower()
                
                # Skip matching if patient is just a pronoun (it, them, etc.)
                pronouns = {"it", "them", "him", "her", "this", "that"}
                patient_words = set(patient_lower.split())
                if patient_words <= pronouns or patient_words - {"the", "a", "an"} <= pronouns:
                    # Patient is just pronouns/articles - don't filter on it
                    pass
                else:
                    # Flexible matching - strip articles, check containment
                    patient_clean = patient_lower.replace("the ", "").replace("a ", "").replace("it/", "")
                    patient_event_clean = event_patient.replace("the ", "").replace("a ", "")
                    if patient_clean not in patient_event_clean and patient_event_clean not in patient_clean:
                        matches = False
            
            if matches:
                results.append(event)
        
        return results
    
    def get_ordering(self, event1_id: str, event2_id: str) -> Optional[str]:
        """Get temporal ordering between two events.
        
        Returns:
            "before" if event1 is before event2
            "after" if event1 is after event2
            "same" if simultaneous
            None if unknown
        """
        # Check direct relations
        for rel in self.relations:
            if rel.event1_id == event1_id and rel.event2_id == event2_id:
                return rel.relation
            if rel.event1_id == event2_id and rel.event2_id == event1_id:
                if rel.relation == "before":
                    return "after"
                elif rel.relation == "after":
                    return "before"
        
        # Check by year if available
        e1 = self.events.get(event1_id)
        e2 = self.events.get(event2_id)
        
        if e1 and e2:
            # Compare eras first
            if e1.era and e2.era and e1.era != Era.UNKNOWN and e2.era != Era.UNKNOWN:
                if e1.era < e2.era:
                    return "before"
                elif e1.era > e2.era:
                    return "after"
                # Same era, check years
                elif e1.year and e2.year:
                    if e1.year < e2.year:
                        return "before"
                    elif e1.year > e2.year:
                        return "after"
                    else:
                        return "same"
        
        return None
    
    def happened_before(self, event1_id: str, event2_id: str) -> Optional[bool]:
        """Check if event1 happened before event2."""
        ordering = self.get_ordering(event1_id, event2_id)
        if ordering == "before":
            return True
        elif ordering == "after":
            return False
        return None
    
    def to_dict(self) -> dict:
        return {
            "events": {k: v.to_dict() for k, v in self.events.items()},
            "relations": [
                {
                    "event1_id": r.event1_id,
                    "relation": r.relation,
                    "event2_id": r.event2_id,
                    "confidence": r.confidence,
                }
                for r in self.relations
            ],
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "EventGraph":
        graph = cls()
        for k, v in d.get("events", {}).items():
            graph.add_event(Event.from_dict(v))
        for r in d.get("relations", []):
            graph.add_relation(EventRelation(**r))
        return graph


class EventExtractor:
    """Extracts events from text using LLM.
    
    Usage:
        extractor = EventExtractor()
        graph = extractor.extract_from_text(text)
        
        # Find events
        events = graph.find_events(agent="Bilbo", action="found")
        
        # Check ordering
        is_before = graph.happened_before("bilbo_found_ring", "gollum_lost_ring")
        
        # For full books, use chunked extraction
        graph = extractor.extract_from_book(text, chunk_size=3000)
    """
    
    # Key event verbs to look for
    EVENT_VERBS = [
        "found", "lost", "took", "gave", "stole", "received",
        "killed", "died", "born", "married", "met",
        "created", "forged", "destroyed", "broke",
        "traveled", "arrived", "left", "fled", "returned",
        "fought", "won", "lost", "defeated", "conquered",
        "said", "told", "revealed", "discovered", "learned",
        "became", "transformed", "awakened",
    ]
    
    # Year patterns
    YEAR_PATTERNS = [
        r"(?:in\s+)?(?:the\s+)?(?:year\s+)?(\d+)\s+(?:of\s+the\s+)?(?:(First|Second|Third|Fourth)\s+Age|([TFS])\.?A\.?)",
        r"(?:(First|Second|Third|Fourth)\s+Age)\s+(\d+)",
        r"([TFS])\.?A\.?\s*(\d+)",
    ]
    
    def __init__(self, use_llm: bool = True, progress_callback=None):
        self.use_llm = use_llm
        self.progress_callback = progress_callback
        self._year_patterns = [re.compile(p, re.IGNORECASE) for p in self.YEAR_PATTERNS]
        self._seen_events: set[str] = set()  # Track for deduplication
    
    def extract_from_book(
        self,
        text: str,
        source_book: str = "",
        chunk_size: int = 3000,
        overlap: int = 200,
    ) -> EventGraph:
        """Extract events from a full book using chunked processing.
        
        Args:
            text: Full book text
            source_book: Name of the source book
            chunk_size: Characters per chunk
            overlap: Overlap between chunks to avoid losing context at boundaries
            
        Returns:
            EventGraph with all events and relations
        """
        graph = EventGraph()
        self._seen_events = set()
        
        # Split into chunks
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence-ending punctuation
                for i in range(end, max(start + chunk_size // 2, end - 200), -1):
                    if text[i] in '.!?"':
                        end = i + 1
                        break
            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else end
        
        total_chunks = len(chunks)
        all_events: list[Event] = []
        all_relations: list[EventRelation] = []
        
        for i, chunk in enumerate(chunks):
            if self.progress_callback:
                self.progress_callback(i + 1, total_chunks, f"Processing chunk {i + 1}/{total_chunks}")
            
            events, relations = self._extract_llm(chunk, source_book, chunk_index=i)
            
            for event in events:
                # Deduplicate based on normalized description
                event_key = self._normalize_event_key(event)
                if event_key not in self._seen_events:
                    self._seen_events.add(event_key)
                    all_events.append(event)
            
            all_relations.extend(relations)
        
        # Add all events to graph
        for event in all_events:
            graph.add_event(event)
        
        # Add relations, filtering invalid ones
        for rel in all_relations:
            if rel.event1_id in graph.events and rel.event2_id in graph.events:
                graph.add_relation(rel)
        
        # Infer additional ordering from year/era
        self._infer_temporal_ordering(graph)
        
        return graph
    
    def _normalize_event_key(self, event: Event) -> str:
        """Create a normalized key for deduplication."""
        parts = []
        if event.agent:
            parts.append(event.agent.lower().strip())
        if event.action:
            # Normalize verb tenses
            action = event.action.lower().strip()
            action = action.rstrip('ed').rstrip('s')
            parts.append(action)
        if event.patient:
            patient = event.patient.lower().strip()
            patient = patient.replace("the ", "").replace("a ", "")
            parts.append(patient)
        return "|".join(parts) if parts else event.description.lower()[:50]
    
    def _infer_temporal_ordering(self, graph: EventGraph) -> None:
        """Infer ordering relationships from year/era data."""
        events_with_time = [
            e for e in graph.events.values()
            if e.era or e.year
        ]
        
        # Sort by era then year
        def sort_key(e: Event):
            # Handle era - might be Era enum or string
            if e.era:
                if hasattr(e.era, 'order'):
                    era_order = e.era.order
                elif isinstance(e.era, str):
                    era_order = Era.from_text(e.era).order
                else:
                    era_order = -1
            else:
                era_order = -1
            # Handle year - ensure it's an int, ignore non-numeric strings
            year = 0
            if e.year:
                try:
                    year = int(e.year)
                except (ValueError, TypeError):
                    pass  # Non-numeric like "a hundred years ago last Thursday"
            return (era_order, year)
        
        sorted_events = sorted(events_with_time, key=sort_key)
        
        # Create "before" relations for events with definite ordering
        for i, e1 in enumerate(sorted_events[:-1]):
            e2 = sorted_events[i + 1]
            
            # Only add if we can be sure of ordering
            if e1.era and e2.era:
                if e1.era < e2.era:
                    # Different eras - definite ordering
                    existing = any(
                        r.event1_id == e1.id and r.event2_id == e2.id
                        for r in graph.relations
                    )
                    if not existing:
                        graph.add_relation(EventRelation(
                            event1_id=e1.id,
                            relation="before",
                            event2_id=e2.id,
                            confidence=0.95,
                        ))
                elif e1.era == e2.era and e1.year and e2.year and e1.year < e2.year:
                    # Same era, different years
                    existing = any(
                        r.event1_id == e1.id and r.event2_id == e2.id
                        for r in graph.relations
                    )
                    if not existing:
                        graph.add_relation(EventRelation(
                            event1_id=e1.id,
                            relation="before",
                            event2_id=e2.id,
                            confidence=0.9,
                        ))
    
    def extract_from_text(self, text: str, source_book: str = "") -> EventGraph:
        """Extract events from text.
        
        Args:
            text: The text to extract from
            source_book: Name of the source book
            
        Returns:
            EventGraph with events and temporal relations
        """
        graph = EventGraph()
        
        if self.use_llm:
            # Use LLM for extraction
            events, relations = self._extract_llm(text, source_book)
            for event in events:
                graph.add_event(event)
            for relation in relations:
                graph.add_relation(relation)
        else:
            # Pattern-based fallback
            events = self._extract_patterns(text, source_book)
            for event in events:
                graph.add_event(event)
        
        return graph
    
    def _extract_llm(
        self,
        text: str,
        source_book: str,
        chunk_index: int = 0,
    ) -> tuple[list[Event], list[EventRelation]]:
        """Extract events using LLM."""
        # Limit text for prompt (should already be chunked but ensure limit)
        text = text[:4000]
        
        prompt = f"""Extract key events from this fantasy text. For each event identify:
- description: Short description (e.g., "Bilbo found the Ring")
- agent: Who did it (e.g., "Bilbo")
- action: The verb/action (e.g., "found")
- patient: What was acted upon (e.g., "the Ring")
- year: Year if mentioned (e.g., 2941)
- era: Age if mentioned (first_age, second_age, third_age, fourth_age)

Also identify temporal relationships between events:
- If one event clearly happened before another
- If one event caused another

Text:
{text}

Return JSON with two arrays:
{{
  "events": [
    {{"id": "unique_id", "description": "...", "agent": "...", "action": "...", "patient": "...", "year": null, "era": null}},
    ...
  ],
  "relations": [
    {{"event1": "id1", "relation": "before", "event2": "id2"}},
    ...
  ]
}}

Focus on significant plot events, not minor actions. Include 5-15 events.

JSON:"""

        llm = LLMClient()
        response = llm.generate(prompt, temperature=0.2, max_tokens=2000)
        
        events = []
        relations = []
        
        if response:
            data = llm.extract_json(response)
            if data and isinstance(data, dict):
                for i, e in enumerate(data.get("events", [])):
                    if isinstance(e, dict) and "description" in e:
                        # Create unique ID incorporating chunk index
                        base_id = e.get("id", f"event_{i}")
                        event_id = f"c{chunk_index}_{base_id}" if chunk_index > 0 else base_id
                        era = None
                        if e.get("era"):
                            era = Era.from_text(e["era"])
                        
                        events.append(Event(
                            id=event_id,
                            description=e["description"],
                            agent=e.get("agent"),
                            action=e.get("action"),
                            patient=e.get("patient"),
                            year=e.get("year"),
                            era=era,
                            source_book=source_book,
                            confidence=0.8,
                        ))
                
                # Build ID map for relations (LLM returns original IDs, we need our prefixed ones)
                id_map = {}
                for i, e in enumerate(data.get("events", [])):
                    if isinstance(e, dict):
                        original_id = e.get("id", f"event_{i}")
                        prefixed_id = f"c{chunk_index}_{original_id}" if chunk_index > 0 else original_id
                        id_map[original_id] = prefixed_id
                
                for r in data.get("relations", []):
                    if isinstance(r, dict) and "event1" in r and "event2" in r:
                        e1_id = id_map.get(r["event1"], r["event1"])
                        e2_id = id_map.get(r["event2"], r["event2"])
                        relations.append(EventRelation(
                            event1_id=e1_id,
                            relation=r.get("relation", "before"),
                            event2_id=e2_id,
                            confidence=0.7,
                        ))
        
        return events, relations
    
    def _extract_patterns(self, text: str, source_book: str) -> list[Event]:
        """Extract events using pattern matching."""
        events = []
        
        # Simple pattern: [Name] [verb] [object]
        for verb in self.EVENT_VERBS:
            pattern = rf"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+{verb}(?:ed|s)?\s+(?:the\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)?)"
            
            for match in re.finditer(pattern, text):
                agent = match.group(1)
                patient = match.group(2)
                
                event_id = f"{agent.lower()}_{verb}_{patient.lower()}".replace(" ", "_")
                
                events.append(Event(
                    id=event_id,
                    description=f"{agent} {verb} {patient}",
                    agent=agent,
                    action=verb,
                    patient=patient,
                    source_text=match.group(0),
                    source_book=source_book,
                    confidence=0.6,
                ))
        
        return events
    
    def extract_year(self, text: str) -> tuple[Optional[int], Optional[Era]]:
        """Extract year and era from text."""
        for pattern in self._year_patterns:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                year = None
                era = None
                
                for g in groups:
                    if g and g.isdigit():
                        year = int(g)
                    elif g:
                        g_lower = g.lower()
                        if g_lower in ('first', 'f'):
                            era = Era.FIRST_AGE
                        elif g_lower in ('second', 's'):
                            era = Era.SECOND_AGE
                        elif g_lower in ('third', 't'):
                            era = Era.THIRD_AGE
                        elif g_lower in ('fourth'):
                            era = Era.FOURTH_AGE
                
                return year, era
        
        return None, None
