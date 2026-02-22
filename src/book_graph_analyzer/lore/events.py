"""Event extraction for temporal reasoning.

Extracts structured events from text:
- Who did what (agent, action, patient)
- When it happened (era, year, relative time)
- Temporal ordering (before/after other events)
"""

import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

from ..llm import LLMClient
from ..extract.resilient_chunk_runner import (
    ResilientChunkRunner,
    ChunkAttemptResult,
    ChunkStatus,
)
from .temporal import Era


logger = logging.getLogger(__name__)


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
        def _s(v):
            if v is None: return None
            if isinstance(v, list): return " ".join(str(x) for x in v if x) or None
            return str(v)
        # Handle era being either an Enum or a plain string
        if self.era is None:
            era_val = None
        elif isinstance(self.era, str):
            era_val = self.era
        else:
            era_val = self.era.value
        return {
            "id": self.id,
            "description": self.description,
            "agent": _s(self.agent),
            "action": _s(self.action),
            "patient": _s(self.patient),
            "era": era_val,
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
    
    def to_dict(self) -> dict:
        return {
            "event1_id": self.event1_id,
            "relation": self.relation,
            "event2_id": self.event2_id,
            "confidence": self.confidence,
            "source_text": self.source_text,
        }


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
                # Same era, check years (handle str/int mismatch)
                elif e1.year and e2.year:
                    try:
                        y1, y2 = int(e1.year), int(e2.year)
                        if y1 < y2:
                            return "before"
                        elif y1 > y2:
                            return "after"
                        else:
                            return "same"
                    except (ValueError, TypeError):
                        pass  # Can't compare non-numeric years
        
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
        checkpoint_file: Optional[str] = None,
        resilient: bool = False,
        fallback_model: Optional[str] = None,
    ) -> EventGraph:
        """Extract events from a full book using chunked processing.
        
        Args:
            text: Full book text
            source_book: Name of the source book
            chunk_size: Characters per chunk
            overlap: Overlap between chunks to avoid losing context at boundaries
            checkpoint_file: Optional path to save/resume progress
            
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
        start_chunk = 0
        
        # Load checkpoint if exists
        if checkpoint_file:
            checkpoint = self._load_checkpoint(checkpoint_file)
            if checkpoint:
                start_chunk = checkpoint.get("next_chunk", 0)
                all_events = [Event(**e) for e in checkpoint.get("events", [])]
                all_relations = [EventRelation(**r) for r in checkpoint.get("relations", [])]
                self._seen_events = set(checkpoint.get("seen_keys", []))
                print(f"  Resuming from checkpoint: chunk {start_chunk}/{total_chunks} ({len(all_events)} events)", flush=True)
        
        if resilient and checkpoint_file and self.use_llm:
            self._run_resilient_chunks(
                chunks=chunks,
                source_book=source_book,
                checkpoint_file=checkpoint_file,
                all_events=all_events,
                all_relations=all_relations,
                fallback_model=fallback_model,
                total_chunks=total_chunks,
            )
        else:
            for i, chunk in enumerate(chunks):
            # Skip already processed chunks
                if i < start_chunk:
                    continue
                
                if self.progress_callback:
                    self.progress_callback(i + 1, total_chunks, f"Processing chunk {i + 1}/{total_chunks}")
            
            # Simple progress print every 10 chunks (visible even without Rich)
                if (i + 1) % 10 == 0 or i == start_chunk:
                    print(f"  [chunk {i + 1}/{total_chunks}]", flush=True)
            
                events, relations = self._extract_llm(chunk, source_book, chunk_index=i)
            
                for event in events:
                # Deduplicate based on normalized description
                    event_key = self._normalize_event_key(event)
                    if event_key not in self._seen_events:
                        self._seen_events.add(event_key)
                        all_events.append(event)
            
                all_relations.extend(relations)
            
            # Save checkpoint after each chunk
                if checkpoint_file:
                    self._save_checkpoint(checkpoint_file, i + 1, total_chunks, all_events, all_relations)
        
        # Clear checkpoint on completion
        if checkpoint_file:
            self._clear_checkpoint(checkpoint_file)
        
        # Add all events to graph
        for event in all_events:
            graph.add_event(event)
        
        # Add relations, filtering invalid ones
        for rel in all_relations:
            if rel.event1_id in graph.events and rel.event2_id in graph.events:
                graph.add_relation(rel)
        
        # Infer additional ordering from year/era
        self._infer_temporal_ordering(graph)

        if resilient and checkpoint_file and self.use_llm:
            summary = self.get_resilient_summary(checkpoint_file)
            print(
                "  Resilient summary: "
                f"ok={summary.get('ok', 0)} retried={summary.get('retried', 0)} "
                f"fallback_success={summary.get('fallback_success', 0)} failed={summary.get('failed', 0)}",
                flush=True,
            )
        
        return graph

    def get_resilient_summary(self, checkpoint_file: str) -> dict:
        ledger_path = Path(checkpoint_file).with_suffix(Path(checkpoint_file).suffix + ".ledger.json")
        try:
            data = json.loads(ledger_path.read_text(encoding="utf-8"))
            return data.get("metrics", {})
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _run_resilient_chunks(
        self,
        *,
        chunks: list[str],
        source_book: str,
        checkpoint_file: str,
        all_events: list[Event],
        all_relations: list[EventRelation],
        fallback_model: Optional[str],
        total_chunks: int,
    ) -> None:
        base = Path(checkpoint_file)
        ledger_file = base.with_suffix(base.suffix + ".ledger.json")
        payload_dir = base.with_suffix(base.suffix + ".payloads")
        payload_dir.mkdir(parents=True, exist_ok=True)

        default_llm = LLMClient()
        primary_model = getattr(default_llm, "model", "")
        if not fallback_model:
            fallback_model = "gpt-4o" if default_llm.provider == "openai" else "llama3.1:70b"

        runner = ResilientChunkRunner(ledger_file)

        def persist_artifact() -> None:
            completed = len(runner.state.ledger)
            self._save_checkpoint(checkpoint_file, completed, total_chunks, all_events, all_relations)

        def process_attempt(chunk_index: int, chunk: str, model: str, attempt_no: int) -> ChunkAttemptResult:
            if self.progress_callback:
                self.progress_callback(
                    min(chunk_index + 1, total_chunks),
                    total_chunks,
                    f"Processing chunk {chunk_index + 1}/{total_chunks} (attempt {attempt_no}, model {model})",
                )

            events, relations, reason, raw = self._extract_llm_once(chunk, source_book, chunk_index=chunk_index, model=model)
            snippet_path = None
            if reason and raw:
                snippet_path = str(payload_dir / f"chunk_{chunk_index:04d}_attempt_{attempt_no}.txt")
                Path(snippet_path).write_text(raw[:3000], encoding="utf-8")

            if reason:
                return ChunkAttemptResult(False, reason=reason, model=model, payload_snippet_path=snippet_path)

            for event in events:
                event_key = self._normalize_event_key(event)
                if event_key not in self._seen_events:
                    self._seen_events.add(event_key)
                    all_events.append(event)
            all_relations.extend(relations)
            return ChunkAttemptResult(True, model=model)

        runner.run(
            chunks=chunks,
            primary_model=primary_model,
            fallback_model=fallback_model,
            process_attempt=process_attempt,
            persist_artifact=persist_artifact,
        )
    
    @staticmethod
    def _coerce_str(val) -> str:
        """Coerce a field that may be str, list, or None to a plain string."""
        if val is None:
            return ""
        if isinstance(val, list):
            return " ".join(str(v) for v in val if v)
        return str(val)

    @staticmethod
    def _stable_id_token(val) -> str:
        """Normalize potentially-structured IDs to stable string tokens.

        LLM payloads occasionally return relation refs as list/object instead of plain
        strings. This normalizes those values so they can be safely used for dict
        lookups and relation IDs.
        """
        if val is None:
            return ""
        if isinstance(val, str):
            return val.strip()
        if isinstance(val, (int, float, bool)):
            return str(val)
        if isinstance(val, (list, dict)):
            try:
                return json.dumps(val, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            except (TypeError, ValueError):
                return str(val)
        return str(val)

    def _normalize_event_key(self, event: Event) -> str:
        """Create a normalized key for deduplication."""
        parts = []
        agent = self._coerce_str(event.agent).lower().strip()
        if agent:
            parts.append(agent)
        if event.action:
            action = self._coerce_str(event.action).lower().strip()
            action = action.rstrip('ed').rstrip('s')
            parts.append(action)
        if event.patient:
            patient = self._coerce_str(event.patient).lower().strip()
            patient = patient.replace("the ", "").replace("a ", "")
            parts.append(patient)
        return "|".join(parts) if parts else event.description.lower()[:50]
    
    def _load_checkpoint(self, checkpoint_file: str) -> Optional[dict]:
        """Load checkpoint from file if it exists."""
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    def _save_checkpoint(
        self,
        checkpoint_file: str,
        next_chunk: int,
        total_chunks: int,
        events: list[Event],
        relations: list[EventRelation],
    ) -> None:
        """Save current progress to checkpoint file."""
        checkpoint = {
            "next_chunk": next_chunk,
            "total_chunks": total_chunks,
            "events": [e.to_dict() for e in events],
            "relations": [r.to_dict() for r in relations],
            "seen_keys": list(self._seen_events),
        }
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f)
    
    def _clear_checkpoint(self, checkpoint_file: str) -> None:
        """Remove checkpoint file on successful completion."""
        try:
            import os
            os.remove(checkpoint_file)
            print(f"  Checkpoint cleared: {checkpoint_file}", flush=True)
        except OSError:
            pass
    
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
                elif e1.era == e2.era and e1.year and e2.year:
                    # Safely compare years (might be str or int)
                    try:
                        y1, y2 = int(e1.year), int(e2.year)
                    except (ValueError, TypeError):
                        continue
                    # Same era, different years
                    if y1 < y2:
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

        events, relations, _reason, _raw = self._extract_llm_once(
            text,
            source_book,
            chunk_index=chunk_index,
            model=None,
        )
        return events, relations

    def _extract_llm_once(
        self,
        text: str,
        source_book: str,
        chunk_index: int = 0,
        model: Optional[str] = None,
    ) -> tuple[list[Event], list[EventRelation], str, str]:
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

        try:
            llm = LLMClient(model=model)
        except TypeError:
            llm = LLMClient()
        logger.info(
            "Event extraction LLM provider=%s model=%s chunk=%d",
            getattr(llm, "provider", "unknown"),
            getattr(llm, "model", "unknown"),
            chunk_index,
        )
        response = llm.generate(prompt, temperature=0.2, max_tokens=2000)
        
        events = []
        relations = []
        
        if response:
            data = llm.extract_json(response)
            if data and isinstance(data, dict):
                dropped_events = 0
                dropped_relations = 0

                for i, e in enumerate(data.get("events", [])):
                    if isinstance(e, dict) and "description" in e:
                        # Create unique ID incorporating chunk index
                        base_id = e.get("id", f"event_{i}")
                        base_id = self._stable_id_token(base_id) or f"event_{i}"
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
                    else:
                        dropped_events += 1
                
                # Build ID map for relations (LLM returns original IDs, we need our prefixed ones)
                id_map = {}
                for i, e in enumerate(data.get("events", [])):
                    if isinstance(e, dict):
                        original_id = e.get("id", f"event_{i}")
                        original_id = self._stable_id_token(original_id) or f"event_{i}"
                        prefixed_id = f"c{chunk_index}_{original_id}" if chunk_index > 0 else original_id
                        id_map[original_id] = prefixed_id
                
                for r in data.get("relations", []):
                    if isinstance(r, dict) and "event1" in r and "event2" in r:
                        e1_raw = self._stable_id_token(r.get("event1"))
                        e2_raw = self._stable_id_token(r.get("event2"))
                        if not e1_raw or not e2_raw:
                            dropped_relations += 1
                            continue
                        e1_id = id_map.get(e1_raw, e1_raw)
                        e2_id = id_map.get(e2_raw, e2_raw)
                        relations.append(EventRelation(
                            event1_id=e1_id,
                            relation=r.get("relation", "before"),
                            event2_id=e2_id,
                            confidence=0.7,
                        ))
                    else:
                        dropped_relations += 1

                if dropped_events or dropped_relations:
                    logger.warning(
                        "Dropped malformed LLM payload rows in event extraction: "
                        "events=%d relations=%d (chunk=%d)",
                        dropped_events,
                        dropped_relations,
                        chunk_index,
                    )
            else:
                logger.warning(
                    "LLM response did not contain valid JSON payload; skipping chunk=%d",
                    chunk_index,
                )
                return [], [], "malformed_json", response
        else:
            logger.warning(
                "LLM request returned empty response; skipping chunk=%d",
                chunk_index,
            )
            return [], [], "empty_response", ""

        return events, relations, "", response
    
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
