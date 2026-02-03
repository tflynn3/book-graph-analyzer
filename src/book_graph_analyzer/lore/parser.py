"""Parse natural language claims into structured queries.

Extracts entities, relationships, and temporal markers from text
to enable knowledge base lookups.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..llm import LLMClient


class ClaimType(Enum):
    """Types of claims that can be validated."""
    ENTITY_EXISTS = "entity_exists"           # "Gandalf is a wizard"
    RELATIONSHIP = "relationship"              # "Gandalf met Bilbo"
    ATTRIBUTE = "attribute"                    # "Gandalf is grey"
    TEMPORAL = "temporal"                      # "Turin lived in the First Age"
    LOCATION = "location"                      # "Rivendell is in the Misty Mountains"
    CAPABILITY = "capability"                  # "Hobbits can turn invisible"
    RULE = "rule"                              # "Only elves can sail West"
    EVENT_ORDER = "event_order"               # "Bilbo found the ring before Gollum"
    UNKNOWN = "unknown"


@dataclass
class ParsedClaim:
    """A structured representation of a claim."""
    original_text: str
    claim_type: ClaimType
    
    # Core entities
    subject: Optional[str] = None
    subject_type: Optional[str] = None  # character, place, object
    
    # For relationship claims
    predicate: Optional[str] = None
    object: Optional[str] = None
    object_type: Optional[str] = None
    
    # For attribute claims
    attribute: Optional[str] = None
    value: Optional[str] = None
    
    # Temporal markers
    time_period: Optional[str] = None  # "First Age", "Second Age", etc.
    relative_time: Optional[str] = None  # "before", "after", "during"
    
    # For event ordering claims
    event1_agent: Optional[str] = None  # Who did event 1
    event1_action: Optional[str] = None  # What they did
    event1_patient: Optional[str] = None  # What it was done to
    event2_agent: Optional[str] = None  # Who did event 2
    event2_action: Optional[str] = None  # What they did
    event2_patient: Optional[str] = None  # What it was done to
    ordering: Optional[str] = None  # "before", "after"
    
    # Modifiers
    negated: bool = False  # "Gandalf did NOT go to Mordor"
    uncertain: bool = False  # "might have", "possibly"
    
    # Confidence in the parse
    confidence: float = 1.0


class ClaimParser:
    """Parses natural language claims into structured form.
    
    Uses a combination of pattern matching and LLM for complex cases.
    
    Usage:
        parser = ClaimParser()
        claim = parser.parse("Turin lived in the First Age")
        # claim.subject = "Turin"
        # claim.claim_type = ClaimType.TEMPORAL
        # claim.time_period = "First Age"
    """
    
    # Temporal markers
    TIME_PERIODS = [
        "First Age", "Second Age", "Third Age", "Fourth Age",
        "Years of the Trees", "Years of the Sun",
        "Elder Days", "Younger Days",
    ]
    
    # Relationship verbs
    RELATIONSHIP_VERBS = [
        "met", "fought", "killed", "married", "loved", "hated",
        "traveled to", "went to", "visited", "lived in", "ruled",
        "created", "forged", "destroyed", "found", "lost",
        "spoke to", "spoke with", "told", "said to",
        "gave", "received", "took", "stole",
        "is father of", "is mother of", "is son of", "is daughter of",
        "is brother of", "is sister of", "is friend of", "is enemy of",
    ]
    
    # Attribute patterns
    ATTRIBUTE_PATTERNS = [
        (r"(\w+) is (?:a |an |the )?(\w+)", "type"),  # "Gandalf is a wizard"
        (r"(\w+) (?:is|was|are|were) (\w+)", "attribute"),  # "Gandalf is grey"
        (r"(\w+) has (\w+)", "possession"),  # "Bilbo has a ring"
    ]
    
    def __init__(self, use_llm: bool = True):
        """Initialize the parser.
        
        Args:
            use_llm: Use LLM for complex parsing
        """
        self.use_llm = use_llm
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns."""
        # Time period pattern
        time_pattern = "|".join(re.escape(t) for t in self.TIME_PERIODS)
        self._time_pattern = re.compile(
            rf"(?:in|during|of)\s+(?:the\s+)?({time_pattern})",
            re.IGNORECASE
        )
        
        # Negation pattern
        self._negation_pattern = re.compile(
            r"\b(not|never|didn't|did not|wasn't|was not|isn't|is not|aren't|are not)\b",
            re.IGNORECASE
        )
        
        # Uncertainty pattern
        self._uncertainty_pattern = re.compile(
            r"\b(might|may|possibly|perhaps|probably|could have|might have)\b",
            re.IGNORECASE
        )
    
    def parse(self, text: str) -> ParsedClaim:
        """Parse a claim from natural language.
        
        Args:
            text: The claim text
            
        Returns:
            Structured ParsedClaim
        """
        claim = ParsedClaim(original_text=text, claim_type=ClaimType.UNKNOWN)
        
        # Check for negation
        if self._negation_pattern.search(text):
            claim.negated = True
        
        # Check for uncertainty
        if self._uncertainty_pattern.search(text):
            claim.uncertain = True
        
        # Check for temporal markers
        time_match = self._time_pattern.search(text)
        if time_match:
            claim.time_period = time_match.group(1)
            claim.claim_type = ClaimType.TEMPORAL
        
        # Try pattern-based parsing first
        parsed = self._parse_patterns(text, claim)
        if parsed.claim_type != ClaimType.UNKNOWN:
            return parsed
        
        # Fall back to LLM parsing for complex cases
        if self.use_llm:
            return self._parse_llm(text, claim)
        
        return claim
    
    def _parse_patterns(self, text: str, claim: ParsedClaim) -> ParsedClaim:
        """Parse using regex patterns."""
        text_lower = text.lower()
        
        # Check for event ordering patterns
        # Use LLM for complex event parsing
        if self.use_llm and ("before" in text_lower or "after" in text_lower):
            event_claim = self._parse_event_order_llm(text, claim)
            if event_claim.claim_type == ClaimType.EVENT_ORDER:
                return event_claim
        
        # Simple patterns as fallback
        # Pattern: "X found Y before Z" (same action implied for Z)
        event_verbs = r"(?:found|lost|took|gave|stole|killed|died|created|forged|destroyed|met|married|arrived|left)"
        
        simple_pattern = re.compile(
            rf"([A-Z]\w+)\s+({event_verbs})\s+(?:the\s+)?(\w+(?:\s+\w+)?)\s+(before|after)\s+([A-Z]\w+)",
            re.IGNORECASE
        )
        match = simple_pattern.search(text)
        if match:
            claim.claim_type = ClaimType.EVENT_ORDER
            claim.event1_agent = match.group(1)
            claim.event1_action = match.group(2)
            claim.event1_patient = match.group(3)
            claim.ordering = match.group(4).lower()
            claim.event2_agent = match.group(5)
            claim.event2_action = claim.event1_action
            claim.event2_patient = claim.event1_patient
            return claim
        
        # Check for relationship verbs
        for verb in self.RELATIONSHIP_VERBS:
            pattern = rf"(\w+(?:\s+\w+)?)\s+{re.escape(verb)}\s+(\w+(?:\s+\w+)?)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                claim.subject = match.group(1).strip()
                claim.predicate = verb
                claim.object = match.group(2).strip()
                claim.claim_type = ClaimType.RELATIONSHIP
                return claim
        
        # Check for "is a" patterns (entity type)
        match = re.search(r"(\w+(?:\s+\w+)?)\s+is\s+(?:a|an)\s+(\w+)", text, re.IGNORECASE)
        if match:
            claim.subject = match.group(1).strip()
            claim.attribute = "type"
            claim.value = match.group(2).strip()
            claim.claim_type = ClaimType.ENTITY_EXISTS
            return claim
        
        # Check for "is/was [adjective]" patterns
        match = re.search(r"(\w+(?:\s+\w+)?)\s+(?:is|was)\s+(\w+)", text, re.IGNORECASE)
        if match:
            claim.subject = match.group(1).strip()
            claim.attribute = "description"
            claim.value = match.group(2).strip()
            claim.claim_type = ClaimType.ATTRIBUTE
            return claim
        
        # Check for location patterns
        match = re.search(r"(\w+(?:\s+\w+)?)\s+is\s+(?:in|at|near)\s+(?:the\s+)?(\w+(?:\s+\w+)?)", text, re.IGNORECASE)
        if match:
            claim.subject = match.group(1).strip()
            claim.predicate = "located_in"
            claim.object = match.group(2).strip()
            claim.claim_type = ClaimType.LOCATION
            return claim
        
        # Check for capability patterns
        match = re.search(r"(\w+(?:\s+\w+)?)\s+can\s+(\w+(?:\s+\w+)?)", text, re.IGNORECASE)
        if match:
            claim.subject = match.group(1).strip()
            claim.attribute = "capability"
            claim.value = match.group(2).strip()
            claim.claim_type = ClaimType.CAPABILITY
            return claim
        
        return claim
    
    def _parse_event_order_llm(self, text: str, claim: ParsedClaim) -> ParsedClaim:
        """Parse event ordering claims using LLM."""
        prompt = f"""Parse this claim about two events and their ordering.

Claim: "{text}"

IMPORTANT: Each event has its own agent, action, and patient. Do NOT copy values from Event 1 to Event 2.

Examples:
- "Gollum lost the ring before Bilbo found it"
  Event1: agent=Gollum, action=lost, patient=the ring
  Event2: agent=Bilbo, action=found, patient=it/the ring
  ordering=before (Event1 happened before Event2)

- "Bard killed Smaug after Smaug attacked Lake-town"  
  Event1: agent=Bard, action=killed, patient=Smaug
  Event2: agent=Smaug, action=attacked, patient=Lake-town
  ordering=after (Event1 happened after Event2)

Extract from the claim:
- event1_agent: Who did Event 1
- event1_action: The EXACT verb for Event 1
- event1_patient: What Event 1 acted upon
- ordering: "before" or "after"
- event2_agent: Who did Event 2
- event2_action: The EXACT verb for Event 2 (may differ from event1_action!)
- event2_patient: What Event 2 acted upon

Return JSON only:

JSON:"""

        llm = LLMClient()
        response = llm.generate(prompt, temperature=0.1, max_tokens=300)
        
        if response:
            data = llm.extract_json(response)
            if data and isinstance(data, dict):
                if data.get("event1_agent") and data.get("ordering"):
                    claim.claim_type = ClaimType.EVENT_ORDER
                    claim.event1_agent = data.get("event1_agent")
                    claim.event1_action = data.get("event1_action")
                    claim.event1_patient = data.get("event1_patient")
                    claim.ordering = data.get("ordering", "").lower()
                    claim.event2_agent = data.get("event2_agent")
                    claim.event2_action = data.get("event2_action")
                    claim.event2_patient = data.get("event2_patient")
                    claim.confidence = 0.8
        
        return claim
    
    def _parse_llm(self, text: str, claim: ParsedClaim) -> ParsedClaim:
        """Parse using LLM for complex cases."""
        prompt = f"""Parse this claim about a fictional world into structured form.

Claim: "{text}"

Extract:
- subject: The main entity being described
- subject_type: character, place, object, race, or concept
- claim_type: one of [entity_exists, relationship, attribute, temporal, location, capability, rule]
- predicate: The relationship or action (if applicable)
- object: The secondary entity (if applicable)  
- attribute: The property being claimed (if applicable)
- value: The value of the attribute (if applicable)
- time_period: Any time reference like "First Age", "Second Age" (if mentioned)
- negated: true if the claim is negative ("did not", "never", etc.)

Return as JSON object. Only include fields that apply.

JSON:"""

        llm = LLMClient()
        response = llm.generate(prompt, temperature=0.1, max_tokens=500)
        
        if response:
            data = llm.extract_json(response)
            if data and isinstance(data, dict):
                if "subject" in data:
                    claim.subject = data["subject"]
                if "subject_type" in data:
                    claim.subject_type = data["subject_type"]
                if "claim_type" in data:
                    try:
                        claim.claim_type = ClaimType(data["claim_type"])
                    except ValueError:
                        pass
                if "predicate" in data:
                    claim.predicate = data["predicate"]
                if "object" in data:
                    claim.object = data["object"]
                if "attribute" in data:
                    claim.attribute = data["attribute"]
                if "value" in data:
                    claim.value = data["value"]
                if "time_period" in data:
                    claim.time_period = data["time_period"]
                if "negated" in data:
                    claim.negated = bool(data["negated"])
                
                claim.confidence = 0.8
        
        return claim
    
    def parse_multiple(self, text: str) -> list[ParsedClaim]:
        """Parse multiple claims from a block of text.
        
        Splits on sentence boundaries and parses each.
        
        Args:
            text: Text potentially containing multiple claims
            
        Returns:
            List of ParsedClaims
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Skip very short fragments
                claims.append(self.parse(sentence))
        
        return claims
