"""Lore consistency checker.

Validates claims against extracted world knowledge from:
- World Bible (rules, cultures, magic systems)
- Entity database (characters, places, objects)
- Relationship graph (who knows whom, who went where)
- Timeline (temporal reasoning)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import json

from rapidfuzz import fuzz

from ..llm import LLMClient
from ..worldbible import WorldBible, WorldBibleCategory
from ..corpus import CrossBookResolver
from ..graph.connection import get_driver
from .parser import ClaimParser, ParsedClaim, ClaimType
from .temporal import Timeline, TemporalExtractor, Era
from .events import EventGraph, EventExtractor


def _compute_confidence(
    claim_text: str,
    found_text: str,
    evidence_type: str = "direct",
) -> float:
    """Compute confidence based on match quality and evidence type.
    
    Args:
        claim_text: What the user claimed (e.g., "Bilbo found the ring")
        found_text: What we found in the graph (e.g., "Bilbo found the Ring")
        evidence_type: "direct" (explicit relation), "year" (inferred from years), "era" (inferred from era)
    
    Returns:
        Confidence 0.0-1.0
    """
    # Base confidence by evidence type
    base = {
        "direct": 0.95,  # Explicit BEFORE/AFTER relation
        "year": 0.88,    # Same era, inferred from year comparison
        "era": 0.80,     # Different eras
    }.get(evidence_type, 0.85)
    
    # Match quality: how well did the claim match what we found?
    match_score = fuzz.token_set_ratio(claim_text.lower(), found_text.lower()) / 100.0
    
    # Combine: base * match_quality, with floor of 0.5 if we found anything
    confidence = base * match_score
    confidence = max(0.5, min(0.95, confidence))  # Clamp to [0.5, 0.95]
    
    return round(confidence, 2)


class ValidationStatus(Enum):
    """Result of validating a claim."""
    VALID = "valid"           # Claim is supported by evidence
    INVALID = "invalid"       # Claim contradicts evidence
    UNKNOWN = "unknown"       # No evidence found either way
    PARTIAL = "partial"       # Some aspects valid, some unknown
    PLAUSIBLE = "plausible"   # Not directly supported but consistent


@dataclass
class Evidence:
    """Evidence supporting or refuting a claim."""
    text: str
    source: str  # e.g., "World Bible: Magic Rules" or "The Hobbit, Ch. 3"
    supports: bool  # True if supports claim, False if refutes
    relevance: float = 1.0


@dataclass
class ValidationResult:
    """Result of validating a claim."""
    claim: ParsedClaim
    status: ValidationStatus
    confidence: float = 0.0
    
    # Evidence
    supporting: list[Evidence] = field(default_factory=list)
    contradicting: list[Evidence] = field(default_factory=list)
    
    # Explanation
    explanation: str = ""
    
    # Suggestions for fixing invalid claims
    suggestions: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "claim": self.claim.original_text,
            "status": self.status.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "supporting_evidence": [
                {"text": e.text, "source": e.source} for e in self.supporting
            ],
            "contradicting_evidence": [
                {"text": e.text, "source": e.source} for e in self.contradicting
            ],
            "suggestions": self.suggestions,
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        icon = {
            ValidationStatus.VALID: "[OK]",
            ValidationStatus.INVALID: "[X]",
            ValidationStatus.UNKNOWN: "[?]",
            ValidationStatus.PARTIAL: "[~]",
            ValidationStatus.PLAUSIBLE: "[~]",
        }[self.status]
        
        lines = [
            f"{icon} {self.claim.original_text}",
            f"    Status: {self.status.value} (confidence: {self.confidence:.0%})",
        ]
        
        if self.explanation:
            lines.append(f"    Reason: {self.explanation}")
        
        if self.supporting:
            lines.append(f"    Evidence FOR ({len(self.supporting)}):")
            for e in self.supporting[:2]:
                lines.append(f"      - {e.text[:80]}... [{e.source}]")
        
        if self.contradicting:
            lines.append(f"    Evidence AGAINST ({len(self.contradicting)}):")
            for e in self.contradicting[:2]:
                lines.append(f"      - {e.text[:80]}... [{e.source}]")
        
        if self.suggestions:
            lines.append(f"    Suggestions:")
            for s in self.suggestions:
                lines.append(f"      - {s}")
        
        return "\n".join(lines)


class LoreChecker:
    """Validates claims against extracted world knowledge.
    
    Usage:
        checker = LoreChecker()
        checker.load_world_bible("middle_earth_bible.json")
        
        result = checker.check("Turin lived in the Second Age")
        print(result.status)  # ValidationStatus.INVALID
        print(result.explanation)  # "Turin died in the First Age"
        
        # Check multiple claims
        results = checker.check_text('''
            Gandalf the Brown arrived at the Shire.
            Bilbo found the Ring in Gollum's cave.
        ''')
    """
    
    def __init__(self, use_llm: bool = True):
        """Initialize the checker.
        
        Args:
            use_llm: Use LLM for semantic matching and complex validation
        """
        self.use_llm = use_llm
        self.parser = ClaimParser(use_llm=use_llm)
        self.temporal_extractor = TemporalExtractor(use_llm=use_llm)
        
        # Knowledge sources
        self.world_bible: Optional[WorldBible] = None
        self.entity_resolver: Optional[CrossBookResolver] = None
        self.timeline: Optional[Timeline] = None
        self.event_graph: Optional[EventGraph] = None
        self._neo4j_driver = None
        
        # Cached knowledge for quick lookup
        self._entity_cache: dict[str, dict] = {}
        self._rule_cache: list[dict] = []
    
    def load_world_bible(self, path: str | Path) -> None:
        """Load a world bible for validation."""
        path = Path(path)
        self.world_bible = WorldBible.load(path)
        self._build_rule_cache()
    
    def load_corpus_entities(self, corpus_name: str) -> None:
        """Load entity resolver for a corpus."""
        self.entity_resolver = CrossBookResolver(corpus_name)
        self._build_entity_cache()
    
    def load_timeline(self, path: str | Path) -> None:
        """Load a pre-extracted timeline."""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            self.timeline = Timeline.from_dict(json.load(f))
    
    def extract_timeline_from_text(self, text: str) -> None:
        """Extract timeline from text on the fly."""
        self.timeline = self.temporal_extractor.extract_from_text(text)
    
    def connect_neo4j(self) -> bool:
        """Connect to Neo4j for relationship queries."""
        self._neo4j_driver = get_driver()
        return self._neo4j_driver is not None
    
    def load_events(self, path: str | Path) -> None:
        """Load pre-extracted event graph."""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            self.event_graph = EventGraph.from_dict(json.load(f))
    
    def extract_events_from_text(self, text: str, source: str = "") -> None:
        """Extract events from text on the fly."""
        extractor = EventExtractor(use_llm=self.use_llm)
        self.event_graph = extractor.extract_from_text(text, source)
    
    def _build_rule_cache(self) -> None:
        """Build searchable cache from world bible rules."""
        if not self.world_bible:
            return
        
        self._rule_cache = []
        for category, rules in self.world_bible.rules.items():
            for rule in rules:
                self._rule_cache.append({
                    "title": rule.title,
                    "description": rule.description,
                    "category": category.value,
                    "keywords": rule.keywords,
                    "source": f"World Bible: {category.value}",
                })
    
    def _build_entity_cache(self) -> None:
        """Build searchable cache from entities."""
        if not self.entity_resolver:
            return
        
        for entity_id, entity in self.entity_resolver.entities.items():
            self._entity_cache[entity.canonical_name.lower()] = {
                "id": entity_id,
                "name": entity.canonical_name,
                "type": entity.entity_type,
                "aliases": list(entity.all_names),
                "books": list(entity.book_clusters.keys()),
                "mentions": entity.total_mentions,
            }
            # Also index by aliases
            for alias in entity.all_names:
                self._entity_cache[alias.lower()] = self._entity_cache[entity.canonical_name.lower()]
    
    def check(self, claim_text: str) -> ValidationResult:
        """Check a single claim.
        
        Args:
            claim_text: The claim to validate
            
        Returns:
            ValidationResult with status and evidence
        """
        # Parse the claim
        claim = self.parser.parse(claim_text)
        
        # Initialize result
        result = ValidationResult(
            claim=claim,
            status=ValidationStatus.UNKNOWN,
            confidence=0.0,
        )
        
        # Route to appropriate checker based on claim type
        if claim.claim_type == ClaimType.ENTITY_EXISTS:
            result = self._check_entity_exists(claim, result)
        elif claim.claim_type == ClaimType.ATTRIBUTE:
            result = self._check_attribute(claim, result)
        elif claim.claim_type == ClaimType.RELATIONSHIP:
            result = self._check_relationship(claim, result)
        elif claim.claim_type == ClaimType.TEMPORAL:
            result = self._check_temporal(claim, result)
        elif claim.claim_type == ClaimType.CAPABILITY:
            result = self._check_capability(claim, result)
        elif claim.claim_type == ClaimType.LOCATION:
            result = self._check_location(claim, result)
        elif claim.claim_type == ClaimType.EVENT_ORDER:
            result = self._check_event_order(claim, result)
        else:
            # Try general rule matching
            result = self._check_against_rules(claim, result)
        
        # No LLM fallback for answers - only extracted data
        # LLM is used for parsing claims, not for generating answers
        
        return result
    
    def check_text(self, text: str) -> list[ValidationResult]:
        """Check multiple claims in a text block.
        
        Args:
            text: Text containing multiple claims/statements
            
        Returns:
            List of ValidationResults
        """
        claims = self.parser.parse_multiple(text)
        return [self.check(c.original_text) for c in claims]
    
    def _check_entity_exists(self, claim: ParsedClaim, result: ValidationResult) -> ValidationResult:
        """Check if an entity exists with claimed properties."""
        if not claim.subject:
            return result
        
        subject_lower = claim.subject.lower()
        
        # Check entity cache
        if subject_lower in self._entity_cache:
            entity = self._entity_cache[subject_lower]
            
            # Entity exists
            if claim.value:
                # Check if the claimed type/attribute matches
                if claim.attribute == "type":
                    if claim.value.lower() in entity.get("type", "").lower():
                        result.status = ValidationStatus.VALID
                        result.confidence = 0.9
                        result.explanation = f"{claim.subject} is indeed a {entity['type']}"
                        result.supporting.append(Evidence(
                            text=f"{entity['name']} is a {entity['type']}",
                            source=f"Entity Database ({entity['mentions']} mentions)",
                            supports=True,
                        ))
                    else:
                        result.status = ValidationStatus.UNKNOWN
                        result.explanation = f"{claim.subject} exists but type '{claim.value}' not confirmed"
            else:
                result.status = ValidationStatus.VALID
                result.confidence = 0.95
                result.explanation = f"{claim.subject} exists in the corpus"
                result.supporting.append(Evidence(
                    text=f"{entity['name']} appears in {entity['books']}",
                    source="Entity Database",
                    supports=True,
                ))
        else:
            result.status = ValidationStatus.UNKNOWN
            result.explanation = f"Entity '{claim.subject}' not found in knowledge base"
        
        return result
    
    def _check_attribute(self, claim: ParsedClaim, result: ValidationResult) -> ValidationResult:
        """Check attribute claims against world bible."""
        if not claim.subject or not claim.value:
            return result
        
        # Search rules for relevant information
        relevant_rules = self._find_relevant_rules(claim.subject, claim.value)
        
        for rule in relevant_rules:
            # Check if rule supports or contradicts
            if self._rule_supports_claim(rule, claim):
                result.supporting.append(Evidence(
                    text=rule["description"],
                    source=rule["source"],
                    supports=True,
                ))
            elif self._rule_contradicts_claim(rule, claim):
                result.contradicting.append(Evidence(
                    text=rule["description"],
                    source=rule["source"],
                    supports=False,
                ))
        
        # Determine status based on evidence
        if result.contradicting and not result.supporting:
            result.status = ValidationStatus.INVALID
            result.confidence = 0.8
            result.explanation = f"Contradicted by: {result.contradicting[0].text[:100]}"
        elif result.supporting and not result.contradicting:
            result.status = ValidationStatus.VALID
            result.confidence = 0.8
            result.explanation = f"Supported by: {result.supporting[0].text[:100]}"
        elif result.supporting and result.contradicting:
            result.status = ValidationStatus.PARTIAL
            result.confidence = 0.5
            result.explanation = "Mixed evidence found"
        
        return result
    
    def _check_relationship(self, claim: ParsedClaim, result: ValidationResult) -> ValidationResult:
        """Check relationship claims against Neo4j graph."""
        if not claim.subject or not claim.predicate:
            return self._check_against_rules(claim, result)
        
        # Query Neo4j if available
        if self._neo4j_driver:
            neo4j_result = self._query_relationship_neo4j(
                claim.subject,
                claim.predicate,
                claim.object,
            )
            
            if neo4j_result is not None:
                if neo4j_result["found"]:
                    result.status = ValidationStatus.VALID
                    result.confidence = 0.95
                    result.explanation = f"Relationship found in knowledge graph"
                    result.supporting.append(Evidence(
                        text=f"{claim.subject} -> {claim.predicate} -> {claim.object}",
                        source=f"Neo4j ({neo4j_result.get('count', 1)} matches)",
                        supports=True,
                    ))
                    return result
                elif neo4j_result.get("entities_exist"):
                    # Entities exist but relationship not found
                    result.status = ValidationStatus.UNKNOWN
                    result.explanation = f"Entities exist but relationship not confirmed"
        
        # Fall back to world bible
        return self._check_against_rules(claim, result)
    
    def _query_relationship_neo4j(
        self,
        subject: str,
        predicate: str,
        obj: Optional[str],
    ) -> Optional[dict]:
        """Query Neo4j for a relationship."""
        if not self._neo4j_driver:
            return None
        
        try:
            with self._neo4j_driver.session() as session:
                # Map common predicates to relationship types
                rel_type_map = {
                    "met": "INTERACTED_WITH",
                    "spoke to": "SPOKE_TO",
                    "spoke with": "SPOKE_TO",
                    "traveled to": "TRAVELED_TO",
                    "went to": "TRAVELED_TO",
                    "visited": "TRAVELED_TO",
                    "lived in": "LIVED_IN",
                    "fought": "FOUGHT",
                    "killed": "KILLED",
                    "created": "CREATED",
                    "forged": "CREATED",
                    "found": "FOUND",
                    "gave": "GAVE",
                    "is father of": "PARENT_OF",
                    "is mother of": "PARENT_OF",
                    "is son of": "CHILD_OF",
                    "is daughter of": "CHILD_OF",
                }
                
                rel_type = rel_type_map.get(predicate.lower(), predicate.upper().replace(" ", "_"))
                
                if obj:
                    # Query for specific relationship
                    query = """
                        MATCH (a)-[r]-(b)
                        WHERE toLower(a.name) CONTAINS toLower($subject)
                          AND toLower(b.name) CONTAINS toLower($object)
                          AND type(r) = $rel_type
                        RETURN count(r) as count
                    """
                    result = session.run(query, subject=subject, object=obj, rel_type=rel_type)
                    record = result.single()
                    
                    if record and record["count"] > 0:
                        return {"found": True, "count": record["count"]}
                    
                    # Check if entities at least exist
                    query = """
                        MATCH (a), (b)
                        WHERE toLower(a.name) CONTAINS toLower($subject)
                          AND toLower(b.name) CONTAINS toLower($object)
                        RETURN count(*) as count
                    """
                    result = session.run(query, subject=subject, object=obj)
                    record = result.single()
                    
                    return {
                        "found": False,
                        "entities_exist": record and record["count"] > 0,
                    }
                else:
                    # Query for any relationship of this type from subject
                    query = """
                        MATCH (a)-[r]->()
                        WHERE toLower(a.name) CONTAINS toLower($subject)
                          AND type(r) = $rel_type
                        RETURN count(r) as count
                    """
                    result = session.run(query, subject=subject, rel_type=rel_type)
                    record = result.single()
                    
                    return {
                        "found": record and record["count"] > 0,
                        "count": record["count"] if record else 0,
                    }
                    
        except Exception as e:
            print(f"Neo4j query error: {e}")
            return None
    
    def _query_event_order_neo4j(self, claim: ParsedClaim) -> ValidationResult | None:
        """Query Neo4j for event ordering."""
        if not self._neo4j_driver:
            return None
        
        try:
            with self._neo4j_driver.session() as session:
                # Build search terms for first event
                event1_terms = []
                if claim.event1_agent:
                    event1_terms.append(claim.event1_agent.lower())
                if claim.event1_action:
                    event1_terms.append(claim.event1_action.lower())
                if claim.event1_patient:
                    event1_terms.append(claim.event1_patient.lower().replace("the ", ""))
                
                # Build search terms for second event
                event2_terms = []
                if claim.event2_agent:
                    event2_terms.append(claim.event2_agent.lower())
                if claim.event2_action:
                    event2_terms.append(claim.event2_action.lower())
                if claim.event2_patient:
                    event2_terms.append(claim.event2_patient.lower().replace("the ", ""))
                
                if not event1_terms or not event2_terms:
                    return None
                
                # Query for matching events
                query = """
                MATCH (e1:Event), (e2:Event)
                WHERE (toLower(e1.agent) CONTAINS $agent1 OR toLower(e1.description) CONTAINS $desc1)
                  AND (toLower(e2.agent) CONTAINS $agent2 OR toLower(e2.description) CONTAINS $desc2)
                OPTIONAL MATCH (e1)-[r]->(e2)
                WHERE type(r) IN ['BEFORE', 'AFTER', 'DURING', 'CAUSES']
                RETURN e1.id as e1_id, e1.description as e1_desc, e1.era as e1_era, e1.year as e1_year,
                       e2.id as e2_id, e2.description as e2_desc, e2.era as e2_era, e2.year as e2_year,
                       type(r) as relation
                LIMIT 1
                """
                
                result = session.run(
                    query,
                    agent1=event1_terms[0] if event1_terms else "",
                    desc1=" ".join(event1_terms),
                    agent2=event2_terms[0] if event2_terms else "",
                    desc2=" ".join(event2_terms),
                )
                record = result.single()
                
                if not record:
                    return None
                
                # Build validation result
                vr = ValidationResult(
                    claim=claim,
                    status=ValidationStatus.UNKNOWN,
                    confidence=0.0,
                )
                
                ordering = record["relation"]
                
                # If no direct relation, infer from era/year
                if not ordering:
                    era1 = record["e1_era"]
                    era2 = record["e2_era"]
                    year1 = record["e1_year"]
                    year2 = record["e2_year"]
                    
                    era_order = {
                        "first_age": 1,
                        "second_age": 2,
                        "third_age": 3,
                        "fourth_age": 4,
                    }
                    
                    if era1 and era2:
                        if era_order.get(era1, 0) < era_order.get(era2, 0):
                            ordering = "BEFORE"
                        elif era_order.get(era1, 0) > era_order.get(era2, 0):
                            ordering = "AFTER"
                        elif year1 and year2:
                            if year1 < year2:
                                ordering = "BEFORE"
                            elif year1 > year2:
                                ordering = "AFTER"
                
                if ordering:
                    ordering_lower = ordering.lower()
                    claimed_order = claim.ordering.lower() if claim.ordering else "before"
                    
                    # Compute confidence from match quality
                    claim_event1 = f"{claim.event1_agent or ''} {claim.event1_action or ''} {claim.event1_patient or ''}".strip()
                    claim_event2 = f"{claim.event2_agent or ''} {claim.event2_action or ''} {claim.event2_patient or ''}".strip()
                    claim_text = f"{claim_event1} {claim_event2}"
                    found_text = f"{record['e1_desc']} {record['e2_desc']}"
                    evidence_type = "direct" if record["relation"] else ("year" if record["e1_year"] and record["e2_year"] else "era")
                    confidence = _compute_confidence(claim_text, found_text, evidence_type)
                    
                    if ordering_lower == claimed_order:
                        vr.status = ValidationStatus.VALID
                        vr.confidence = confidence
                        vr.explanation = f"Event '{record['e1_desc']}' is {ordering_lower} '{record['e2_desc']}'"
                        vr.supporting.append(Evidence(
                            text=f"{record['e1_desc']} {ordering_lower} {record['e2_desc']}",
                            source="Neo4j Event Graph",
                            supports=True,
                        ))
                    else:
                        vr.status = ValidationStatus.INVALID
                        vr.confidence = confidence
                        vr.explanation = f"Event '{record['e1_desc']}' is actually {ordering_lower} '{record['e2_desc']}', not {claimed_order}"
                        vr.contradicting.append(Evidence(
                            text=f"{record['e1_desc']} is {ordering_lower} {record['e2_desc']}",
                            source="Neo4j Event Graph",
                            supports=False,
                        ))
                else:
                    vr.status = ValidationStatus.UNKNOWN
                    vr.explanation = f"Cannot determine ordering between events"
                
                return vr
                
        except Exception as e:
            print(f"Neo4j event query error: {e}")
            return None
    
    def _check_temporal(self, claim: ParsedClaim, result: ValidationResult) -> ValidationResult:
        """Check temporal claims (X lived in Y Age)."""
        if not claim.subject or not claim.time_period:
            return result
        
        era = Era.from_text(claim.time_period)
        
        # Check timeline if available
        if self.timeline and era != Era.UNKNOWN:
            alive = self.timeline.alive_during_era(claim.subject, era)
            
            if alive is True:
                result.status = ValidationStatus.VALID
                result.confidence = 0.9
                result.explanation = f"{claim.subject} was alive during the {claim.time_period}"
                
                entity = self.timeline.get_entity(claim.subject)
                if entity and entity.source_text:
                    result.supporting.append(Evidence(
                        text=entity.source_text,
                        source="Timeline",
                        supports=True,
                    ))
                return result
            
            elif alive is False:
                result.status = ValidationStatus.INVALID
                result.confidence = 0.9
                
                entity = self.timeline.get_entity(claim.subject)
                if entity:
                    if entity.birth_era and era < entity.birth_era:
                        result.explanation = f"{claim.subject} was not yet born in the {claim.time_period} (born in {entity.birth_era.value.replace('_', ' ').title()})"
                    elif entity.death_era and era > entity.death_era:
                        result.explanation = f"{claim.subject} had already died by the {claim.time_period} (died in {entity.death_era.value.replace('_', ' ').title()})"
                    
                    if entity.source_text:
                        result.contradicting.append(Evidence(
                            text=entity.source_text,
                            source="Timeline",
                            supports=False,
                        ))
                return result
        
        # Fall back to world bible search
        relevant = self._find_relevant_rules(claim.subject, claim.time_period)
        
        for rule in relevant:
            rule_text = rule["description"].lower()
            subject_lower = claim.subject.lower()
            period_lower = claim.time_period.lower()
            
            if subject_lower in rule_text and period_lower in rule_text:
                result.supporting.append(Evidence(
                    text=rule["description"],
                    source=rule["source"],
                    supports=True,
                ))
            # Check for contradicting eras
            elif subject_lower in rule_text:
                for other_era in ["first age", "second age", "third age", "fourth age"]:
                    if other_era in rule_text and other_era != period_lower:
                        result.contradicting.append(Evidence(
                            text=rule["description"],
                            source=rule["source"],
                            supports=False,
                        ))
        
        if result.contradicting and not result.supporting:
            result.status = ValidationStatus.INVALID
            result.confidence = 0.7
            result.explanation = f"Evidence suggests {claim.subject} was in a different era"
        elif result.supporting:
            result.status = ValidationStatus.VALID
            result.confidence = 0.7
        
        return result
    
    def _check_capability(self, claim: ParsedClaim, result: ValidationResult) -> ValidationResult:
        """Check capability claims (X can do Y)."""
        # Search magic rules and creature rules
        if claim.subject and claim.value:
            relevant = self._find_relevant_rules(claim.subject, claim.value)
            relevant.extend(self._find_relevant_rules(claim.subject, "can"))
            
            for rule in relevant:
                result.supporting.append(Evidence(
                    text=rule["description"],
                    source=rule["source"],
                    supports=True,
                ))
            
            if result.supporting:
                result.status = ValidationStatus.PLAUSIBLE
                result.confidence = 0.6
        
        return result
    
    def _check_location(self, claim: ParsedClaim, result: ValidationResult) -> ValidationResult:
        """Check location claims."""
        return self._check_against_rules(claim, result)
    
    def _check_event_order(self, claim: ParsedClaim, result: ValidationResult) -> ValidationResult:
        """Check event ordering claims (X did Y before Z did W)."""
        # Try Neo4j first if connected and no local event graph
        if self._neo4j_driver and not self.event_graph:
            neo4j_result = self._query_event_order_neo4j(claim)
            if neo4j_result:
                return neo4j_result
        
        if not self.event_graph:
            result.status = ValidationStatus.UNKNOWN
            result.explanation = "No event graph loaded. Run 'bga lore events --neo4j' to extract events first."
            return result
        
        # Find matching events in the graph
        events1 = self.event_graph.find_events(
            agent=claim.event1_agent,
            action=claim.event1_action,
            patient=claim.event1_patient,
        )
        
        events2 = self.event_graph.find_events(
            agent=claim.event2_agent,
            action=claim.event2_action,
            patient=claim.event2_patient,
        )
        
        if not events1:
            result.explanation = f"Could not find event: {claim.event1_agent} {claim.event1_action} {claim.event1_patient}"
            return result
        
        if not events2:
            result.explanation = f"Could not find event: {claim.event2_agent} {claim.event2_action} {claim.event2_patient}"
            return result
        
        # Check ordering between first matching events
        e1 = events1[0]
        e2 = events2[0]
        
        ordering = self.event_graph.get_ordering(e1.id, e2.id)
        
        if ordering:
            claimed_order = claim.ordering  # "before" or "after"
            
            # Compute confidence from match quality
            claim_event1 = f"{claim.event1_agent or ''} {claim.event1_action or ''} {claim.event1_patient or ''}".strip()
            claim_event2 = f"{claim.event2_agent or ''} {claim.event2_action or ''} {claim.event2_patient or ''}".strip()
            claim_text = f"{claim_event1} {claim_event2}"
            found_text = f"{e1.description} {e2.description}"
            evidence_type = "year" if (e1.year and e2.year) else ("era" if (e1.era and e2.era) else "direct")
            confidence = _compute_confidence(claim_text, found_text, evidence_type)
            
            if ordering == claimed_order:
                result.status = ValidationStatus.VALID
                result.confidence = confidence
                result.explanation = f"Event '{e1.description}' is {ordering} '{e2.description}'"
                result.supporting.append(Evidence(
                    text=f"{e1.description} ({e1.year_text or e1.era.value if e1.era else 'unknown'}) {ordering} {e2.description} ({e2.year_text or e2.era.value if e2.era else 'unknown'})",
                    source="Event Graph",
                    supports=True,
                ))
            else:
                result.status = ValidationStatus.INVALID
                result.confidence = confidence
                result.explanation = f"Event '{e1.description}' is actually {ordering} '{e2.description}', not {claimed_order}"
                result.contradicting.append(Evidence(
                    text=f"{e1.description} is {ordering} {e2.description}",
                    source="Event Graph",
                    supports=False,
                ))
                
                # Add suggestion
                correct_claim = claim.original_text.replace(claimed_order, ordering)
                result.suggestions.append(f"Did you mean: {correct_claim}?")
        else:
            result.status = ValidationStatus.UNKNOWN
            result.explanation = f"Cannot determine ordering between '{e1.description}' and '{e2.description}'"
        
        return result
    
    def _check_event_order_llm(self, claim: ParsedClaim, result: ValidationResult) -> ValidationResult:
        """Use LLM to check event ordering when no event graph available."""
        if not self.use_llm:
            return result
        
        # Build context from world bible if available
        context = ""
        if self.world_bible and self._rule_cache:
            # Find relevant rules
            relevant = []
            for term in [claim.event1_agent, claim.event2_agent, claim.event1_patient]:
                if term:
                    relevant.extend(self._find_relevant_rules(term)[:5])
            
            if relevant:
                context = "World knowledge:\n" + "\n".join([
                    f"- {r['description'][:150]}" for r in relevant[:10]
                ])
        
        prompt = f"""Evaluate if this claim about event ordering is correct based on fantasy lore.

Claim: "{claim.original_text}"

{context}

Based on general Tolkien lore (if applicable) or the provided context:
1. When did "{claim.event1_agent} {claim.event1_action} {claim.event1_patient}" occur?
2. When did "{claim.event2_agent} {claim.event2_action or claim.event1_action} {claim.event2_patient or claim.event1_patient}" occur?
3. Is the claimed ordering ({claim.ordering}) correct?

Return JSON:
{{
  "status": "valid" | "invalid" | "unknown",
  "event1_time": "description of when event 1 happened",
  "event2_time": "description of when event 2 happened", 
  "actual_ordering": "before" | "after" | "same" | "unknown",
  "explanation": "brief explanation"
}}

JSON:"""

        llm = LLMClient()
        response = llm.generate(prompt, temperature=0.1, max_tokens=500)
        
        if response:
            data = llm.extract_json(response)
            if data and isinstance(data, dict):
                status_map = {
                    "valid": ValidationStatus.VALID,
                    "invalid": ValidationStatus.INVALID,
                    "unknown": ValidationStatus.UNKNOWN,
                }
                result.status = status_map.get(data.get("status", "unknown"), ValidationStatus.UNKNOWN)
                result.confidence = 0.7 if result.status != ValidationStatus.UNKNOWN else 0.3
                result.explanation = data.get("explanation", "")
                
                if data.get("event1_time"):
                    result.supporting.append(Evidence(
                        text=f"{claim.event1_agent}'s action: {data['event1_time']}",
                        source="LLM Analysis",
                        supports=(result.status == ValidationStatus.VALID),
                    ))
                
                if data.get("event2_time"):
                    result.supporting.append(Evidence(
                        text=f"{claim.event2_agent}'s action: {data['event2_time']}",
                        source="LLM Analysis",
                        supports=(result.status == ValidationStatus.VALID),
                    ))
                
                # Suggestion if invalid
                if result.status == ValidationStatus.INVALID and data.get("actual_ordering"):
                    actual = data["actual_ordering"]
                    if actual in ("before", "after") and actual != claim.ordering:
                        correct_claim = claim.original_text.replace(claim.ordering, actual)
                        result.suggestions.append(f"Did you mean: {correct_claim}?")
        
        return result
    
    def _check_against_rules(self, claim: ParsedClaim, result: ValidationResult) -> ValidationResult:
        """General check against all rules."""
        search_terms = []
        if claim.subject:
            search_terms.append(claim.subject)
        if claim.object:
            search_terms.append(claim.object)
        if claim.value:
            search_terms.append(claim.value)
        
        for term in search_terms:
            relevant = self._find_relevant_rules(term)
            for rule in relevant:
                result.supporting.append(Evidence(
                    text=rule["description"],
                    source=rule["source"],
                    supports=True,
                    relevance=0.5,  # Lower relevance for general matches
                ))
        
        if result.supporting:
            result.status = ValidationStatus.PLAUSIBLE
            result.confidence = 0.4
        
        return result
    
    def _check_semantic(self, claim: ParsedClaim, result: ValidationResult) -> ValidationResult:
        """Use LLM for semantic matching against world bible."""
        if not self.world_bible or not self._rule_cache:
            return result
        
        # Build context from relevant rules
        context_rules = self._rule_cache[:30]  # Limit context size
        context = "\n".join([
            f"- {r['title']}: {r['description'][:200]}"
            for r in context_rules
        ])
        
        prompt = f"""Based on this world knowledge, evaluate if the claim is valid, invalid, or unknown.

World Knowledge:
{context}

Claim to evaluate: "{claim.original_text}"

Respond with JSON:
{{
  "status": "valid" | "invalid" | "unknown" | "plausible",
  "confidence": 0.0-1.0,
  "explanation": "brief reason",
  "contradicts": "any contradicting rule (or null)",
  "supports": "any supporting rule (or null)"
}}

JSON:"""

        llm = LLMClient()
        response = llm.generate(prompt, temperature=0.1, max_tokens=500)
        
        if response:
            data = llm.extract_json(response)
            if data and isinstance(data, dict):
                status_map = {
                    "valid": ValidationStatus.VALID,
                    "invalid": ValidationStatus.INVALID,
                    "unknown": ValidationStatus.UNKNOWN,
                    "plausible": ValidationStatus.PLAUSIBLE,
                }
                result.status = status_map.get(data.get("status", "unknown"), ValidationStatus.UNKNOWN)
                result.confidence = float(data.get("confidence", 0.5))
                result.explanation = data.get("explanation", "")
                
                if data.get("supports"):
                    result.supporting.append(Evidence(
                        text=data["supports"],
                        source="World Bible (LLM match)",
                        supports=True,
                    ))
                
                if data.get("contradicts"):
                    result.contradicting.append(Evidence(
                        text=data["contradicts"],
                        source="World Bible (LLM match)",
                        supports=False,
                    ))
        
        return result
    
    def _find_relevant_rules(self, *terms: str) -> list[dict]:
        """Find rules relevant to the given terms."""
        relevant = []
        
        for rule in self._rule_cache:
            text = f"{rule['title']} {rule['description']}".lower()
            for term in terms:
                if term and term.lower() in text:
                    relevant.append(rule)
                    break
        
        return relevant
    
    def _rule_supports_claim(self, rule: dict, claim: ParsedClaim) -> bool:
        """Check if a rule supports the claim."""
        # Simple heuristic - if claim terms appear in positive context
        text = rule["description"].lower()
        
        if claim.subject and claim.subject.lower() in text:
            if claim.value and claim.value.lower() in text:
                # Both subject and value present - likely supports
                return not claim.negated
        
        return False
    
    def _rule_contradicts_claim(self, rule: dict, claim: ParsedClaim) -> bool:
        """Check if a rule contradicts the claim."""
        # Simple heuristic
        text = rule["description"].lower()
        
        if claim.subject and claim.subject.lower() in text:
            # Check for negation in rule
            if any(neg in text for neg in ["not", "never", "cannot", "forbidden"]):
                if claim.value and claim.value.lower() in text:
                    return not claim.negated  # Double negation
        
        return False
