"""LoreConflict model — tracks intra/inter-book contradictions in the Tolkien corpus.

Tolkien revised his legendarium constantly; contradictions are canon, not errors.
LoreConflict nodes record these contradictions, their sources, and how to resolve
them so the lore checker avoids false positives and false negatives.

Node stored in Neo4j as (:LoreConflict { ... }).

Author revision periods follow Tolkien scholarship conventions:
  early  (~1910s-1930s): Book of Lost Tales, pre-Silmarillion
  middle (~1940s-1960s): LotR era, first Silmarillion drafts
  late   (~1960s-1973): The finished legendarium, Unfinished Tales manuscripts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConflictType(str, Enum):
    DIRECT_CONTRADICTION = "direct_contradiction"  # Two claims flatly contradict
    RETCON               = "retcon"                # Later revision supersedes earlier
    AMBIGUITY            = "ambiguity"             # Tolkien was deliberately vague
    INTERPRETATION       = "interpretation"        # Scholars disagree, not Tolkien himself


class ResolutionPolicy(str, Enum):
    USE_LATER_TEXT        = "use_later_text"         # Default for retcons
    USE_EARLIER_TEXT      = "use_earlier_text"       # Occasionally earlier is more reliable
    BOTH_VALID_IN_UNIVERSE = "both_valid_in_universe" # In-universe narrators differ
    FLAG_FOR_HUMAN        = "flag_for_human"          # Genuinely ambiguous
    USE_MOST_CITED        = "use_most_cited"          # Most referenced version wins
    IRRESOLVABLE          = "irresolvable"            # Tolkien died before resolving


class AuthorPeriod(str, Enum):
    EARLY  = "early"   # ~1910s-1930s: Book of Lost Tales, pre-Silmarillion
    MIDDLE = "middle"  # ~1940s-1960s: LotR era, first Silmarillion drafts
    LATE   = "late"    # ~1960s-1973: finished legendarium, Unfinished Tales


# Author period ordering (earlier = smaller number)
AUTHOR_PERIOD_ORDER: dict[str, int] = {
    AuthorPeriod.EARLY:  0,
    AuthorPeriod.MIDDLE: 1,
    AuthorPeriod.LATE:   2,
}


# ---------------------------------------------------------------------------
# ConflictClaim — one side of a contradiction
# ---------------------------------------------------------------------------

@dataclass
class ConflictClaim:
    """A single contradictory claim within a LoreConflict.

    Represents one 'side' of the conflict — a specific statement from a
    specific source with a known author period.
    """

    statement: str                            # The conflicting claim
    source_book: str                          # Book the claim comes from
    author_period: str                        # AuthorPeriod value
    confidence: float = 1.0
    source_passage_id: Optional[str] = None   # Passage node ID if available
    source_id: Optional[str] = None
    editorial_status: Optional[str] = None
    source_authority_weight: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "statement": self.statement,
            "source_book": self.source_book,
            "author_period": self.author_period,
            "confidence": self.confidence,
        }
        if self.source_passage_id:
            d["source_passage_id"] = self.source_passage_id
        if self.source_id:
            d["source_id"] = self.source_id
        if self.editorial_status:
            d["editorial_status"] = self.editorial_status
        if self.source_authority_weight is not None:
            d["source_authority_weight"] = self.source_authority_weight
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ConflictClaim":
        return cls(
            statement=d["statement"],
            source_book=d.get("source_book", "Unknown"),
            author_period=d.get("author_period", AuthorPeriod.MIDDLE),
            confidence=float(d.get("confidence", 1.0)),
            source_passage_id=d.get("source_passage_id"),
            source_id=d.get("source_id"),
            editorial_status=d.get("editorial_status"),
            source_authority_weight=(
                float(d["source_authority_weight"])
                if d.get("source_authority_weight") is not None
                else None
            ),
        )

    def period_order(self) -> int:
        """Return the chronological order of this claim's author period."""
        return AUTHOR_PERIOD_ORDER.get(self.author_period, 99)


# ---------------------------------------------------------------------------
# LoreConflict
# ---------------------------------------------------------------------------

@dataclass
class LoreConflict:
    """A tracked contradiction between two or more lore claims.

    Stored as (:LoreConflict { ... }) in Neo4j.
    Connected to LoreRules via (LoreRule)-[:CONFLICTS_WITH {conflict_id}]->(LoreRule).
    """

    id: str
    summary: str                              # Human-readable description
    conflict_type: str                        # ConflictType value
    claims: list[ConflictClaim] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)
    resolution_policy: str = ResolutionPolicy.FLAG_FOR_HUMAN
    resolution_notes: str = ""
    resolved: bool = False

    @property
    def is_resolved(self) -> bool:
        return self.resolved

    @property
    def needs_human_review(self) -> bool:
        return self.resolution_policy == ResolutionPolicy.FLAG_FOR_HUMAN and not self.resolved

    def winning_claim(self) -> Optional[ConflictClaim]:
        """Return the 'winning' claim per the resolution policy.

        Returns None if the conflict is unresolved or both claims are valid.
        """
        if not self.claims:
            return None

        policy = self.resolution_policy

        if policy == ResolutionPolicy.USE_LATER_TEXT:
            # Highest author_period order wins
            return max(self.claims, key=lambda c: c.period_order())

        elif policy == ResolutionPolicy.USE_EARLIER_TEXT:
            return min(self.claims, key=lambda c: c.period_order())

        elif policy == ResolutionPolicy.USE_MOST_CITED:
            # Highest confidence wins as a proxy for citation frequency
            return max(self.claims, key=lambda c: c.confidence)

        elif policy in (
            ResolutionPolicy.BOTH_VALID_IN_UNIVERSE,
            ResolutionPolicy.FLAG_FOR_HUMAN,
            ResolutionPolicy.IRRESOLVABLE,
        ):
            return None  # No single winner

        return None

    def suppresses_lore_violation(self, rule_id: str, entity_id: str) -> bool:
        """Should this conflict suppress a lore violation for the given rule + entity?

        Returns True when the conflict covers this rule/entity and the resolution
        policy indicates the 'violation' is actually a known ambiguity.
        """
        covers_rule = rule_id in self.rule_ids
        covers_entity = not self.entity_ids or entity_id in self.entity_ids

        if not (covers_rule and covers_entity):
            return False

        # both_valid_in_universe fully suppresses the violation
        if self.resolution_policy == ResolutionPolicy.BOTH_VALID_IN_UNIVERSE:
            return True

        return False

    def downgrades_to_soft(self, rule_id: str, entity_id: str) -> bool:
        """Should a HARD violation be downgraded to SOFT due to this conflict?

        Per spec: if resolution_policy = 'flag_for_human', downgrade HARD → SOFT.
        """
        covers_rule = rule_id in self.rule_ids
        covers_entity = not self.entity_ids or entity_id in self.entity_ids

        if not (covers_rule and covers_entity):
            return False

        return self.resolution_policy == ResolutionPolicy.FLAG_FOR_HUMAN

    def to_neo4j_props(self) -> dict:
        """Serialise to a flat Neo4j property dict (claims stored as JSON strings)."""
        import json
        return {
            "id": self.id,
            "summary": self.summary,
            "conflict_type": self.conflict_type,
            "entity_ids": self.entity_ids,
            "rule_ids": self.rule_ids,
            "claims": json.dumps([c.to_dict() for c in self.claims]),
            "resolution_policy": self.resolution_policy,
            "resolution_notes": self.resolution_notes,
            "resolved": self.resolved,
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "summary": self.summary,
            "conflict_type": self.conflict_type,
            "claims": [c.to_dict() for c in self.claims],
            "entity_ids": self.entity_ids,
            "rule_ids": self.rule_ids,
            "resolution_policy": self.resolution_policy,
            "resolution_notes": self.resolution_notes,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LoreConflict":
        claims_raw = d.get("claims", [])
        # Handle both JSON string (from Neo4j) and list (from Python)
        if isinstance(claims_raw, str):
            import json
            claims_raw = json.loads(claims_raw)
        return cls(
            id=d["id"],
            summary=d.get("summary", ""),
            conflict_type=d.get("conflict_type", ConflictType.AMBIGUITY),
            claims=[ConflictClaim.from_dict(c) for c in claims_raw],
            entity_ids=list(d.get("entity_ids", [])),
            rule_ids=list(d.get("rule_ids", [])),
            resolution_policy=d.get("resolution_policy", ResolutionPolicy.FLAG_FOR_HUMAN),
            resolution_notes=d.get("resolution_notes", ""),
            resolved=bool(d.get("resolved", False)),
        )

    def brief(self) -> str:
        """One-line summary for display."""
        status = "✓" if self.resolved else "?"
        return (
            f"[{status}] {self.id}  [{self.conflict_type}]  "
            f"({self.resolution_policy})  {self.summary[:80]}"
        )

    def detail(self) -> str:
        """Multi-line detail for display."""
        lines = [
            f"Conflict: {self.id}",
            f"  Type: {self.conflict_type}",
            f"  Resolution: {self.resolution_policy}",
            f"  Resolved: {'Yes' if self.resolved else 'No'}",
            f"  Summary: {self.summary}",
        ]
        if self.resolution_notes:
            lines.append(f"  Notes: {self.resolution_notes}")
        if self.entity_ids:
            lines.append(f"  Entities: {', '.join(self.entity_ids)}")
        if self.rule_ids:
            lines.append(f"  Rules: {', '.join(self.rule_ids)}")
        lines.append(f"  Claims ({len(self.claims)}):")
        for i, claim in enumerate(self.claims, 1):
            lines.append(
                f"    [{i}] ({claim.author_period} / {claim.source_book}) "
                f"conf={claim.confidence:.0%}: {claim.statement}"
            )
        winner = self.winning_claim()
        if winner:
            lines.append(f"  → Active claim: [{winner.author_period}] {winner.statement}")
        return "\n".join(lines)
