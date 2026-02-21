"""LoreRule model — lore rules as executable Cypher contracts.

A LoreRule is a world-law that can be validated against Neo4j graph data.
Each rule carries a `cypher_check` query that, when run, returns violation
strings. Empty result = no violation.

Hardness:
  HARD — never violates; hard violations block scene acceptance
  SOFT — usually holds; soft violations produce warnings, not rejections
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Valid category values
RULE_CATEGORIES = frozenset({
    "race",
    "magic",
    "cosmology",
    "geography",
    "politics",
    "metaphysics",
    "objects",
    "history",
})


@dataclass
class LoreRule:
    """A world-law represented as a graph node.

    Stored in Neo4j as (:LoreRule { ... }).
    The cypher_check field is a Cypher query that returns violation strings
    when the rule is violated by the proposed scene's entity set.
    """

    id: str
    statement: str                     # Human-readable: 'Elves cannot die of age or disease'
    category: str                      # One of RULE_CATEGORIES
    hardness: str                      # 'HARD' | 'SOFT'

    scope_entity_type: Optional[str] = None   # 'Elf' | 'Maia' | 'Ring-bearer' | None (universal)
    scope_era: Optional[str] = None           # 'Third Age' | None (all eras)
    cypher_check: Optional[str] = None        # Runnable Cypher returning violations
    confidence: float = 1.0
    source_passage_ids: list[str] = field(default_factory=list)
    contradicted_by_rule_ids: list[str] = field(default_factory=list)

    def to_neo4j_props(self) -> dict:
        """Serialise to Neo4j property dict."""
        props: dict = {
            "id": self.id,
            "statement": self.statement,
            "category": self.category,
            "hardness": self.hardness,
            "confidence": self.confidence,
            "source_passage_ids": self.source_passage_ids,
            "contradicted_by_rule_ids": self.contradicted_by_rule_ids,
        }
        if self.scope_entity_type is not None:
            props["scope_entity_type"] = self.scope_entity_type
        if self.scope_era is not None:
            props["scope_era"] = self.scope_era
        if self.cypher_check is not None:
            props["cypher_check"] = self.cypher_check
        return props

    @classmethod
    def from_dict(cls, d: dict) -> "LoreRule":
        return cls(
            id=d["id"],
            statement=d["statement"],
            category=d.get("category", "metaphysics"),
            hardness=d.get("hardness", "SOFT"),
            scope_entity_type=d.get("scope_entity_type"),
            scope_era=d.get("scope_era"),
            cypher_check=d.get("cypher_check"),
            confidence=float(d.get("confidence", 1.0)),
            source_passage_ids=list(d.get("source_passage_ids", [])),
            contradicted_by_rule_ids=list(d.get("contradicted_by_rule_ids", [])),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "statement": self.statement,
            "category": self.category,
            "hardness": self.hardness,
            "scope_entity_type": self.scope_entity_type,
            "scope_era": self.scope_era,
            "cypher_check": self.cypher_check,
            "confidence": self.confidence,
            "source_passage_ids": self.source_passage_ids,
            "contradicted_by_rule_ids": self.contradicted_by_rule_ids,
        }

    @property
    def is_hard(self) -> bool:
        return self.hardness == "HARD"

    @property
    def is_soft(self) -> bool:
        return self.hardness == "SOFT"


@dataclass
class LoreViolation:
    """A single lore violation detected by running a LoreRule's cypher_check."""

    rule_id: str
    rule_statement: str
    hardness: str          # 'HARD' | 'SOFT'
    description: str       # The violation message returned by cypher_check
    blocking: bool         # True for HARD, False for SOFT

    def __str__(self) -> str:
        tag = "[HARD]" if self.blocking else "[SOFT]"
        return f"{tag} {self.rule_statement}: {self.description}"


@dataclass
class LoreValidationResult:
    """Result of validating a scene or text against all applicable LoreRules."""

    scene_id: str
    passed: bool                        # True if no HARD violations
    hard_violations: list[LoreViolation] = field(default_factory=list)
    soft_warnings: list[LoreViolation] = field(default_factory=list)
    rules_checked: int = 0

    @property
    def has_hard_violations(self) -> bool:
        return len(self.hard_violations) > 0

    @property
    def has_soft_warnings(self) -> bool:
        return len(self.soft_warnings) > 0

    def all_violations(self) -> list[LoreViolation]:
        return self.hard_violations + self.soft_warnings

    def summary(self) -> str:
        lines = [f"Lore Validation: {self.scene_id}"]
        status = "✓ PASS" if self.passed else "✗ FAIL"
        lines.append(f"  Status: {status}  |  Rules checked: {self.rules_checked}")
        if self.hard_violations:
            lines.append(f"\n  HARD violations ({len(self.hard_violations)}) — BLOCKED:")
            for v in self.hard_violations:
                lines.append(f"    • [{v.rule_id}] {v.description}")
        if self.soft_warnings:
            lines.append(f"\n  SOFT warnings ({len(self.soft_warnings)}) — allowed:")
            for v in self.soft_warnings:
                lines.append(f"    ~ [{v.rule_id}] {v.description}")
        if not self.hard_violations and not self.soft_warnings:
            lines.append("  No violations found.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "passed": self.passed,
            "rules_checked": self.rules_checked,
            "hard_violations": [
                {"rule_id": v.rule_id, "description": v.description}
                for v in self.hard_violations
            ],
            "soft_warnings": [
                {"rule_id": v.rule_id, "description": v.description}
                for v in self.soft_warnings
            ],
        }
