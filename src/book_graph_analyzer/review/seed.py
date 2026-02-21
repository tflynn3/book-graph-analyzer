"""Helpers to seed review queue from pipeline outputs."""

from __future__ import annotations

from typing import Any

from .store import ReviewStore


def seed_entities(store: ReviewStore, entities: list[dict[str, Any]]) -> int:
    """Add low-confidence entity resolution candidates to review queue."""
    added = 0
    for e in entities:
        conf = float(e.get("confidence", e.get("cluster_confidence", 0.0)))
        needs_review = bool(e.get("needs_review", False))
        if needs_review or (0.6 <= conf < 0.85):
            item_id = e.get("id") or f"entity_{e.get('canonical_name', 'unknown').lower().replace(' ', '_')}"
            store.add_item(
                "entity",
                conf,
                payload=e,
                item_id=item_id,
                source="entity_resolution_v2",
                needs_review=True,
            )
            added += 1
    return added


def seed_conflicts(store: ReviewStore, conflicts: list[dict[str, Any]]) -> int:
    """Add LoreConflicts with resolution_policy='flag_for_human'."""
    added = 0
    for c in conflicts:
        policy = str(c.get("resolution_policy", ""))
        if policy == "flag_for_human":
            item_id = c.get("id") or "conflict_unknown"
            store.add_item(
                "conflict",
                float(c.get("confidence", 0.7)),
                payload=c,
                item_id=item_id,
                source="lore_conflict",
                needs_review=True,
            )
            added += 1
    return added


def seed_rules(store: ReviewStore, rules: list[dict[str, Any]]) -> int:
    """Add low-confidence LoreRules to review queue."""
    added = 0
    for r in rules:
        conf = float(r.get("confidence", 0.0))
        if conf < 0.85:
            item_id = r.get("id") or f"rule_{abs(hash(r.get('statement', 'rule'))) % 10_000_000}"
            store.add_item(
                "rule",
                conf,
                payload=r,
                item_id=item_id,
                source="rule_extractor",
                needs_review=True,
            )
            added += 1
    return added


def seed_relationships(store: ReviewStore, relationships: list[dict[str, Any]]) -> int:
    """Add low-confidence relationship extractions (<0.75)."""
    added = 0
    for rel in relationships:
        conf = float(rel.get("confidence", 0.0))
        if conf < 0.75:
            rid = rel.get("id") or f"rel_{abs(hash(str(rel))) % 10_000_000}"
            store.add_item(
                "relationship",
                conf,
                payload=rel,
                item_id=rid,
                source="relationship_extractor",
                needs_review=True,
            )
            added += 1
    return added
