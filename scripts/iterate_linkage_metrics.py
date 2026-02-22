from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "bookgraph123")
BOOK_MATCH = "hobbit"

GENERIC_TERMS = {
    "they", "them", "their", "theirs", "he", "she", "it", "we", "you", "i",
    "company", "group", "party", "troop", "army", "men", "elves", "dwarves",
    "goblins", "orcs", "wolves", "eagles", "people", "someone", "something",
    "everyone", "anyone", "others", "companions", "dwarven company", "the group",
}


@dataclass
class Metrics:
    total_events: int
    linked_events: int
    unlinked_events: int
    linked_pct: float
    link_type_counts: dict[str, int]
    duplicate_semantic_event_count: int
    direct_before_cycle_count: int


def normalize_text(v: str | None) -> str:
    if not v:
        return ""
    return " ".join(v.lower().strip().split())


def pull_events(session):
    q = """
    MATCH (e:Event)
    WHERE toLower(coalesce(e.source_book, '')) CONTAINS $book
    OPTIONAL MATCH (ent)-[r:PARTICIPATED_IN|INVOLVED_IN]->(e)
    RETURN e.id AS id,
           e.agent AS agent,
           e.action AS action,
           e.patient AS patient,
           collect(DISTINCT type(r) + ':' + coalesce(r.role, '')) AS rel_types,
           count(DISTINCT ent) AS linked_entities
    """
    return session.run(q, book=BOOK_MATCH).data()


def compute_metrics(rows: list[dict]) -> Metrics:
    total = len(rows)
    linked = sum(1 for r in rows if r["linked_entities"] > 0)
    unlinked = total - linked

    type_counts = Counter()
    semantic = Counter()
    for r in rows:
        for t in r["rel_types"]:
            if t != ":":
                type_counts[t] += 1
        key = (
            normalize_text(r.get("agent")),
            normalize_text(r.get("action")),
            normalize_text(r.get("patient")),
        )
        semantic[key] += 1

    dup_count = sum(v - 1 for v in semantic.values() if v > 1)

    # direct 2-cycles only (A->B and B->A)
    # divide by 2 because each pair is found twice
    direct_cycles = None
    # calculated outside in query for speed/clarity

    return Metrics(
        total_events=total,
        linked_events=linked,
        unlinked_events=unlinked,
        linked_pct=(linked / total * 100.0) if total else 0.0,
        link_type_counts=dict(type_counts),
        duplicate_semantic_event_count=dup_count,
        direct_before_cycle_count=direct_cycles or 0,
    )


def before_cycle_count(session) -> int:
    q = """
    MATCH (a:Event)-[:BEFORE]->(b:Event)
    WHERE toLower(coalesce(a.source_book, '')) CONTAINS $book
      AND toLower(coalesce(b.source_book, '')) CONTAINS $book
      AND (b)-[:BEFORE]->(a)
      AND id(a) < id(b)
    RETURN count(*) AS c
    """
    return int(session.run(q, book=BOOK_MATCH).single()["c"])


def reason_buckets(rows: list[dict]) -> dict[str, int]:
    buckets = Counter()
    for r in rows:
        if r["linked_entities"] > 0:
            continue
        agent = normalize_text(r.get("agent"))
        patient = normalize_text(r.get("patient"))
        action = normalize_text(r.get("action"))

        if (agent in GENERIC_TERMS) or (patient in GENERIC_TERMS):
            buckets["generic agent/patient term"] += 1
        elif not agent and not patient:
            buckets["no agent/patient on event"] += 1
        elif len(agent) <= 2 or len(patient) <= 2:
            buckets["too short/noisy mention"] += 1
        elif action in {"is", "was", "were", "had", "did", "went", "came"}:
            buckets["low-information action"] += 1
        else:
            buckets["no candidate entity found"] += 1
    return dict(buckets)


def main(out_path: str = "data/output/iterate_linkage_metrics.json") -> None:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as s:
        rows = pull_events(s)
        metrics = compute_metrics(rows)
        metrics.direct_before_cycle_count = before_cycle_count(s)
        reasons = reason_buckets(rows)

    payload = {
        "book_match": BOOK_MATCH,
        "metrics": asdict(metrics),
        "reasons": reasons,
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
