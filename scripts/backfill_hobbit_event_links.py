from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "bookgraph123")
BOOK_MATCH = "hobbit"
THRESHOLD = 0.78

GENERIC = {
    "they", "them", "their", "he", "she", "it", "we", "you", "i", "company", "group",
    "party", "men", "elves", "dwarves", "goblins", "orcs", "people", "someone", "companions",
    "the group", "the company", "dwarven company",
}


@dataclass
class Candidate:
    node_id: int
    label: str
    canonical_name: str
    aliases: list[str]


def norm(s: str | None) -> str:
    return " ".join((s or "").lower().strip().split())


def token_set(s: str) -> set[str]:
    return set(re.findall(r"[a-z]+", s.lower()))


def best_match(text: str, candidates: list[Candidate]):
    n = norm(text)
    if not n:
        return None, 0.0, "empty"

    # exact on name/alias
    for c in candidates:
        if n == norm(c.canonical_name) or any(n == norm(a) for a in c.aliases):
            return c, 1.0, "exact"

    # token overlap + containment
    n_tokens = token_set(n)
    best = (None, 0.0, "none")
    for c in candidates:
        forms = [c.canonical_name] + c.aliases
        for f in forms:
            fn = norm(f)
            if not fn:
                continue
            if n in fn or fn in n:
                score = 0.84
                if score > best[1]:
                    best = (c, score, "contains")
            ft = token_set(fn)
            if n_tokens and ft:
                j = len(n_tokens & ft) / len(n_tokens | ft)
                if j >= 0.5:
                    score = 0.8 + (j * 0.1)
                    if score > best[1]:
                        best = (c, score, "token_jaccard")
            fr = SequenceMatcher(None, n, fn).ratio()
            if fr >= 0.86:
                score = 0.6 + fr * 0.25
                if score > best[1]:
                    best = (c, score, "fuzzy")
    return best


def main(out_path: str = "data/output/hobbit_backfill_report.json"):
    d = GraphDatabase.driver(URI, auth=AUTH)
    with d.session() as s:
        cand_rows = s.run("""
            MATCH (n)
            WHERE n:Character OR n:Place OR n:Object
            RETURN id(n) AS node_id,
                   head(labels(n)) AS label,
                   coalesce(n.canonical_name, n.name, '') AS canonical_name,
                   coalesce(n.aliases, []) AS aliases
        """).data()
        candidates = [Candidate(**r) for r in cand_rows if r.get("canonical_name")]

        events = s.run("""
            MATCH (e:Event)
            WHERE toLower(coalesce(e.source_book,'')) CONTAINS $book
            RETURN e.id AS id, e.agent AS agent, e.patient AS patient
        """, book=BOOK_MATCH).data()

        linked = 0
        reason = Counter()
        method_counts = Counter()

        for e in events:
            eid = e["id"]
            created = 0
            for role, field, rel_type in [
                ("agent", e.get("agent"), "PARTICIPATED_IN"),
                ("patient", e.get("patient"), "INVOLVED_IN"),
            ]:
                mention = norm(field)
                if not mention:
                    continue
                if mention in GENERIC:
                    reason["generic agent/patient terms"] += 1
                    continue
                c, conf, method = best_match(mention, candidates)
                if c is None:
                    reason["no candidate entity found"] += 1
                    continue
                if conf < THRESHOLD:
                    reason["confidence below threshold"] += 1
                    continue
                q = f"""
                MATCH (n) WHERE id(n) = $nid
                MATCH (e:Event {{id: $eid}})
                MERGE (n)-[r:{rel_type}]->(e)
                SET r.role = $role,
                    r.link_method = $method,
                    r.link_confidence = $conf
                RETURN count(r) AS c
                """
                s.run(q, nid=c.node_id, eid=eid, role=role, method=method, conf=round(conf, 3)).single()
                method_counts[method] += 1
                created += 1

            if created > 0:
                linked += 1
            else:
                if not norm(e.get("agent")) and not norm(e.get("patient")):
                    reason["no agent/patient on event"] += 1

        report = {
            "events_seen": len(events),
            "events_with_new_links": linked,
            "link_methods": dict(method_counts),
            "unlinked_reasons": dict(reason),
            "threshold": THRESHOLD,
        }

    d.close()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
