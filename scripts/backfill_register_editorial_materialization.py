#!/usr/bin/env python3
"""Backfill/register+editorial materialization repair.

Repairs two failure modes without re-extracting events:
1) Register nodes disconnected because entity join only used exact id.
2) Editorial/provenance edges missing due brittle entity/passage/source matching.

Usage:
  python scripts/backfill_register_editorial_materialization.py --dry-run
  python scripts/backfill_register_editorial_materialization.py --apply
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from book_graph_analyzer.graph.connection import get_driver, init_schema


@dataclass
class Stat:
    name: str
    count: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return p.parse_args()


def run_query(session, query: str, **params) -> int:
    rec = session.run(query, **params).single()
    if not rec:
        return 0
    # prefer explicit updated counter
    for key in ("updated", "count", "created"):
        if key in rec:
            return int(rec[key] or 0)
    return 0


def main() -> int:
    args = parse_args()
    dry_run = bool(args.dry_run)

    driver = get_driver()
    if not driver:
        raise SystemExit("Cannot connect to Neo4j")

    init_schema()

    with driver.session() as session:
        # Register profile backfill: attach to Character when only canonical_id/canonical_name matches.
        q_profile = """
        MATCH (rp:RegisterProfile)
        WHERE NOT EXISTS { MATCH (:Character)-[:HAS_REGISTER_PROFILE]->(rp) }
        OPTIONAL MATCH (c:Character)
        WHERE c.id = rp.entity_id
           OR c.canonical_id = rp.entity_id
           OR toLower(coalesce(c.canonical_name,'')) = toLower(replace(rp.entity_id,'char_',''))
        WITH rp, collect(c) AS chars
        WITH rp, [x IN chars WHERE x IS NOT NULL] AS chars
        WHERE size(chars) = 1
        WITH rp, head(chars) AS c
        FOREACH (_ IN CASE WHEN $apply THEN [1] ELSE [] END |
            MERGE (c)-[:HAS_REGISTER_PROFILE]->(rp)
        )
        RETURN count(rp) AS updated
        """

        q_obs = """
        MATCH (obs:RegisterObservation)
        WHERE NOT EXISTS { MATCH (:Character)-[:HAS_REGISTER_OBSERVATION]->(obs) }
        OPTIONAL MATCH (c:Character)
        WHERE c.id = obs.entity_id
           OR c.canonical_id = obs.entity_id
           OR toLower(coalesce(c.canonical_name,'')) = toLower(replace(obs.entity_id,'char_',''))
        WITH obs, collect(c) AS chars
        WITH obs, [x IN chars WHERE x IS NOT NULL] AS chars
        WHERE size(chars) = 1
        WITH obs, head(chars) AS c
        FOREACH (_ IN CASE WHEN $apply THEN [1] ELSE [] END |
            MERGE (c)-[:HAS_REGISTER_OBSERVATION]->(obs)
        )
        RETURN count(obs) AS updated
        """

        q_editorial_attest = """
        MATCH (p:Passage)
        WHERE p.source_id IS NOT NULL
          AND NOT EXISTS { MATCH (p)-[:ATTESTED_IN]->(:Source) }
        FOREACH (_ IN CASE WHEN $apply THEN [1] ELSE [] END |
            MERGE (s:Source {id: p.source_id})
            SET s.source_title = coalesce(s.source_title, p.source_title)
            MERGE (p)-[:ATTESTED_IN {source_stratum: coalesce(p.source_stratum, 'core_text'), confidence: 1.0}]->(s)
        )
        RETURN count(p) AS updated
        """

        stats = [
            Stat("register_profile_links", run_query(session, q_profile, apply=args.apply)),
            Stat("register_observation_links", run_query(session, q_obs, apply=args.apply)),
            Stat("passage_source_attested", run_query(session, q_editorial_attest, apply=args.apply)),
        ]

    driver.close()

    mode = "DRY-RUN" if dry_run else "APPLY"
    print(f"[{mode}] backfill summary")
    for s in stats:
        print(f"- {s.name}: {s.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
