from __future__ import annotations

import json
from pathlib import Path

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "bookgraph123")

SEEDS = [
    ("Character", Path("data/seeds/characters.json")),
    ("Place", Path("data/seeds/places.json")),
    ("Object", Path("data/seeds/objects.json")),
]


def upsert_entities(session, label: str, entities: list[dict]) -> int:
    q = f"""
    UNWIND $rows AS row
    MERGE (n:{label} {{canonical_id: row.id}})
    SET n.canonical_name = row.canonical_name,
        n.name = row.canonical_name,
        n.aliases = row.aliases,
        n.entity_type = toLower($label)
    RETURN count(*) AS c
    """
    rows = [
        {
            "id": e["id"],
            "canonical_name": e["canonical_name"],
            "aliases": e.get("aliases", []),
        }
        for e in entities
        if e.get("id") and e.get("canonical_name")
    ]
    if not rows:
        return 0
    return int(session.run(q, rows=rows, label=label).single()["c"])


def main() -> None:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as s:
        total = 0
        for label, path in SEEDS:
            data = json.loads(path.read_text(encoding="utf-8"))
            n = upsert_entities(s, label, data)
            total += n
            print(f"{label}: upserted {n}")
    driver.close()
    print(f"Total upserted: {total}")


if __name__ == "__main__":
    main()
