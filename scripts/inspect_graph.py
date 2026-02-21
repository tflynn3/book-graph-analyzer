"""Inspect what's actually in the Neo4j graph."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.chdir(Path(__file__).parent.parent)

from book_graph_analyzer.graph.connection import get_driver

d = get_driver()
with d.session() as s:
    print("=== Relationship Types ===")
    for row in s.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"):
        print(" ", row["relationshipType"])

    print("\n=== Non-Event Edges (sample) ===")
    rows = s.run("""
        MATCH (a)-[r]->(b) WHERE NOT a:Event AND NOT b:Event
        RETURN type(r) as t, labels(a)[0] as la, labels(b)[0] as lb,
               a.name as an, b.name as bn LIMIT 20
    """)
    for row in rows:
        print(f"  ({row['la']}) {row['an']} --{row['t']}--> ({row['lb']}) {row['bn']}")

    print("\n=== Character<->Event links ===")
    cnt = s.run("MATCH (c:Character)-[r]-(e:Event) RETURN count(r) as cnt").single()["cnt"]
    print(f"  {cnt} total")

    print("\n=== Event node sample (full) ===")
    evts = s.run("""
        MATCH (e:Event) WHERE e.agent IS NOT NULL
        RETURN e.description, e.agent, e.action, e.patient, e.era, e.year
        LIMIT 5
    """)
    for row in evts:
        print(f"  desc:    {row['e.description']}")
        print(f"  agent:   {row['e.agent']}")
        print(f"  action:  {row['e.action']}")
        print(f"  patient: {row['e.patient']}")
        print(f"  era/yr:  {row['e.era']} / {row['e.year']}")
        print()

    print("=== Event->Event relationship sample ===")
    evtrels = s.run("""
        MATCH (a:Event)-[r]->(b:Event)
        RETURN a.description as a, type(r) as rel, b.description as b
        LIMIT 5
    """)
    for row in evtrels:
        print(f"  {row['a'][:40]} --{row['rel']}--> {row['b'][:40]}")
