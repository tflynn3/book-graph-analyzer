"""Find cross-book references."""
from book_graph_analyzer.graph.connection import get_driver

d = get_driver()
s = d.session()

print("=== Searching for Aragorn/Strider ===")
r = s.run("""
    MATCH (c) 
    WHERE toLower(c.name) CONTAINS 'aragorn' OR toLower(c.name) CONTAINS 'strider'
    RETURN c.name, labels(c) LIMIT 10
""")
for x in r: 
    print(f"  {x[0]} ({x[1]})")

print()
print("=== Searching for Beren/Luthien ===")
r = s.run("""
    MATCH (c) 
    WHERE toLower(c.name) CONTAINS 'beren' OR toLower(c.name) CONTAINS 'luthien'
    RETURN c.name, labels(c) LIMIT 10
""")
for x in r: 
    print(f"  {x[0]} ({x[1]})")

print()
print("=== Books in graph ===")
r = s.run("MATCH (b:Book) RETURN b.title, b.book_id")
for x in r:
    print(f"  {x[0]} ({x[1]})")

d.close()
