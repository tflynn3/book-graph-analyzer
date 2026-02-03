"""Quick script to check graph contents."""
from book_graph_analyzer.graph.connection import get_driver

driver = get_driver()
with driver.session() as session:
    # Count by label
    print("=== Node Counts ===")
    result = session.run("""
        MATCH (n) 
        RETURN labels(n)[0] as label, count(*) as count 
        ORDER BY count DESC
    """)
    for r in result:
        print(f"  {r['label']}: {r['count']}")
    
    print("\n=== Sample Characters ===")
    result = session.run("MATCH (c:Character) RETURN c.name as name LIMIT 20")
    for r in result:
        print(f"  - {r['name']}")
    
    print("\n=== Sample Relationships ===")
    result = session.run("""
        MATCH (a)-[r]->(b) 
        RETURN a.name as src, type(r) as rel, b.name as dst
        LIMIT 15
    """)
    for r in result:
        print(f"  {r['src']} --[{r['rel']}]--> {r['dst']}")

driver.close()
