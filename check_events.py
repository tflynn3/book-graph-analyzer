"""Quick event check script."""
from book_graph_analyzer.graph.connection import get_driver

driver = get_driver()
with driver.session() as session:
    # Check Smaug events
    result = session.run("""
        MATCH (e:Event) 
        WHERE toLower(e.description) CONTAINS 'smaug'
        RETURN e.description
        LIMIT 10
    """)
    print("Smaug events in Neo4j:")
    for r in result:
        print(f"  - {r['e.description']}")
    
    # Check relations
    result = session.run("""
        MATCH (e1:Event)-[r:BEFORE]->(e2:Event)
        WHERE toLower(e1.description) CONTAINS 'battle'
           OR toLower(e2.description) CONTAINS 'battle'
        RETURN e1.description, e2.description
        LIMIT 5
    """)
    print("\nBattle temporal relations:")
    for r in result:
        print(f"  - {r['e1.description']} BEFORE {r['e2.description']}")
    
    # Count totals
    result = session.run("MATCH (e:Event) RETURN count(e) as cnt")
    print(f"\nTotal events in Neo4j: {result.single()['cnt']}")
    
    result = session.run("MATCH ()-[r:BEFORE]->() RETURN count(r) as cnt")
    print(f"Total BEFORE relations: {result.single()['cnt']}")

driver.close()
