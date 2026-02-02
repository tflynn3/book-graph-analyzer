"""Load The Hobbit data into Neo4j for visualization."""

import json
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from book_graph_analyzer.graph.connection import get_driver, init_schema
from book_graph_analyzer.config import get_settings


def main():
    print("Loading The Hobbit data into Neo4j...")
    
    # Load the generic analysis results
    data_file = Path("data/exports/hobbit_generic.json")
    if not data_file.exists():
        print(f"Error: {data_file} not found. Run 'bga analyze' first.")
        return
    
    with open(data_file) as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['entities'])} entities, {len(data['relationships'])} relationships")
    
    # Connect to Neo4j
    driver = get_driver()
    if not driver:
        print("Error: Could not connect to Neo4j")
        return
    
    # Initialize schema
    print("Initializing schema...")
    init_schema()
    
    # Clear existing data
    print("Clearing existing data...")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    
    # Load entities
    print("Loading entities...")
    with driver.session() as session:
        for entity in data['entities']:
            # Skip entities with very few mentions
            if entity['mentions'] < 3:
                continue
            
            label = entity['type'].title() if entity['type'] != 'unknown' else 'Entity'
            
            session.run(f"""
                CREATE (e:{label} {{
                    id: $id,
                    name: $name,
                    mentions: $mentions,
                    aliases: $aliases
                }})
            """, 
                id=entity['id'],
                name=entity['canonical_name'],
                mentions=entity['mentions'],
                aliases=entity.get('aliases', []),
            )
    
    # Count nodes created
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count")
        node_count = result.single()["count"]
        print(f"Created {node_count} nodes")
    
    # Load relationships
    print("Loading relationships...")
    rel_count = 0
    with driver.session() as session:
        for rel in data['relationships']:
            try:
                result = session.run("""
                    MATCH (a {id: $source})
                    MATCH (b {id: $target})
                    MERGE (a)-[r:""" + rel['predicate'] + """]->(b)
                    ON CREATE SET r.count = 1
                    ON MATCH SET r.count = r.count + 1
                    RETURN r
                """,
                    source=rel['subject_id'],
                    target=rel['object_id'],
                )
                if result.single():
                    rel_count += 1
            except Exception as e:
                pass  # Skip relationships where entities don't exist
    
    print(f"Created {rel_count} relationships")
    
    # Show sample of what's in the graph
    print("\nSample query - Characters with most connections:")
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Character)-[r]-()
            RETURN c.name as name, count(r) as connections
            ORDER BY connections DESC
            LIMIT 10
        """)
        for record in result:
            print(f"  {record['name']}: {record['connections']} connections")
    
    print("\nGraph loaded! Open http://localhost:7474 in your browser.")
    print("Default credentials: neo4j / bookgraph123")
    print("\nTry these Cypher queries:")
    print("  MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")
    print("  MATCH (c:Character) RETURN c ORDER BY c.mentions DESC LIMIT 20")
    print("  MATCH path = (a)-[*1..2]-(b) WHERE a.name = 'Bilbo' RETURN path LIMIT 30")
    
    driver.close()


if __name__ == "__main__":
    main()
