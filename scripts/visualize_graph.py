"""Generate an interactive HTML visualization of the character graph."""

import json
from pathlib import Path
from pyvis.network import Network
import networkx as nx


def main():
    # Load the analysis results
    data_file = Path("data/exports/hobbit_generic.json")
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        return
    
    with open(data_file) as f:
        data = json.load(f)
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes for top entities (by mentions)
    entity_map = {e['id']: e for e in data['entities']}
    
    # Filter to entities with enough mentions
    top_entities = [e for e in data['entities'] if e['mentions'] >= 10]
    
    print(f"Adding {len(top_entities)} entities with 10+ mentions")
    
    for entity in top_entities:
        # Size based on mentions (log scale)
        size = min(50, 10 + entity['mentions'] // 5)
        
        # Color by type
        colors = {
            'character': '#4CAF50',  # Green
            'place': '#2196F3',       # Blue
            'object': '#FF9800',      # Orange
            'unknown': '#9E9E9E',     # Gray
        }
        color = colors.get(entity['type'], '#9E9E9E')
        
        G.add_node(
            entity['id'],
            label=entity['canonical_name'],
            title=f"{entity['canonical_name']}\nType: {entity['type']}\nMentions: {entity['mentions']}",
            size=size,
            color=color,
        )
    
    # Add edges for relationships
    edge_count = 0
    for rel in data['relationships']:
        source = rel['subject_id']
        target = rel['object_id']
        
        # Only add if both nodes exist in our filtered set
        if source in G.nodes and target in G.nodes:
            G.add_edge(
                source, 
                target,
                title=rel['predicate'],
                label=rel['predicate'],
            )
            edge_count += 1
    
    print(f"Added {edge_count} relationships")
    
    # Also add co-occurrence edges for entities that appear together frequently
    # This is inferred from being mentioned in the same passage
    print("Adding co-occurrence edges...")
    
    # Create pyvis network
    net = Network(
        height='800px', 
        width='100%',
        bgcolor='#222222',
        font_color='white',
        directed=True,
    )
    
    # Configure physics
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.05,
    )
    
    # Add from networkx
    net.from_nx(G)
    
    # Save
    output_path = Path("data/exports/hobbit_graph.html")
    net.save_graph(str(output_path))
    
    print(f"\nVisualization saved to: {output_path.absolute()}")
    print("Open this file in your browser to explore the graph!")
    
    # Also print some stats
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    if G.number_of_nodes() > 0:
        print(f"\nTop nodes by connections:")
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        for node_id, degree in top_nodes:
            name = entity_map.get(node_id, {}).get('canonical_name', node_id)
            print(f"  {name}: {degree} connections")


if __name__ == "__main__":
    main()
