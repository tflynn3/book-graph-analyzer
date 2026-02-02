"""Create a static graph image for sharing."""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx


def main():
    print("Creating graph image...")
    
    # Load seed data
    entity_info = {}
    
    seed_chars = Path("data/seeds/characters.json")
    if seed_chars.exists():
        with open(seed_chars) as f:
            for char in json.load(f):
                entity_info[char['id']] = {
                    'name': char['canonical_name'],
                    'type': 'character',
                }
    
    seed_places = Path("data/seeds/places.json")
    if seed_places.exists():
        with open(seed_places) as f:
            for place in json.load(f):
                entity_info[place['id']] = {
                    'name': place['canonical_name'],
                    'type': 'place',
                }
    
    # Load relationship data
    rel_file = Path("data/exports/hobbit_relationships.json")
    if rel_file.exists():
        with open(rel_file) as f:
            rel_data = json.load(f)
    else:
        print("No relationship file")
        return
    
    # Build graph
    G = nx.Graph()  # Undirected for cleaner visualization
    
    # Count relationships between entities
    edge_counts = defaultdict(int)
    entity_mentions = defaultdict(int)
    
    for record in rel_data.get('relationships', []):
        for rel in record.get('relationships', []):
            subj = rel.get('subject_id')
            obj = rel.get('object_id')
            
            if subj and obj and subj in entity_info and obj in entity_info:
                edge_key = tuple(sorted([subj, obj]))
                edge_counts[edge_key] += 1
                entity_mentions[subj] += 1
                entity_mentions[obj] += 1
    
    # Add top characters from the data
    for name, count in rel_data.get('top_characters', []):
        for eid, info in entity_info.items():
            if info['name'] == name:
                entity_mentions[eid] = max(entity_mentions[eid], count)
    
    # Get nodes with relationships
    nodes_with_edges = set()
    for (a, b) in edge_counts.keys():
        nodes_with_edges.add(a)
        nodes_with_edges.add(b)
    
    # Add key characters even without relationships
    key_chars = ['bilbo_baggins', 'gandalf', 'thorin_oakenshield', 'smaug', 
                 'gollum', 'balin', 'beorn', 'bard', 'elrond']
    for char in key_chars:
        if char in entity_info:
            nodes_with_edges.add(char)
    
    # Add nodes
    for eid in nodes_with_edges:
        info = entity_info.get(eid, {'name': eid, 'type': 'unknown'})
        G.add_node(eid, **info)
    
    # Add edges
    for (a, b), count in edge_counts.items():
        if count >= 1:
            G.add_edge(a, b, weight=count)
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    if G.number_of_nodes() == 0:
        print("No nodes to visualize!")
        return
    
    # Create figure
    plt.figure(figsize=(16, 12), facecolor='#1a1a2e')
    ax = plt.gca()
    ax.set_facecolor('#1a1a2e')
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node colors by type
    colors = []
    for node in G.nodes():
        ntype = G.nodes[node].get('type', 'unknown')
        if ntype == 'character':
            colors.append('#4CAF50')
        elif ntype == 'place':
            colors.append('#2196F3')
        elif ntype == 'object':
            colors.append('#FF9800')
        else:
            colors.append('#9E9E9E')
    
    # Node sizes by mentions
    sizes = []
    for node in G.nodes():
        mentions = entity_mentions.get(node, 1)
        sizes.append(300 + mentions * 20)
    
    # Draw edges
    edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='#ffffff', 
                           width=[w * 0.5 for w in edge_weights])
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.9)
    
    # Labels
    labels = {node: G.nodes[node].get('name', node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_color='white',
                           font_weight='bold')
    
    plt.title("The Hobbit - Character & Place Network", fontsize=16, color='white', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save
    output_path = Path("data/exports/hobbit_graph.png")
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', edgecolor='none',
                bbox_inches='tight')
    plt.close()
    
    print(f"Saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
