"""Create a graph based on entity co-occurrence in passages."""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from book_graph_analyzer.extract import EntityExtractor
from book_graph_analyzer.ingest.loader import load_book
from book_graph_analyzer.ingest.splitter import split_into_passages


def main():
    print("Building co-occurrence graph from The Hobbit...")
    
    # Load and extract
    text_file = Path("data/texts/the_hobbit.txt")
    if not text_file.exists():
        print("Text file not found!")
        return
    
    extractor = EntityExtractor(use_llm=False)
    
    print("Extracting entities...")
    results, stats = extractor.extract_from_file(text_file, "The Hobbit")
    
    print(f"Found {stats.total_entities_resolved} resolved entities")
    
    # Build co-occurrence matrix
    cooccurrence = defaultdict(lambda: defaultdict(int))
    entity_counts = defaultdict(int)
    
    for result in results:
        # Get resolved entities in this passage
        resolved = [e for e in result.entities if e.canonical_id]
        
        # Count co-occurrences
        for i, e1 in enumerate(resolved):
            entity_counts[e1.canonical_id] += 1
            for e2 in resolved[i+1:]:
                if e1.canonical_id != e2.canonical_id:
                    pair = tuple(sorted([e1.canonical_id, e2.canonical_id]))
                    cooccurrence[pair[0]][pair[1]] += 1
    
    print(f"Unique entities: {len(entity_counts)}")
    
    # Get entity info
    entity_info = {}
    for char in extractor.resolver.db.characters.values():
        entity_info[char.id] = {'name': char.canonical_name, 'type': 'character'}
    for place in extractor.resolver.db.places.values():
        entity_info[place.id] = {'name': place.canonical_name, 'type': 'place'}
    for obj in extractor.resolver.db.objects.values():
        entity_info[obj.id] = {'name': obj.canonical_name, 'type': 'object'}
    
    # Build graph
    G = nx.Graph()
    
    # Add top entities as nodes
    top_entities = sorted(entity_counts.items(), key=lambda x: -x[1])[:40]
    
    for eid, count in top_entities:
        info = entity_info.get(eid, {'name': eid, 'type': 'unknown'})
        G.add_node(eid, name=info['name'], type=info['type'], mentions=count)
    
    # Add edges for co-occurrences
    min_cooccur = 3  # Minimum co-occurrences to create edge
    
    for e1, connections in cooccurrence.items():
        for e2, count in connections.items():
            if count >= min_cooccur and e1 in G.nodes and e2 in G.nodes:
                G.add_edge(e1, e2, weight=count)
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create visualization
    plt.figure(figsize=(20, 16), facecolor='#1a1a2e')
    ax = plt.gca()
    ax.set_facecolor('#1a1a2e')
    
    # Layout - use kamada_kawai for better distribution
    pos = nx.kamada_kawai_layout(G)
    
    # Node colors by type
    colors = []
    for node in G.nodes():
        ntype = G.nodes[node].get('type', 'unknown')
        if ntype == 'character':
            colors.append('#4CAF50')  # Green
        elif ntype == 'place':
            colors.append('#2196F3')  # Blue
        elif ntype == 'object':
            colors.append('#FF9800')  # Orange
        else:
            colors.append('#9E9E9E')
    
    # Node sizes based on mentions
    sizes = []
    for node in G.nodes():
        mentions = G.nodes[node].get('mentions', 1)
        sizes.append(min(3000, 200 + mentions * 5))
    
    # Edge widths based on co-occurrence
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + (w / max_weight) * 4 for w in edge_weights]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='#ffffff', 
                          width=edge_widths)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, 
                          alpha=0.9, edgecolors='white', linewidths=1)
    
    # Labels
    labels = {node: G.nodes[node].get('name', node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='white',
                           font_weight='bold')
    
    # Legend
    legend_elements = [
        plt.scatter([], [], c='#4CAF50', s=200, label='Character'),
        plt.scatter([], [], c='#2196F3', s=200, label='Place'),
        plt.scatter([], [], c='#FF9800', s=200, label='Object'),
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12,
               facecolor='#2a2a4e', edgecolor='white', labelcolor='white')
    
    plt.title("The Hobbit - Entity Co-occurrence Network\\n(entities appearing together in passages)", 
             fontsize=18, color='white', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save
    output_path = Path("data/exports/hobbit_cooccurrence.png")
    plt.savefig(output_path, dpi=150, facecolor='#1a1a2e', 
               bbox_inches='tight', pad_inches=0.5)
    plt.close()
    
    print(f"\\nSaved to: {output_path.absolute()}")
    
    # Print top connections
    print("\\nTop character connections:")
    for node in sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:10]:
        info = G.nodes[node]
        if info.get('type') == 'character':
            neighbors = list(G.neighbors(node))
            neighbor_names = [G.nodes[n].get('name', n) for n in neighbors[:5]]
            print(f"  {info['name']}: connected to {', '.join(neighbor_names)}")


if __name__ == "__main__":
    main()
