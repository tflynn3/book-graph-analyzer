"""Build a full graph visualization combining entity + relationship data."""

import json
from pathlib import Path
from collections import defaultdict
from pyvis.network import Network
import networkx as nx

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    print("Building full Hobbit graph...")
    
    # Load the relationship data (from seeded extraction)
    rel_file = Path("data/exports/hobbit_relationships.json")
    if rel_file.exists():
        with open(rel_file) as f:
            rel_data = json.load(f)
        print(f"Loaded {len(rel_data.get('relationships', []))} relationship records")
    else:
        rel_data = {"relationships": []}
        print("No relationship file found")
    
    # Load seed data for canonical names
    seed_chars = Path("data/seeds/characters.json")
    seed_places = Path("data/seeds/places.json")
    seed_objects = Path("data/seeds/objects.json")
    
    entity_info = {}
    
    if seed_chars.exists():
        with open(seed_chars) as f:
            for char in json.load(f):
                entity_info[char['id']] = {
                    'name': char['canonical_name'],
                    'type': 'character',
                    'aliases': char.get('aliases', []),
                }
    
    if seed_places.exists():
        with open(seed_places) as f:
            for place in json.load(f):
                entity_info[place['id']] = {
                    'name': place['canonical_name'],
                    'type': 'place',
                    'aliases': place.get('aliases', []),
                }
    
    if seed_objects.exists():
        with open(seed_objects) as f:
            for obj in json.load(f):
                entity_info[obj['id']] = {
                    'name': obj['canonical_name'],
                    'type': 'object',
                    'aliases': obj.get('aliases', []),
                }
    
    print(f"Loaded {len(entity_info)} seed entities")
    
    # Build graph from relationships
    G = nx.DiGraph()
    
    # Track entity mentions
    entity_mentions = defaultdict(int)
    relationship_counts = defaultdict(lambda: defaultdict(int))
    
    # Process all relationship records
    for record in rel_data.get('relationships', []):
        for rel in record.get('relationships', []):
            subj = rel.get('subject_id')
            obj = rel.get('object_id')
            pred = rel.get('predicate')
            
            if subj and obj and pred:
                entity_mentions[subj] += 1
                entity_mentions[obj] += 1
                relationship_counts[(subj, obj)][pred] += 1
    
    # Also count from top_characters in the data
    for name, count in rel_data.get('top_characters', []):
        # Find matching entity
        for eid, info in entity_info.items():
            if info['name'] == name:
                entity_mentions[eid] = max(entity_mentions[eid], count)
                break
    
    # Add nodes for entities that have relationships or significant mentions
    significant_entities = set()
    for (subj, obj), preds in relationship_counts.items():
        significant_entities.add(subj)
        significant_entities.add(obj)
    
    # Also add top mentioned entities
    for eid, count in sorted(entity_mentions.items(), key=lambda x: -x[1])[:50]:
        significant_entities.add(eid)
    
    # Add key characters we know should be there
    key_characters = [
        'bilbo_baggins', 'gandalf', 'thorin_oakenshield', 'smaug', 'gollum',
        'balin', 'beorn', 'bard', 'elrond', 'thranduil', 'dain',
        'fili', 'kili', 'bombur', 'dwalin',
    ]
    for char in key_characters:
        if char in entity_info:
            significant_entities.add(char)
    
    # Key places
    key_places = [
        'the_shire', 'bag_end', 'rivendell', 'mirkwood', 'erebor', 
        'lake_town', 'dale', 'goblin_town', 'misty_mountains',
    ]
    for place in key_places:
        if place in entity_info:
            significant_entities.add(place)
    
    print(f"Adding {len(significant_entities)} significant entities")
    
    # Colors by type
    colors = {
        'character': '#4CAF50',  # Green
        'place': '#2196F3',       # Blue  
        'object': '#FF9800',      # Orange
    }
    
    # Add nodes
    for eid in significant_entities:
        info = entity_info.get(eid, {'name': eid, 'type': 'unknown'})
        mentions = entity_mentions.get(eid, 1)
        
        size = min(50, 15 + mentions // 10)
        color = colors.get(info['type'], '#9E9E9E')
        
        G.add_node(
            eid,
            label=info['name'],
            title=f"{info['name']}\nType: {info['type']}\nMentions: {mentions}",
            size=size,
            color=color,
        )
    
    # Add edges
    for (subj, obj), preds in relationship_counts.items():
        if subj in significant_entities and obj in significant_entities:
            # Use the most common predicate
            main_pred = max(preds.items(), key=lambda x: x[1])[0]
            total_count = sum(preds.values())
            
            G.add_edge(
                subj,
                obj,
                title=f"{main_pred} (x{total_count})",
                label=main_pred,
                width=min(5, 1 + total_count),
            )
    
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Create pyvis network
    net = Network(
        height='900px',
        width='100%',
        bgcolor='#1a1a2e',
        font_color='white',
        directed=True,
    )
    
    # Configure physics for better layout
    net.set_options("""
    {
        "nodes": {
            "font": {"size": 14, "face": "arial"}
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
            "color": {"inherit": true},
            "smooth": {"type": "continuous"}
        },
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -5000,
                "centralGravity": 0.5,
                "springLength": 150,
                "springConstant": 0.05
            },
            "minVelocity": 0.75
        }
    }
    """)
    
    # Add from networkx
    net.from_nx(G)
    
    # Save
    output_path = Path("data/exports/hobbit_full_graph.html")
    net.save_graph(str(output_path))
    
    print(f"\n✓ Visualization saved to: {output_path.absolute()}")
    print("\nOpen this file in your browser to explore!")
    
    # Print stats
    print(f"\nTop connected characters:")
    degrees = dict(G.degree())
    for node, degree in sorted(degrees.items(), key=lambda x: -x[1])[:15]:
        info = entity_info.get(node, {'name': node, 'type': '?'})
        print(f"  {info['name']}: {degree} connections")


if __name__ == "__main__":
    main()
