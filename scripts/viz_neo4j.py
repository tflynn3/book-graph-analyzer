"""Interactive visualization of the Neo4j knowledge graph.
Shows Character nodes connected through Events as edges with full context.
"""
from pathlib import Path
import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.chdir(Path(__file__).parent.parent)

from book_graph_analyzer.graph.connection import get_driver
from pyvis.network import Network

OUTPUT = Path("data/exports/full_graph.html")

COLORS = {
    "Character": "#4CAF50",
    "Place":     "#2196F3",
    "Object":    "#E91E63",
    "Event":     "#FF9800",
}

driver = get_driver()
net = Network(height="950px", width="100%", bgcolor="#1a1a2e",
              font_color="white", directed=True)
net.barnes_hut(gravity=-5000, central_gravity=0.3,
               spring_length=220, spring_strength=0.04, damping=0.12)
net.show_buttons(filter_=["physics"])

added_nodes = set()
edge_count  = 0

with driver.session() as s:

    # ── Top Characters (real nodes, by how many events reference them) ──────
    chars = s.run("""
        MATCH (e:Event) WHERE e.agent IS NOT NULL
        WITH e.agent as name, count(e) as cnt
        ORDER BY cnt DESC LIMIT 60
        RETURN name, cnt
    """)
    char_names = {}
    for row in chars:
        name = row["name"]
        if not name or name == "None": continue
        nid = f"char_{name}"
        if nid not in added_nodes:
            size = min(70, 15 + int(row["cnt"]) // 3)
            net.add_node(nid, label=name,
                         title=f"Character/Agent: {name}\nEvents as agent: {row['cnt']}",
                         color=COLORS["Character"], size=size, shape="dot")
            added_nodes.add(nid)
            char_names[name] = nid

    # ── Places (from event patients / descriptions) ──────────────────────────
    places = s.run("""
        MATCH (p:Place) WHERE p.name IS NOT NULL AND p.name <> 'None'
        RETURN p.name as name LIMIT 30
    """)
    place_ids = {}
    for row in places:
        name = row["name"]
        nid = f"place_{name}"
        if nid not in added_nodes:
            net.add_node(nid, label=name,
                         title=f"Place: {name}",
                         color=COLORS["Place"], size=18, shape="diamond")
            added_nodes.add(nid)
            place_ids[name] = nid

    # ── Objects ──────────────────────────────────────────────────────────────
    objs = s.run("""
        MATCH (o:Object) WHERE o.name IS NOT NULL AND o.name <> 'None'
        RETURN o.name as name LIMIT 15
    """)
    for row in objs:
        name = row["name"]
        nid = f"obj_{name}"
        if nid not in added_nodes:
            net.add_node(nid, label=name,
                         title=f"Object: {name}",
                         color=COLORS["Object"], size=14, shape="star")
            added_nodes.add(nid)

    # ── Character→Character edges via shared events ──────────────────────────
    # For every event that has BOTH an agent and a patient that is a known agent,
    # draw an edge from agent to patient with the action + context as the label
    co_events = s.run("""
        MATCH (e:Event)
        WHERE e.agent IS NOT NULL AND e.patient IS NOT NULL
          AND e.agent <> 'None' AND e.patient <> 'None'
        RETURN e.agent as agent, e.action as action,
               e.patient as patient, e.description as desc,
               e.era as era, e.year as year
        LIMIT 600
    """)
    for row in co_events:
        agent   = row["agent"]
        patient = row["patient"]
        action  = row["action"] or "related to"
        desc    = row["desc"] or ""
        era     = row["era"] or ""
        year    = f" ({row['year']})" if row["year"] else ""

        # Try to find patient in our known character set (partial match)
        patient_nid = None
        for cname, cnid in char_names.items():
            if patient.lower() in cname.lower() or cname.lower() in patient.lower():
                patient_nid = cnid
                break

        agent_nid = char_names.get(agent)

        # Add patient as a node if not already there
        if patient_nid is None:
            pnid = f"char_{patient}"
            if pnid not in added_nodes:
                net.add_node(pnid, label=patient,
                             title=f"Agent/Patient: {patient}",
                             color="#A5D6A7", size=10, shape="dot")
                added_nodes.add(pnid)
                char_names[patient] = pnid
            patient_nid = pnid

        if agent_nid and patient_nid and agent_nid != patient_nid:
            context = f"{era}{year}" if era else ""
            title   = f"{desc}\n{context}".strip()
            net.add_edge(agent_nid, patient_nid,
                         label=action[:20],
                         title=title,
                         color="#FF980099", width=1.5)
            edge_count += 1

    # ── Temporal event chain (top 150 most-connected events as mini-nodes) ──
    top_events = s.run("""
        MATCH (e:Event)-[r:BEFORE|CAUSED]->(f:Event)
        WHERE e.description IS NOT NULL AND f.description IS NOT NULL
        WITH e, f, type(r) as rel
        LIMIT 150
        RETURN e.id as eid, e.description as edesc, e.agent as eagent,
               f.id as fid, f.description as fdesc, f.agent as fagent,
               rel
    """)
    for row in top_events:
        for nid, desc, agent in [
            (f"evt_{row['eid']}", row["edesc"], row["eagent"]),
            (f"evt_{row['fid']}", row["fdesc"], row["fagent"]),
        ]:
            if nid not in added_nodes:
                label = (desc or "")[:35] + ("..." if len(desc or "") > 35 else "")
                net.add_node(nid, label=label, title=desc or "",
                             color=COLORS["Event"], size=8, shape="ellipse")
                added_nodes.add(nid)

        src, dst = f"evt_{row['eid']}", f"evt_{row['fid']}"
        color = "#ff6b6b" if row["rel"] == "CAUSED" else "#55555588"
        net.add_edge(src, dst, title=row["rel"], color=color, width=1)
        edge_count += 1

        # Pin events to their agent character nodes
        for nid, agent in [(f"evt_{row['eid']}", row["eagent"]),
                           (f"evt_{row['fid']}", row["fagent"])]:
            if agent and agent in char_names:
                net.add_edge(char_names[agent], nid,
                             color="#4CAF5044", width=0.8, title="agent of")
                edge_count += 1

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
net.save_graph(str(OUTPUT))
print(f"Done! {len(added_nodes)} nodes, {edge_count} edges -> {OUTPUT}")
