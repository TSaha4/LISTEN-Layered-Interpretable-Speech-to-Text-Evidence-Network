import argparse
import json
import webbrowser
from pathlib import Path
from pyvis.network import Network
import matplotlib.colors as mcolors

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def get_color_for_weight(weight):
    # Map weight (0-1) to a color (light blue to deep red)
    cmap = mcolors.LinearSegmentedColormap.from_list("importance", ["#E0F7FA", "#D32F2F"])
    return rgb_to_hex(cmap(weight)[:3])

def visualize(input_json, output_html):
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "evidence_graph" not in data:
        print("Error: 'evidence_graph' not found in input JSON.")
        return

    graph_data = data["evidence_graph"]
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    print(f"Visualizing {len(nodes)} nodes and {len(edges)} edges...")

    # Create pyvis network
    net = Network(height="800px", width="100%", bgcolor="#fcfcfc", font_color="black", directed=True)
    
    # Add nodes
    for node in nodes:
        node_id = node["id"]
        speaker = node.get("speaker", "UNKNOWN")
        text = node.get("text", "")
        weight = node.get("weight", 0.0)
        
        color = get_color_for_weight(weight)
        
        # Tooltip content (HTML)
        title = f"<b>{speaker}</b> (Weight: {weight:.3f})<br><br>{text}"
        
        # Label to show on graph
        label = f"{speaker}\n{node_id}"
        
        # Node size based on weight (min 15, max 45)
        size = 15 + (weight * 30)
        
        net.add_node(
            node_id,
            label=label,
            title=title,
            color=color,
            size=size
        )

    # Add edges
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        weight = edge.get("weight", 0.1)
        relation = edge.get("relation", "")
        
        # Determine edge styling based on relation
        edge_color = "#999999"
        if relation == "question_link":
            edge_color = "#1E88E5" # Blue
        elif relation.startswith("shared_entity"):
            edge_color = "#43A047" # Green
        elif relation == "semantic_similarity":
            edge_color = "#FB8C00" # Orange
            
        title = f"Relation: {relation}<br>Weight: {weight:.3f}"
        
        net.add_edge(
            source,
            target,
            title=title,
            color=edge_color,
            value=max(0.5, weight * 5)  # thickness
        )
        
    # Configure physics for better layout (helps nodes separate cleanly)
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)
    
    # Save and show
    net.save_graph(output_html)
    print(f"Graph saved to {output_html}")
    
    # Open in browser
    abs_path = Path(output_html).absolute()
    webbrowser.open(f"file://{abs_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Evidence Graph")
    parser.add_argument("--input", required=True, help="Path to pipeline output JSON")
    parser.add_argument("--output", default="evidence_graph.html", help="Path to save HTML graph")
    
    args = parser.parse_args()
    visualize(args.input, args.output)
