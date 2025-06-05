import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import os

def normalize(value, min_val, max_val, out_min, out_max):
    """Generic normalization function."""
    if max_val == min_val:
        return (out_min + out_max) / 2
    return out_min + (value - min_val) * (out_max - out_min) / (max_val - min_val)

def gexf_to_pdf(gexf_path, output_pdf='network_figure.pdf', layout_prog='sfdp'):
    G = nx.read_gexf(gexf_path)

    #if not G.is_directed():
    #    G = G.to_directed()

    A = to_agraph(G)

    # Gather weight stats for normalization
    node_weights = [float(G.nodes[n].get('weight', 0)) for n in G.nodes]
    edge_weights = [float(G[u][v].get('weight', 1.0)) for u, v in G.edges]

    min_node_w, max_node_w = min(node_weights), max(node_weights)
    min_edge_w, max_edge_w = min(edge_weights), max(edge_weights)

    for n in A.nodes():
        node_id = n.get_name()
        data = G.nodes[node_id]
        label = data.get('label', node_id)
        color = data.get('color', '#cccccc')
        weight = float(data.get('weight', 0.1))
        size = normalize(weight, min_node_w, max_node_w, 0.5, 3)  # in inches
        #size = weight*10.0

        n.attr.update({
            'label': label,         # Show label inside the node
            'fontsize': 12,
            'fontname': 'Arial',    # Change font to Arial
            'fillcolor': color,
            'color': color,
            'style': 'filled',
            'width': str(size),
            'height': str(size),
            'shape': 'circle',
            'fixedsize': 'true'
        })


    for e in A.edges():
        src, tgt = e[0], e[1]
        data = G[src][tgt]
        color = data.get('color', '#888888')
        weight = float(data.get('weight', 1.0))
        width = normalize(weight, min_edge_w, max_edge_w, 1, 15.0)

        e.attr.update({
            'color': color,
            'penwidth': str(width),
            'arrowhead': 'normal'
        })

    A.graph_attr.update({
        'splines': True,        
        'outputorder': 'edgesfirst',
        'dpi': 600,        
        'overlap': False,
        'sep': 10
    })

    A.layout(prog=layout_prog)
    A.draw(output_pdf, format='png')

    return os.path.abspath(output_pdf)

def main():
# Example usage
 #gexf_to_pdf("output/authors/akh_v5_1209_final_worksheet_2002_2023.gexf", "output.pdf")
 gexf_to_pdf("output/authors/akh_v5_1209_final_worksheet_2002_2023.gexf", "authors.png")
 gexf_to_pdf("output/regions/akh_v5_1209_final_worksheet_2002_2023.gexf", "regions.png")

if __name__ == '__main__':
    main()

