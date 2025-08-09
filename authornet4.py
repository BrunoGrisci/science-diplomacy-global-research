# Bruno Iochins Grisci
# September 2024

from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from networkx.drawing.nx_agraph import graphviz_layout

from color_scheme import subcontinent_colors

from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np

def plot_graph_pdf(G, output_path, node_size_map, color_map):
   



    plt.figure(figsize=(20, 20))
    ax = plt.gca()


    # Split connected components
    components = list(nx.connected_components(G.to_undirected()))
    components = sorted(components, key=len, reverse=True)

    pos = {}

    # Layout for main component
    main_component = components[0]
    subgraph = G.subgraph(main_component)

    try:
        main_pos = graphviz_layout(subgraph, prog='sfdp')
    except:
        main_pos = nx.spring_layout(subgraph, k=0.5)

    pos.update(main_pos)

    # Calculate bounding box of the main component
    x_vals = [p[0] for p in main_pos.values()]
    y_vals = [p[1] for p in main_pos.values()]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    x_range = x_max - x_min
    y_range = y_max - y_min

    # =====================
    # Place isolated nodes (single-node components) below
    # =====================

    # Flatten all components except the main one
    isolated_nodes = [node for comp in components[1:] if len(comp) == 1 for node in comp]

    # X positioning: spread out below main
    spacing = x_range * 0.1 if x_range > 0 else 1  # fallback spacing if graph is tiny
    start_x = x_min
    fixed_y = y_min - y_range * 0.2  # below the main component

    for idx, node in enumerate(isolated_nodes):
        x = start_x + idx * spacing
        y = fixed_y
        pos[node] = (x, y)

    # =====================
    # Place small multi-node components (optional)
    # =====================
    small_components = [comp for comp in components[1:] if len(comp) > 1]

    current_x = start_x + len(isolated_nodes) * spacing + spacing  # after isolated nodes

    for comp in small_components:
        grid_size = int(np.ceil(np.sqrt(len(comp))))

        for idx, node in enumerate(comp):
            col = idx % grid_size
            row = idx // grid_size

            x = current_x + col * spacing
            y = fixed_y - row * spacing  # can stack further down if needed

            pos[node] = (x, y)

        current_x += (grid_size + 1) * spacing  # move right for next component

    # =================
    # Draw Regular Edges (no self-loops)
    # =================
    edges_no_selfloops = [(u, v) for u, v in G.edges() if u != v]
    edge_weights = [G[e[0]][e[1]].get('weight', 1) for e in edges_no_selfloops]
    edge_widths = [0.2 * w for w in edge_weights]  # Adjust multiplier as needed
    edge_colors = [color_map.get(e[0], "#1f78b4") for e in edges_no_selfloops]

    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges_no_selfloops,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.5
    )

    # =================
    # Draw Self-Loops Manually
    # =================
    for node in G.nodes():
        if G.has_edge(node, node):
            x, y = pos[node]
            #loop_size = node_size_map.get(node, 300)*0.0001  # Adjust as needed
            loop_size = 0.05
            circle = plt.Circle((x, y), radius=loop_size,
                                color=color_map.get(node, "#1f78b4"), fill=False, linewidth=0.2*G[node][node].get('weight', 1), alpha=0.6)
            ax.add_artist(circle)

    # =================
    # Draw Nodes
    # =================
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[3*node_size_map.get(node, 300) for node in G.nodes()],
        node_color=[color_map.get(node, "#1f78b4") for node in G.nodes()],
        edgecolors='black',
        linewidths=0.5,
        alpha=0.95
    )

    # =================
    # Draw Labels Below Nodes
    # =================
    label_pos = {k: (v[0], v[1] - 0.05) for k, v in pos.items()}  # Y-offset for labels

    nx.draw_networkx_labels(
        G, label_pos,
        font_size=12,
        font_family='Arial'
    )

    # =================
    # Final Plot Settings
    # =================
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()


def get_color(subcontinent):
    if subcontinent in subcontinent_colors:
        return subcontinent_colors[subcontinent]
    else:
        return "#000000"  # Black for ERROR  

def main():
    # Path to your CSV file
    output_dir = "output/authors/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_file_path = 'akh_v5_1209_final_worksheet.csv'
    
    period_years_list = [(2002, 2023), (2002, 2008), (2009, 2012), (2013, 2019), (2020, 2023)]
    for period_years in period_years_list:

        csv_final_path = csv_file_path.replace('.csv', '_{}_{}.csv'.format(period_years[0], period_years[1]))
        print(csv_final_path)

        # Columns to keep
        columns_to_keep = ['affiliation_country', 'affiliation_subregion_manual', 'affiliation_region_manual', 'target_country_manual', 'target_subregion_manual', 'target_region_manual', 'affliation_north_or_south', 'SD-related (YES/NO)', 'coverDate']

        # Read the CSV file and keep only the specified columns
        df = pd.read_csv(csv_file_path, usecols=columns_to_keep)

        # Create a new column "Year" and save the year from "coverDate"
        df['Year'] = pd.to_datetime(df['coverDate']).dt.year

        df.replace(r'\s*(.*?)\s*', r'\1', regex=True) 

        # Drop rows where "SD-related (YES/NO)" is "no"
        df = df[df['SD-related (YES/NO)'].str.lower() == 'yes']

        # Drop rows where 'affiliation_country' is "None"
        df = df[df['affiliation_country'].str.lower() != 'none']

        # Drop rows where 'target_region_manual' is "None"
        df = df[df['target_region_manual'].str.lower() != 'none']

        # Drop rows where 'Year' is out of bounds
        df = df[df['Year'] >= period_years[0]]
        df = df[df['Year'] <= period_years[1]]

        # Drop rows with NaN values in any column
        df = df.dropna()

        region_dictionary_csv = '22072024_Countries aggregated_BrunoGrisci-3007 - UN Geoscheme.csv'
        region_dictionary = pd.read_csv(region_dictionary_csv, usecols=['Country or Area', 'Region Name', 'Sub-region Name'])
        region_dictionary.replace(r'\s*(.*?)\s*', r'\1', regex=True) 
        print(region_dictionary)
        country_to_region = region_dictionary.set_index('Country or Area').T.to_dict('list')
        subregion_to_region = region_dictionary.set_index('Sub-region Name').T.to_dict('list')

        # Initialize an empty list to hold the edges
        edges = []

        # Iterate over each row in the DataFrame
        for _, row in df.iterrows():
            source_countries = row['affiliation_country'].replace(',',';').split(';')
            source_countries = list(map(str.strip, source_countries))    
            if len(source_countries) > 1: 
                edges = edges + list(combinations(source_countries, 2))
            else:
                edges = edges + [(source_countries[0], source_countries[0])]


        # Count the occurrences of each edge
        edge_counts = Counter(edges)

        # Create a directed graph
        G = nx.Graph()

        # Add edges to the graph with the edge weight based on the count
        for edge, count in edge_counts.items():
            G.add_edge(edge[0], edge[1], weight=count)

        print(edge_counts)

        # Calculate centrality measures  
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G)

        # Create a DataFrame with the centrality measures
        centrality_df = pd.DataFrame({
            'Node': list(G.nodes),
            'Degree Centrality': [degree_centrality[node] for node in G.nodes],
            'Betweenness Centrality': [betweenness_centrality[node] for node in G.nodes],
            'Closeness Centrality': [closeness_centrality[node] for node in G.nodes],
            'Eigenvector Centrality': [eigenvector_centrality[node] for node in G.nodes]
        })

        # Save the centrality measures table to a CSV file
        centrality_df.to_csv(output_dir + csv_final_path.replace('.csv', '_authors_centrality.csv'), index=False)

        # Display the centrality measures table
        print(centrality_df)



        # Count occurrences of each country (node) in the affiliation data
        node_occurrences = df['affiliation_country'].str.split(';').explode().str.strip().value_counts()
        print(node_occurrences)


        # Create node size mapping based on node occurrences
        node_size_dict = {node: node_occurrences.get(node, 1) * 10 for node in G.nodes()}  # Adjust multiplier

        # Create color mapping
        node_color_dict = {}
        for node in G.nodes:
            continent = country_to_region.get(node, country_to_region.get(node, [node, node]))
            color = get_color(continent[1])
            node_color_dict[node] = color

        # Plot and save as PDF
        plot_graph_pdf(
            G,
            output_path=output_dir + csv_final_path.replace('.csv', '_authors_network.pdf'),
            node_size_map=node_size_dict,
            color_map=node_color_dict
        )        


if __name__ == '__main__':
    main()
