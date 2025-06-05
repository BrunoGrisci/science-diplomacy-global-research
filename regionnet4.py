# Bruno Iochins Grisci
# September 2024

from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import warnings
import csv

from color_scheme import subcontinent_colors

import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout

import numpy as np

def plot_graph_pdf(G, output_path, node_size_map, color_map):
    plt.figure(figsize=(20, 20))
    ax = plt.gca()

    for node in G.nodes():
        G.nodes[node]['weight'] = 300*node_size_map.get(node, 300)
        G.nodes[node]['size'] = 300*node_size_map.get(node, 300)

    '''
    # Convert to AGraph
    A = to_agraph(G)

    # Graph-level attributes for overlap control
    A.graph_attr.update({
        'layout': 'sfdp',
        'overlap': 'prism',
        'sep': '+20',
        'splines': 'true'
    })

    # Node-level attributes
    for n in G.nodes():
        size = G.nodes[n].get('size', 300)
        radius = (size ** 0.5) / 72  # convert pts² → radius in inches
        diameter = 2 * radius
        node = A.get_node(n)
        node.attr['width'] = f"{diameter:.4f}"
        node.attr['height'] = f"{diameter:.4f}"
        node.attr['fixedsize'] = 'true'
        node.attr['shape'] = 'circle'

    # Layout and position extraction
    A.layout(prog='sfdp')
    pos = graphviz_layout(G, prog='sfdp')    
    '''

    # Split connected components
    components = list(nx.connected_components(G.to_undirected()))
    components = sorted(components, key=len, reverse=True)

    pos = {}

    # Layout for main component
    main_component = components[0]
    subgraph = G.subgraph(main_component)

    #try:
    main_pos = graphviz_layout(subgraph, prog='twopi')
    #main_pos = {k: (v[0]*3000, v[1]*3000) for k, v in main_pos.items()}
    #except:
    #    main_pos = nx.spring_layout(subgraph, k=0.5)

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
    edges_no_selfloops = [(u, v) for u, v in G.edges() if u != v ]
    edge_weights = [G[e[0]][e[1]].get('weight', 1) for e in edges_no_selfloops]
    edge_widths = [0.5*w for w in edge_weights]  # Adjust multiplier as needed
    edge_colors = [color_map.get(e[0], "#1f78b4") for e in edges_no_selfloops]

    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges_no_selfloops,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.5,
        arrows=True,
        arrowsize=10,
        connectionstyle='arc3,rad=0.2',
    )


    # =================
    # Draw Nodes
    # =================
    nx.draw_networkx_nodes(
        G, pos,
        node_size=[300*node_size_map.get(node, 300) for node in G.nodes()],
        node_color=[color_map.get(node, "#1f78b4") for node in G.nodes()],
        edgecolors='black',
        linewidths=[0.5*G[node][node].get('weight', 1) if G.has_edge(node, node) else 0 for node in G.nodes()],
        #linewidths=0,
        alpha=0.95
    )

    '''
    # =================
    # Draw Self-Loops Manually
    # =================
    for node in G.nodes():
        if G.has_edge(node, node):
            x, y = pos[node]
            loop_size = 0.005  # Adjust as needed
            #loop_size = 1
            circle = plt.Circle((x, y), radius=loop_size,
                                color=color_map.get(node, "#1f78b4"), fill=False, linewidth=G[node][node].get('weight', 1), alpha=0.6)
            ax.add_artist(circle)

    # =================
    # Draw Labels Below Nodes
    # =================
    label_pos = {k: (v[0], v[1] - 10) for k, v in pos.items()}  # Y-offset for labels

    nx.draw_networkx_labels(
        G, label_pos,
        font_size=12,
        font_family='Arial'
    )
    '''

    # Draw nodes manually
    # Convert node attributes
    node_sizes = [300*node_size_map.get(node, 300) for node in G.nodes()]
    line_widths = [0.5*G[node][node].get('weight', 1) if G.has_edge(node, node) else 0 for node in G.nodes()]   
    for node, size, lw in zip(G.nodes(), node_sizes, line_widths):
        x, y = pos[node]
        
        # Convert node size (pts²) to radius in display coordinates
        radius_pts = (size ** 0.5) / 2  # approximate radius in points
        total_radius_pts = radius_pts + lw  # account for line width

        # Convert point offset to data units
        inv = ax.transData.inverted()
        display_coord = ax.transData.transform((x, y))
        label_coord = (display_coord[0], display_coord[1] - total_radius_pts - 5)  # 5pt padding
        label_x, label_y = inv.transform(label_coord)

        # Draw label just below
        ax.text(label_x, label_y, node, ha='center', va='top', fontsize=12, font='Arial', zorder=3)



    # =================
    # Final Plot Settings
    # =================
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()

def first_non_none_elements(list_of_lists):
    if not list_of_lists:
        return []

    # Determine the length of the sublists
    length_of_sublists = len(list_of_lists[0])
    result = []

    for i in range(length_of_sublists):
        for sublist in list_of_lists:
            if sublist[i].lower().rstrip() != "none":
                result.append(sublist[i])
                break
        else:
            result.append("None")  # If all elements at this index are None, append None

    return result

def all_elements_same_length(lst):
    if not lst:  # Check if the list is empty
        return True
    first_length = len(lst[0])
    for item in lst:
        if len(item) != first_length:
            return False
    return True

def get_color(subcontinent):
    if subcontinent in subcontinent_colors:
        return subcontinent_colors[subcontinent]
    else:
        return "#000000"  # Black for ERROR        

def main():
    # Path to your CSV file
    output_dir = "output/regions/"
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

        print(df)

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

        print(df)

        valid_nodes_csv = '22072024_Countries aggregated_BrunoGrisci-3007 - validnodes.csv'
        valid_nodes = pd.read_csv(valid_nodes_csv)
        valid_nodes.replace(r'\s*(.*?)\s*', r'\1', regex=True) 

        region_dictionary_csv = '22072024_Countries aggregated_BrunoGrisci-3007 - UN Geoscheme.csv'
        region_dictionary = pd.read_csv(region_dictionary_csv, usecols=['Country or Area', 'Region Name', 'Sub-region Name'])
        print(region_dictionary)
        country_to_region = region_dictionary.set_index('Country or Area').T.to_dict('list')
        subregion_to_region = region_dictionary.set_index('Sub-region Name').T.to_dict('list')
        print(country_to_region)
        print(subregion_to_region)

        # Initialize an empty list to hold the edges
        edges = []

        # Iterate over each row in the DataFrame
        for _, row in df.iterrows():
            source_countries = row['affiliation_country'].replace(',',';').split(';')
            source_subregion = row['affiliation_subregion_manual'].replace(',',';').split(';')
            source_region    = row['affiliation_region_manual'].replace(',',';').split(';')
            target_countries = row['target_country_manual'].replace(',',';').split(';')
            target_subregion = row['target_subregion_manual'].replace(',',';').split(';')
            target_region    = row['target_region_manual'].replace(',',';').split(';')
            #target_other     = row['target_other_manual'].replace(',',';').split(';')

            source_countries = list(map(str.strip, source_countries))
            source_subregion = list(map(str.strip, source_subregion))
            source_region = list(map(str.strip, source_region))
            target_countries = list(map(str.strip, target_countries))
            target_subregion = list(map(str.strip, target_subregion))
            target_region = list(map(str.strip, target_region))
            #target_other = list(map(str.strip, target_other))

            if not all_elements_same_length([source_countries, source_subregion, source_region]):
                print(source_countries, len(source_countries))
                print(source_subregion, len(source_subregion))
                print(source_region, len(source_region))
                raise Exception("Source lists above have different sizes.") 

            if not all_elements_same_length([target_countries, target_subregion, target_region]):
                print(target_countries, len(target_countries))
                print(target_subregion, len(target_subregion))
                print(target_region, len(target_region))
                #print(target_other, len(target_other))
                raise Exception("Target lists above have different sizes.") 

            source_list = first_non_none_elements([source_countries, source_subregion, source_region])
            target_list = first_non_none_elements([target_countries, target_subregion, target_region])

            for from_region in source_list:
                for to_region in target_list:
                    edges.append((from_region, to_region)) 

        # Count the occurrences of each edge
        edge_counts = Counter(edges)

        # Print the result in a beautiful way
        print("Counter results:")
        for item, count in edge_counts.items():
            print(f"{item}: {count}")

        # Save the result to a .csv file
        counter_file = output_dir + csv_final_path.replace('.csv', '_counter_results.csv')
        with open(counter_file, 'w', newline='') as csvfile:
            fieldnames = ['Item', 'Count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for item, count in edge_counts.items():
                writer.writerow({'Item': item, 'Count': count})

        print("\nResults saved to " + counter_file)

        # Create a directed graph
        G = nx.DiGraph()

        # Add edges to the graph with the edge weight based on the count
        for edge, count in edge_counts.items():
            G.add_edge(edge[0], edge[1], weight=count)
            
            if not valid_nodes['NODES'].str.contains(edge[0]).any():
                warnings.warn("WARNING: {} not in valid nodes!".format(edge[0]))

            if not valid_nodes['NODES'].str.contains(edge[1]).any():
                warnings.warn("WARNING: {} not in valid nodes!".format(edge[1]))   

        # Calculate centrality measures
        in_degree_centrality = nx.in_degree_centrality(G)
        out_degree_centrality = nx.out_degree_centrality(G)    
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G)

        # Create a DataFrame with the centrality measures
        centrality_df = pd.DataFrame({
            'Node': list(G.nodes),
            'In-Degree Centrality': [in_degree_centrality[node] for node in G.nodes],
            'Out-Degree Centrality': [out_degree_centrality[node] for node in G.nodes],
            'Degree Centrality': [degree_centrality[node] for node in G.nodes],
            'Betweenness Centrality': [betweenness_centrality[node] for node in G.nodes],
            'Closeness Centrality': [closeness_centrality[node] for node in G.nodes],
            'Eigenvector Centrality': [eigenvector_centrality[node] for node in G.nodes]
        })

        # Save the centrality measures table to a CSV file
        centrality_df.to_csv(output_dir + csv_final_path.replace('.csv', '_centrality.csv'), index=False)

        # Display the centrality measures table
        print(centrality_df)

        # Count occurrences of each country (node) in the affiliation data
        node_occurrences = df['affiliation_country'].str.split(';').explode().str.strip().value_counts()
        print(node_occurrences)


        # Create node size mapping based on node occurrences
        node_size_dict = {node: node_occurrences.get(node, 1) for node in G.nodes()}  # Adjust multiplier

        # Create color mapping
        node_color_dict = {}
        for node in G.nodes:
            continent = country_to_region.get(node, country_to_region.get(node, [node, node]))
            color = get_color(continent[1])
            node_color_dict[node] = color

        # Plot and save as PDF
        plot_graph_pdf(
            G,
            output_path=output_dir + csv_final_path.replace('.csv', '_region_network.pdf'),
            node_size_map=node_size_dict,
            color_map=node_color_dict
        )        

if __name__ == '__main__':
    main()
