# Bruno Iochins Grisci
# September 2024

from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from pyvis.network import Network

from color_scheme import subcontinent_colors

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

        # Convert NetworkX graph to a pyvis network
        #pyvis_graph = Network(height='100%', width='100%', notebook=False, directed=True)
        pyvis_graph = Network(height='1500px', width='100%', notebook=False, directed=False)

        # Count occurrences of each country (node) in the affiliation data
        node_occurrences = df['affiliation_country'].str.split(';').explode().str.strip().value_counts()
        print(node_occurrences)

        # Add nodes and edges to the pyvis network
        for node in G.nodes:
            continent = country_to_region.get(node, country_to_region.get(node, [node, node]))  # Default to 'Other' if not found
            # Check the value of the continent and set the color accordingly
            color = get_color(continent[1])
                   
            # Set the node size based on its occurrence in the data
            node_size = int(node_occurrences.get(node, 1))  # Adjust multiplier for visibility

            pyvis_graph.add_node(node, size=node_size, color=color)

        for edge in G.edges(data=True):
            if edge[2]['weight'] > 0:  # Adjust threshold as needed
                pyvis_graph.add_edge(edge[0], edge[1], value=edge[2]['weight'])

        # Set edge width and physics settings for better visualization
        pyvis_graph.set_edge_smooth('continuous')
        #pyvis_graph.show_buttons(filter_=['physics'])
        #pyvis_graph.show_buttons()
        pyvis_graph.set_options('''
        var options = {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -2000,
              "centralGravity": 0.3,
              "springLength": 150
            },
            "maxVelocity": 1,
            "minVelocity": 0.1,
            "solver": "barnesHut",
            "timestep": 0.5,
            "adaptiveTimestep": true
          }
        }
        ''')

        # Save the interactive visualization to an HTML file
        pyvis_graph.save_graph(output_dir + csv_final_path.replace('.csv', '_authors.html'))

if __name__ == '__main__':
    main()
