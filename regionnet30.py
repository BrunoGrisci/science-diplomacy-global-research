# Bruno Iochins Grisci
# September 2024

from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from pyvis.network import Network
import warnings
import csv

from color_scheme import subcontinent_colors

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

        # Convert NetworkX graph to a pyvis network
        #pyvis_graph = Network(height='100%', width='100%', notebook=False, directed=True)
        pyvis_graph = Network(height='1500px', width='100%', notebook=False, directed=True)

        # Add nodes and edges to the pyvis network
        for node in G.nodes:
            continent = country_to_region.get(node, None)  # Default to 'Other' if not found
            if not continent:
                continent = subregion_to_region.get(node, [node, node])
            # Check the value of the continent and set the color accordingly
            color = get_color(continent[1])
           
            # Set the node size based on its in-degree
            node_size = in_degree_centrality.get(node, 1) * 100  # Adjust the multiplier as needed for visibility
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
        #pyvis_graph.show(output_dir + csv_final_path.replace('.csv', '.html'), notebook=False)
        pyvis_graph.save_graph(output_dir + csv_final_path.replace('.csv', '.html'))

if __name__ == '__main__':
    main()
