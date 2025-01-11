import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from pyvis.network import Network

def replace_labels(df, prev, new):
    df['affiliation_country'] = df['affiliation_country'].str.replace(prev, new)
    df['Countries/regions/none'] = df['Countries/regions/none'].str.replace(prev, new)
    return df

def main():
    # Path to your CSV file
    csv_file_path = '21052024_SD1_Qual_Gals_Final.csv'

    # Columns to keep
    columns_to_keep = ['affiliation_country', 'SD-related (YES/NO)', 'Countries/regions/none']

    # Read the CSV file and keep only the specified columns
    df = pd.read_csv(csv_file_path, usecols=columns_to_keep)

    # Drop rows where "SD-related (YES/NO)" is "no"
    df = df[df['SD-related (YES/NO)'].str.lower() == 'yes']

    # Drop rows where "Countries/regions/none" is "none"
    df = df[df['Countries/regions/none'].str.lower() != 'none']

    # Drop rows with NaN values in any column
    df = df.dropna()

    # Replace labels
    replacements = {
        "Russian Federation": "Russia",
        "Soviet Union": "USSR",
        "Soivet Union": "USSR",
        "United States": "USA",
        "Uinted States": "USA",
        "Untied States": "USA",
        "Czech Republic": "Czechia",
        "Great United Kingdom": "UK",
        "United Kingdom": "UK",
        "Britain": "UK",
        "United Arab Emirates": "UAE",
        "Pkistan": "Pakistan",
        "Easter Europe": "Eastern Europe",
        "Antartic": "Antarctica",
        "Antarctic": "Antarctica",
        "Antarcticaa": "Antarctica",
        "Artic": "Arctic",
        "Global South countries": "Global South",
        "Carribean": "Caribbean",
        "El Salcador": "El Salvador",
        "Gemany": "Germany",
        "Great UK": "UK",
        "Africa South of Sahara": "Sub-Saharan Africa",
        "Sub-Sahara Africa": "Sub-Saharan Africa"
    }
    for prev, new in replacements.items():
        df = replace_labels(df, prev, new)

    # Display the resulting DataFrame
    print(df)

    # Initialize an empty list to hold the edges
    edges = []

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        countries = row['affiliation_country'].replace(',', ';').split(';')
        regions = row['Countries/regions/none'].replace(',', ';').split(';')

        # Create tuples for each combination of country and region
        for country in countries:
            for region in regions:
                if country.strip() and region.strip():
                    edges.append((country.strip(), region.strip()))

    # Display the list of edges
    print(edges)

    # Count the occurrences of each edge
    edge_counts = Counter(edges)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph with the edge weight based on the count
    for edge, count in edge_counts.items():
        G.add_edge(edge[0], edge[1], weight=count)

    print(edge_counts)

    # Convert NetworkX graph to a pyvis network
    pyvis_graph = Network(height='1500px', width='100%', notebook=False, directed=True)

    # Add nodes and edges to the pyvis network
    for node in G.nodes:
        pyvis_graph.add_node(node)

    for edge in G.edges(data=True):
        if edge[2]['weight'] > 0:  # Adjust threshold as needed
            pyvis_graph.add_edge(edge[0], edge[1], value=edge[2]['weight'])

    # Set edge width and physics settings for better visualization
    pyvis_graph.set_edge_smooth('continuous')

    # Configure physics settings for improved layout
    pyvis_graph.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=150)

    # Set options for the visualization
    pyvis_graph.set_options('''
    var options = {
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.05,
          "springLength": 150
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "timestep": 0.5
      },
      "nodes": {
        "scaling": {
          "min": 10,
          "max": 50
        }
      },
      "edges": {
        "scaling": {
          "min": 1,
          "max": 10
        },
        "color": {
          "inherit": true
        },
        "smooth": true
      }
    }
    ''')

    # Save the interactive visualization to an HTML file
    pyvis_graph.show("network.html", notebook=False)

if __name__ == '__main__':
    main()
