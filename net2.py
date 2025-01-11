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
    columns_to_keep = ['affiliation_country', 'SD-related (YES/NO)', 'Countries/regions/none', 'coverDate']

    # Read the CSV file and keep only the specified columns
    df = pd.read_csv(csv_file_path, usecols=columns_to_keep)

    # Create a new column "Year" and save the year from "coverDate"
    df['Year'] = pd.to_datetime(df['coverDate']).dt.year

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
        countries = row['affiliation_country'].replace(',',';').split(';')
        regions = row['Countries/regions/none'].replace(',',';').split(';')

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
    centrality_df.to_csv("centrality.csv", index=False)

    # Display the centrality measures table
    print(centrality_df)



    country_to_subcontinent = {
        # Global North
        'USA': 'Global North',
        'Canada': 'Global North',
        'Western European countries': 'Global North',
        'Australia': 'Global North',
        'New Zealand': 'Global North',
        'Japan': 'Global North',
        'South Korea': 'Global North',
        'Israel': 'Global North',
        'Singapore': 'Global North',
        'Switzerland': 'Global North',
        'Norway': 'Global North',
        'Sweden': 'Global North',
        'Denmark': 'Global North',
        'Finland': 'Global North',
        'Germany': 'Global North',
        'France': 'Global North',
        'Netherlands': 'Global North',
        'United Kingdom': 'Global North',
        'Ireland': 'Global North',
        'Austria': 'Global North',
        'Belgium': 'Global North',
        'Luxembourg': 'Global North',
        'Italy': 'Global North',
        'Spain': 'Global North',
        'Portugal': 'Global North',
        'Greece': 'Global North',
        'Cyprus': 'Global North',
        'Malta': 'Global North',
        'Iceland': 'Global North',
        'Greenland': 'Global North',
        'Alaska': 'Global North',
        'California': 'Global North',
        'East Asia': 'Global North',
        'Southeast Asia': 'Global North',
        'Central Europe': 'Global North',
        'Eastern Europe': 'Global North',
        'Southern Europe': 'Global North',
        'Western Europe': 'Global North',
        'Northern Europe': 'Global North',
        'North Sea nations': 'Global North',
        'Greenland': 'Global North',
        'Siberia': 'Global North',
        'Mongol Empire': 'Global North',
        'Arctic Ocean': 'Global North',
        'Gulf of California': 'Global North',
        'Alaska': 'Global North',
        'Developing countries': 'Global North',
        'least developing countries': 'Global North',
        'Eastern Europe': 'Global North',
        'Central Europe': 'Global North',
        'Western Europe': 'Global North',
        'Southern Europe': 'Global North',
        'Northern Europe': 'Global North',
        'North Sea nations': 'Global North',
        'Greenland': 'Global North',
        'Siberia': 'Global North',
        'Mongol Empire': 'Global North',
        'Arctic Ocean': 'Global North',
        'Gulf of California': 'Global North',
        'Alaska': 'Global North',
        'Developing countries': 'Global North',

        # Global South
        'Brazil': 'Global South',
        'China': 'Global South',
        'India': 'Global South',
        'South Africa': 'Global South',
        'Russia': 'Global South',
        'Iran': 'Global South',
        'Pakistan': 'Global South',
        'Mexico': 'Global South',
        'Argentina': 'Global South',
        'Egypt': 'Global South',
        'Turkey': 'Global South',
        'Indonesia': 'Global South',
        'Thailand': 'Global South',
        'Vietnam': 'Global South',
        'Philippines': 'Global South',
        'Iran': 'Global South',
        'Saudi Arabia': 'Global South',
        'UAE': 'Global South',
        'Iraq': 'Global South',
        'Kuwait': 'Global South',
        'Qatar': 'Global South',
        'Egypt': 'Global South',
        'Kenya': 'Global South',
        'Morocco': 'Global South',
        'Guatemala': 'Global South',
        'Chile': 'Global South',
        'Colombia': 'Global South',
        'Venezuela': 'Global South',
        'Ethiopia': 'Global South',
        'Middle East': 'Global South',
        'Cuba': 'Global South',
        'Singapore': 'Global South',
        'Sri Lanka': 'Global South',
        'Nepal': 'Global South',
        'Vietnam': 'Global South',
        'Asia': 'Global South',
        'Bolivia': 'Global South',
        'South Asia': 'Global South',
        'Afghanistan': 'Global South',
        'Bangladesh': 'Global South',
        'Bhutan': 'Global South',
        'Maldives': 'Global South',
        'Myanmar': 'Global South',
        'North Macedonia': 'Global South',
        'SEE': 'Global South',
        'Balkans': 'Global South',
        'Americas': 'Global South',
        'Caribbean': 'Global South',
        'Indonesia': 'Global South',
        'Philippines': 'Global South',
        'Brunei': 'Global South',
        'Peru': 'Global South',
        'South America': 'Global South',
        'Central America': 'Global South',
        'Mauritius': 'Global South',
        'Antarctica': 'Global South',
        'Panama': 'Global South',
        'Slovakia': 'Global South',
        'Austria': 'Global South',
        'Lithuania': 'Global South',
        'Central America': 'Global South',
        'Ukraine': 'Global South',
        'Cyprus': 'Global South',
        'Israel': 'Global South',
        'Palestine': 'Global South',
        'Jordan': 'Global South',
        'Latvia': 'Global South',
        'Central Europe': 'Global South',
        'Eastern Europe': 'Global South',
        'Southern Europe': 'Global South',
        'Greenland': 'Global South',
        'East Asia': 'Global South',
        'Southeast Asia': 'Global South',
        'Slovenia': 'Global South',
        'Southern Mediterranean region': 'Global South',
        'Sub-Saharan Africa': 'Global South',
        'Western Europe': 'Global South',
        'Greece': 'Global South',
        'Taiwan': 'Global South',
        'South Asia': 'Global South',
        'North Africa': 'Global South',
        'MENA': 'Global South',
        'Algeria': 'Global South',
        'Tunisia': 'Global South',
        'Romania': 'Global South',
        'Bulgaria': 'Global South',
        'Yugoslavia': 'Global South',
        'Bosnia and Herzegovina': 'Global South',
        'Montenegro': 'Global South',
        'Serbia': 'Global South',
        'Kosovo': 'Global South',
        'Indo-Pacific': 'Global South',
        'Southern Ocean': 'Global South',
        'Iceland': 'Global South',
        'Arctic Ocean': 'Global South',
        'Bahrain': 'Global South',
        'Lebanon': 'Global South',
        'Libya': 'Global South',
        'Syria': 'Global South',
        'Malta': 'Global South',
        'Arab countries': 'Global South',
        'Sierra Leone': 'Global South',
        'North Africa': 'Global South',
        'MENA': 'Global South',
        'Algeria': 'Global South',
        'Tunisia': 'Global South',
        'Romania': 'Global South',
        'Bulgaria': 'Global South',
        'Yugoslavia': 'Global South',
        'Bosnia and Herzegovina': 'Global South',
        'Montenegro': 'Global South',
        'Serbia': 'Global South',
        'Kosovo': 'Global South'
    }






    # Convert NetworkX graph to a pyvis network
    #pyvis_graph = Network(height='100%', width='100%', notebook=False, directed=True)
    pyvis_graph = Network(height='1500px', width='100%', notebook=False, directed=True)

    # Add nodes and edges to the pyvis network
    for node in G.nodes:
        subcontinent = country_to_subcontinent.get(node, 'Other')  # Default to 'Other' if not found
        if subcontinent == 'Global North':
            color='#1f78b4' # Blue color for Global North
        elif subcontinent == 'Global South':
            color='#33a02c'  # Green color for Global South
        else:
            color='#a6cee3'  # Light blue for Other
       
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
    pyvis_graph.show("network.html", notebook=False)

if __name__ == '__main__':
    main()
