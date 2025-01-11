import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

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

    # Replace "Russian Federation" with "Russia"
    df = replace_labels(df, "Russian Federation", "Russia")
    df = replace_labels(df, "United States", "USA")
    df = replace_labels(df, "Czech Republic", "Czechia")
    df = replace_labels(df, "United Kingdom", "UK")
    df = replace_labels(df, "United Arab Emirates", "UAE")

    # Display the resulting DataFrame
    print(df)

    # Initialize an empty list to hold the edges
    edges = []

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        countries = row['affiliation_country'].split(';')
        regions = row['Countries/regions/none'].split(';')

        # Create tuples for each combination of country and region
        for country in countries:
            for region in regions:
                edges.append((country.strip(), region.strip()))

    # Display the list of edges
    print(edges)

    # Count the occurrences of each edge
    edge_counts = Counter(edges)

    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph with the edge weight based on the count
    for edge, count in edge_counts.items():
        if count > 1:  # Adjust threshold as needed
            G.add_edge(edge[0], edge[1], weight=count)

    print(edge_counts)

    # Draw the graph
    # Position nodes using a spring layout
    #pos = nx.spring_layout(G, k=0.3, iterations=50)  # Adjust k and iterations as needed
    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(120, 100))  # Set the figure size

    # Draw the edges with widths proportional to their counts
    edges = G.edges(data=True)
    edge_widths = [data['weight'] for _, _, data in edges]

    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=15, font_weight='bold', 
            edge_color=edge_widths, edge_cmap=plt.cm.Blues, width=edge_widths, arrowstyle='->', arrowsize=20)
    
    # Display the graph with adjusted margins
    plt.margins(0.1)
    plt.title('Directed Network Graph')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
