# Bruno Iochins Grisci
# September 2024

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter

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

def filter_data_by_year(df, year):
    """Filter the DataFrame to only include rows for a specific year."""
    return df[df['Year'] <= year]

def draw_network(G, dict_c2r, dict_s2r, pos, ax, year, title, color_map):
    """Draw the network for the given year."""
    ax.clear()
    ax.set_title(f"{title} - Year: {year}")

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[color_map.get(node, color_map.get(dict_c2r.get(node, node)[1], '#000000')) for node in G.nodes()],        
        node_size=[G.degree(node) * 100 for node in G.nodes()],
        ax=ax
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

def animate_network_growth(df, period_years, dict_c2r, dict_s2r, color_map, title):
    # Initialize a figure for plotting
    fig, ax = plt.subplots(figsize=(20, 20))

    # Create a directed graph
    G = nx.DiGraph()

    # Initialize positions (will be updated dynamically)
    pos = None

    def update(year):
        nonlocal pos  # Allow modification of pos across calls
        # Filter data for the current year
        current_df = filter_data_by_year(df, year)

        # Build edges from the filtered data
        edges = []
        # Iterate over each row in the DataFrame
        for _, row in current_df.iterrows():
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

        # Add edges progressively for the current year
        for edge, count in edge_counts.items():
            G.add_edge(edge[0], edge[1], weight=count)

        # Update node positions only if graph has nodes
        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, pos=pos, fixed=G.nodes()) if pos else nx.spring_layout(G)

        # Clear previous plot and draw updated network
        draw_network(G, dict_c2r, dict_s2r, pos, ax, year, title, color_map)

    # Create the animation using FuncAnimation
    ani = animation.FuncAnimation(fig, update, frames=range(period_years[0], period_years[1] + 1), repeat=False)

    # Save the animation as an MP4 using ffmpeg
    ani.save('network_growth_animation.mp4', writer='ffmpeg', fps=1)

    #plt.show()

# Example usage
if __name__ == '__main__':
    # Assume df is your DataFrame already processed with the necessary columns
    csv_file_path = 'v4_1209_final_worksheet.csv'
    period_years = [2002, 2023]

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
    # Drop rows with NaN values in any column
    df = df.dropna()

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

    # Define color map for regions
    color_map = {
        # Africa
        "Northern Africa": "#2E8B57",             # Sea Green
        "Sub-Saharan Africa": "#8FBC8F",          # Dark Sea Green
        "Africa": "#32CD32",                      # Lime Green
        
        # Americas
        "Latin America and the Caribbean": "#FF6347",  # Tomato
        "Northern America": "#FF4500",            # Orange Red
        "Americas": "#B22222",                    # Firebrick

        # Asia
        "Central Asia": "#FFD700",                # Gold
        "Eastern Asia": "#FFEC8B",                # Light Goldenrod Yellow
        "South-eastern Asia": "#FFA500",          # Orange
        "Southern Asia": "#FF8C00",               # Dark Orange
        "Western Asia": "#FFDAB9",                # Peach Puff
        "Asia": "#FFFFE0",                        # Light Yellow

        # Europe
        "Eastern Europe": "#1E90FF",              # Dodger Blue
        "Northern Europe": "#00BFFF",             # Deep Sky Blue
        "Southern Europe": "#4682B4",             # Steel Blue
        "Western Europe": "#6495ED",              # Cornflower Blue
        "Europe": "#4169E1",                      # Royal Blue

        # Oceania
        "Australia and New Zealand": "#9370DB",   # Medium Purple
        "Melanesia": "#8A2BE2",                   # Blue Violet
        "Micronesia": "#BA55D3",                  # Medium Orchid
        "Polynesia": "#DDA0DD",                   # Plum
        "Oceania": "#9932CC",                     # Dark Orchid

        # Others
        "Antarctica": "#E0FFFF",                  # Light Cyan
        "Arctic": "#00FFFF",                      # Cyan
        "Atlantic": "#4682B4"                     # Steel Blue
    }

    # Call the function to animate the network growth
    animate_network_growth(df, period_years, country_to_region, subregion_to_region, color_map, "Network Growth Over Time")
