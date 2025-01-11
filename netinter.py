# Bruno Iochins Grisci
# September 2024

import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import ipywidgets as widgets
from collections import Counter
from IPython.display import display

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

def build_network_graph(df, dict_c2r, dict_s2r):
    """Builds the network graph for a specific year."""
    G = nx.DiGraph()
    
    # Build edges from the filtered data
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

    # Add edges to the graph
    for edge, count in edge_counts.items():
        G.add_edge(edge[0], edge[1], weight=count)

    return G

def plot_network(G, pos, dict_c2r, dict_s2r, color_map):
    """Plots the network using Plotly."""
    # Extract node positions
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]
    
    # Define edge trace
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                     mode='lines', line=dict(width=0.5, color='#888'),
                                     hoverinfo='none'))

    # Define node trace
    node_trace = go.Scatter(x=x_nodes, y=y_nodes, mode='markers',
                            marker=dict(size=[G.degree(node) * 10 for node in G.nodes()],
                                        color=[color_map.get(node, color_map.get(dict_c2r.get(node, node)[1], '#000000')) for node in G.nodes()],
                                        colorscale='Viridis'),
                            text=list(G.nodes()), hoverinfo='text')

    layout = go.Layout(showlegend=False, hovermode='closest',
                       margin=dict(b=0, l=0, r=0, t=40),
                       xaxis=dict(showgrid=False, zeroline=False),
                       yaxis=dict(showgrid=False, zeroline=False),
                       title="Network Graph")

    return go.Figure(data=edge_trace + [node_trace], layout=layout)

def interactive_network(df, period_years, dict_c2r, dict_s2r, color_map):
    # Initialize a directed graph
    G = nx.DiGraph()

    # Initialize positions
    pos = None

    # Slider widget to select year
    year_slider = widgets.IntSlider(
        value=period_years[0],
        min=period_years[0],
        max=period_years[1],
        step=1,
        description='Year:',
        continuous_update=False
    )

    # Output area for the graph
    output = widgets.Output()

    @output.capture()
    def update_network(year):
        # Clear output area
        output.clear_output(wait=True)

        # Filter data for the current year
        current_df = filter_data_by_year(df, year)

        # Build the network graph for the current year
        G = build_network_graph(current_df, dict_c2r, dict_s2r)

        # Update positions only if the graph has nodes
        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G)  # You can use other layouts like circular_layout

        # Plot the updated graph
        fig = plot_network(G, pos, dict_c2r, dict_s2r, color_map)
        fig.show()

    # Update network when the slider value changes
    interactive = widgets.interactive_output(update_network, {'year': year_slider})
    # Display the slider and output area
    display(year_slider, interactive)

# Example usage
if __name__ == '__main__':
    # Assume df is your DataFrame already processed with the necessary columns
    csv_file_path = 'v4_1209_final_worksheet.csv'
    period_years = [2002, 2023]

    # Columns to keep
    columns_to_keep = ['affiliation_country', 'affiliation_subregion_manual', 'affiliation_region_manual', 'target_country_manual', 'target_subregion_manual', 'target_region_manual', 'affliation_north_or_south', 'SD-related (YES/NO)', 'coverDate']
    df = pd.read_csv(csv_file_path, usecols=columns_to_keep)
    df['Year'] = pd.to_datetime(df['coverDate']).dt.year
    df = df[df['SD-related (YES/NO)'].str.lower() == 'yes']
    df = df[df['affiliation_country'].str.lower() != 'none']
    df = df[df['target_region_manual'].str.lower() != 'none']
    df = df.dropna()

    # Load the region mappings
    region_dictionary_csv = '22072024_Countries aggregated_BrunoGrisci-3007 - UN Geoscheme.csv'
    region_dictionary = pd.read_csv(region_dictionary_csv, usecols=['Country or Area', 'Region Name', 'Sub-region Name'])
    country_to_region = region_dictionary.set_index('Country or Area').T.to_dict('list')
    subregion_to_region = region_dictionary.set_index('Sub-region Name').T.to_dict('list')

    # Define color map for regions (as you had before)
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

    # Create the interactive visualization
    interactive_network(df, period_years, country_to_region, subregion_to_region, color_map)
