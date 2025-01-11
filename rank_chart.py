# Bruno Iochins Grisci
# November 2024

import re
import pandas as pd
import matplotlib.pyplot as plt

# Function to extract the year range
def extract_year_range(column_name):
    match = re.search(r"\d{4}_\d{4}", column_name.replace(" ", ""))  # Match "YYYY_YYYY"
    return match.group(0) if match else column_name  # Return the match or the original if no match

def main():
    # Load the CSV file
    file_path = "output/authors/merged_EigenvectorCentralityB.csv"  # Replace with your file path
    data = pd.read_csv(file_path, index_col=0)

    # Sort rows by rank for visualization
    sorted_data = data.rank(ascending=False, method="min", axis=0)

    # Prepare the chart
    plt.figure(figsize=(15, 12))
    
    # Plot each country's data
    for country in data.index:
        y_values = data.loc[country]  # Original values
        # Plot with scatter for size proportionality
        plt.scatter(
            [extract_year_range(col) for col in data.columns], 
            y_values, 
            s=y_values * 200,  # Scale sizes (adjust multiplier for better visualization)
            label=country, 
            alpha=0.6
        )
      
        # Annotate the last data point with the country name
        plt.text(
            [extract_year_range(col) for col in data.columns][-1],  # Last year
            y_values.iloc[-1],  # Value in the last year
            country,  # Country name
            fontsize=10,
            ha='left',  # Align to the bottom of the point
            va='bottom',
            alpha=0.8
        )


        # Connect points with lines
        plt.plot([extract_year_range(col) for col in data.columns], y_values, marker=None, alpha=0.5)
        # Annotate points with original values
        for year in data.columns:
            plt.text(
                extract_year_range(year), 
                y_values[year], 
                f"{y_values[year]:.2f}",  # Display original value
                ha="center", 
                va="top", 
                fontsize=8, 
                alpha=0.7
            )

    # Customize the chart
    plt.title("Rank Changes with" + file_path.replace("output/authors/merged_", ""), fontsize=16)
    plt.xlabel("Time period", fontsize=14)
    plt.ylabel(file_path.replace("output/authors/merged_", ""), fontsize=14)
    plt.xticks(rotation=45)
    #plt.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Country")
    plt.grid(alpha=0.3)

    # Show the plot
    plt.tight_layout()
    plt.savefig(file_path.replace(".csv", ".pdf"), dpi=300)
    plt.show()

if __name__ == "__main__":
    main()