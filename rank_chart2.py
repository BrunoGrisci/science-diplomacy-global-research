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

    # Prepare the chart
    plt.figure(figsize=(15, 12))

    # Plot each country's data
    for country in data.index:
        y_values = data.loc[country]  # Original values
        years = [extract_year_range(col) for col in data.columns]  # Cleaned x-axis labels

        # Plot scatter points with sizes proportional to values
        plt.scatter(
            years,
            y_values,
            s=y_values * 200,  # Scale sizes (adjust multiplier as needed)
            alpha=0.6
        )

        # Annotate the last data point with the country name
        plt.text(
            years[-1],  # Last year
            y_values.iloc[-1],  # Value in the last year
            country,  # Country name
            fontsize=10,
            ha='left',  # Align to the left of the point
            va='center',  # Vertical alignment centered
            alpha=0.8,
            backgroundcolor="white" 
        )

        # Connect points with lines
        plt.plot(years, y_values, alpha=0.5)

        # Annotate data points with their values
        for i, year in enumerate(years):
            plt.text(
                year, 
                y_values.iloc[i], 
                f"{y_values.iloc[i]:.2f}",  # Display original value
                ha="center", 
                va="bottom", 
                fontsize=8, 
                alpha=0.7,
                #backgroundcolor="white"  # Add a background for better visibility
            )

    # Customize the chart
    plt.title(
        f"Rank Changes with {file_path.split('/')[-1].replace('merged_', '').replace('.csv', '')}",
        fontsize=16
    )
    plt.xlabel("Time Period", fontsize=14)
    plt.ylabel(file_path.split('/')[-1].replace('merged_', '').replace('.csv', ''), fontsize=14)
    plt.xticks(fontsize=12, rotation=0)  # Adjust font size and rotation of x-axis labels
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)

    # Add a source or context for the data
    #plt.figtext(0.5, 0.01, "Source: Your Dataset Name", ha="center", fontsize=10)

    # Save the plot as a PDF
    output_file = file_path.replace(".csv", ".pdf")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
