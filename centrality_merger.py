# Bruno Iochins Grisci
# November 2024

import pandas as pd
import glob
import os

# Specify the folder where your .csv files are located
folder_path = "output/authors/"

# Example: Only merge files starting with pattern
csv_files = glob.glob(os.path.join(folder_path, "akh_v5_1209_final_worksheet*authors_centrality.csv"))

# Dictionary to store data for each column across files
columns_dict = {}

# Process each file
for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file, index_col=0)  # Assuming row labels are in the first column
    # Extract each column
    for col_name in df.columns:
        if col_name not in columns_dict:
            columns_dict[col_name] = []  # Initialize a list for this column
        # Rename the column to include the file name for distinction
        renamed_col = df[col_name].rename(f"{col_name}_{os.path.basename(file)}")
        columns_dict[col_name].append(renamed_col)

# Write output files for each column
for col_name, columns in columns_dict.items():
    # Combine all columns of the same name, aligning rows by index
    merged_df = pd.concat(columns, axis=1, join='outer')
    # Fill missing values with 0.0
    merged_df = merged_df.fillna(0.0)    
    # Save the merged DataFrame to a new CSV
    output_file = os.path.join(folder_path, f"merged_{col_name}.csv")
    merged_df.to_csv(output_file)

print("Merging complete. Check your output files.")
