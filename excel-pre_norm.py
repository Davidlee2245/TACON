import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import multiprocessing as mp
from functools import partial

def find_excel_files(root_dir="./datasets"):
    """Find all Excel files in directory"""
    excel_files = []
    print("root_dir:", root_dir)
    print("Current working directory:", os.getcwd())
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.xlsx', '.xls', '.csv')):
                full_path = os.path.join(root, file)
                excel_files.append(full_path)
                print(f"Found: {full_path}")
    
    return excel_files


def extract_concentration(filename):
    """Extract concentration from filename"""
    try:
        # Get the part before 'mM' in the filename
        concentration = filename.split('mM')[0]
        return int(concentration)
    except:
        print(f"Could not extract concentration from {filename}")
        return None

def process_spectrometer_data(excel_path):
    """Process UV-VIS spectrometer data and separate by wells"""
    
    # Extract concentration and determine file order
    filename = os.path.basename(excel_path)
    concentration = extract_concentration(filename)
    
    # Get all files with same concentration
    directory = os.path.dirname(excel_path)
    all_files = []
    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)) and f.startswith(f"{concentration}mM"):
            all_files.append(f)
    
    # Sort files by date (assuming date is at the end)
    all_files.sort()
    
    # Debug print to check files
    print(f"Filename being searched: {filename}")
    print(f"All files found: {all_files}")
    
    # Get the order number (1-based index)
    try:
        file_order = all_files.index(filename) + 1
    except ValueError:
        print(f"Warning: Could not find exact filename match. Using order 1.")
        file_order = 1
    
    # Read the excel file
    print(f"Processing: {excel_path}")
    df = pd.read_excel(excel_path)
    
    # Print first few rows and columns for debugging
    print("\nFirst few rows and columns:")
    print(df.head())
    
    # Get wavelengths from first column
    wavelengths = df.iloc[:, 0]

    # Create directory for separated files
    output_dir = 'separated_wells_norm'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each well
    for well in range(1, 97):  # Start with fewer wells for testing
        # Initialize dataframe for this well
        well_data = pd.DataFrame({'Wavelength': wavelengths})
        
        # Get columns for this well
        well_cols = []
        for col in df.columns:
            try:
                # Split column name and check if it starts with current well number
                parts = str(col).strip().split()
                if parts[0] == str(well):
                    well_cols.append(col)
            except:
                continue
        
        if well_cols:
            print(f"\nProcessing Well {well}")
            print(f"Found columns: {well_cols}")
            
            # Add each time point to well_data and normalize
            for col in well_cols:
                # Get the original data
                original_data = df[col]
                # Normalize by dividing by the maximum value
                max_value = original_data.max()
                normalized_data = original_data / max_value
                # Add normalized data to well_data
                well_data[f"{col}_normalized"] = normalized_data
                # Also keep original data if needed
                #well_data[col] = original_data
            
            # Update output filename format
            output_path = os.path.join(output_dir, f'{concentration}_{file_order}_well_{well}.xlsx')
            well_data.to_excel(output_path, index=False)
            print(f"Saved data for well {well}")
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            for col in well_cols:
                plt.plot(wavelengths, well_data[f"{col}_normalized"], label=col)
            plt.title(f'Well {well} - Time Series')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Signal')
            plt.legend()
            # Update plot filename format
            plt.savefig(os.path.join(output_dir, f'{concentration}_{file_order}_well_{well}_plot.png'))
            plt.close()
        else:
            print(f"No data found for well {well}")
    
    return output_dir


def visualize_excel_data(excel_path):
    """Visualize data from a single Excel file"""
    print(f"\nProcessing: {excel_path}")
    
    # Read file based on extension
    if excel_path.endswith('.csv'):
        df = pd.read_csv(excel_path)
    else:
        df = pd.read_excel(excel_path)
    
    # Create a figure for this file
    plt.figure(figsize=(15, 8))
    plt.suptitle(f'Data Visualization: {os.path.basename(excel_path)}')
    
    # Plot numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(2, len(numerical_cols)//2 + 1, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'{col}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return df

def main():
    # Find all excel files
    excel_files = []
    for root, dirs, files in os.walk("./datasets"):
        for file in files:
            if file.endswith(('.xlsx', '.xls')) and not file.startswith('~$'):
                full_path = os.path.join(root, file)
                excel_files.append(full_path)
                print(f"Found: {full_path}")
    
    # Create a pool of workers
    num_cores = mp.cpu_count()  # Get number of CPU cores
    print(f"Using {num_cores} CPU cores")
    
    # Create and configure the process pool
    with mp.Pool(processes=num_cores) as pool:
        # Process files in parallel
        results = pool.map(process_spectrometer_data, excel_files)
    
    print("All files processed")
    for output_dir in results:
        print(f"Processed data saved in: {output_dir}")

if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()