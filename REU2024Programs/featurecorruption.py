import pandas as pd
import numpy as np
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def is_floatable(col):
    try:
        pd.to_numeric(col, errors='raise')
        return True
    except ValueError:
        return False

def corrupt_features(file_path, output_folder, corruption_percentage):
    df = pd.read_csv(file_path)
    
    # Determine numeric columns safely
    numeric_columns = [col for col in df.columns if is_floatable(df[col])]
    
    # Corrupt each numeric feature based on the corruption percentage
    for column in numeric_columns:
        if column == 'label' or column == 'Label':
            continue
        # Convert column to float to avoid dtype issues
        df[column] = df[column].astype(float)
        mask = np.random.rand(len(df[column])) < corruption_percentage / 100
        # Simple corruption method: add random noise based on the standard deviation of the column
        if np.any(mask):
            std_dev = df[column].std()
            random_noise = np.random.normal(0, std_dev, size=len(df[column]))
            df.loc[mask, column] += random_noise[mask]
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    output_file_name = os.path.join(output_folder, os.path.basename(file_path))
    df.to_csv(output_file_name, index=False)
    print(f"Corrupted dataset saved as {output_file_name}")

def handle_file(file_path, output_folder, corruption_percentage):
    relative_path = os.path.relpath(file_path, input_folder)
    new_output_folder = os.path.join(output_folder, os.path.dirname(relative_path))
    corrupt_features(file_path, new_output_folder, corruption_percentage)

def corrupt_all_files(input_folder, corruption_percentage):
    output_folder = input_folder + '_POISONEDFN66'
    
    # Ensure the output directory does not exist to start fresh
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    # Get all CSV files in the input folder
    csv_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_folder) for f in filenames if f.endswith('.csv')]
    
    # Use a number of threads based on the number of CPUs
    num_threads = multiprocessing.cpu_count()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(handle_file, csv_files, [output_folder]*len(csv_files), [corruption_percentage]*len(csv_files))

# Example usage
input_folder = 'IOTBOTNET2020'  # or 'IOTBOTNET2020'
corruption_percentage = 66  # Percentage of corruption
corrupt_all_files(input_folder, corruption_percentage)
