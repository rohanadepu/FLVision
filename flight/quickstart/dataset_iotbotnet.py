import os
import random
import pandas as pd
import gdown
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import psutil
from pathlib import Path
import numpy as np


def ensure_dataset_downloaded(dataset_path, gdrive_zip_url):
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Downloading from {gdrive_zip_url}...")
        os.makedirs("./datasets", exist_ok=True)

        zip_path = "./datasets/IOTBOTNET2020.zip"

        gdown.download(gdrive_zip_url, zip_path, quiet=False)

        print("Download completed. Extracting...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("./datasets")

        os.remove(zip_path)
        print("Dataset extracted successfully.")


def memory_status(stage):
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    print(f"[{stage}] Memory Usage: {mem.rss / (1024 * 1024):.2f} MB RSS")


def clean_and_preprocess_dataframe(df, relevant_features):
    """Clean and preprocess the dataframe to ensure all features are numeric."""
    print(f"[DEBUG] Preprocessing dataframe with {len(df)} rows...")

    # Replace infinite values with NaN
    df.replace([float('inf'), -float('inf')], float('nan'), inplace=True)

    # Ensure all relevant feature columns are numeric
    for col in relevant_features:
        if col in df.columns:
            # Check if column exists and convert to numeric
            original_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Count and report conversion issues
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"[WARNING] Column '{col}' had {nan_count} non-numeric values converted to NaN")

            if original_dtype != df[col].dtype:
                print(f"[INFO] Column '{col}' converted from {original_dtype} to {df[col].dtype}")
        else:
            print(f"[WARNING] Feature column '{col}' not found in dataframe columns: {df.columns.tolist()}")

    # Drop rows with any NaN values in relevant features
    initial_len = len(df)
    df.dropna(subset=relevant_features, inplace=True)
    dropped_count = initial_len - len(df)
    if dropped_count > 0:
        print(f"[INFO] Dropped {dropped_count} rows containing NaN values in feature columns")

    # Ensure Label column is properly formatted
    if 'Label' in df.columns:
        df['Label'] = df['Label'].apply(lambda x: 'Normal' if x == 'Normal' else 'Anomaly')
        print(f"[INFO] Label distribution: {df['Label'].value_counts().to_dict()}")

    # Final verification that all feature columns are numeric
    for col in relevant_features:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"[ERROR] Column '{col}' is still not numeric: {df[col].dtype}")
            else:
                print(f"[SUCCESS] Column '{col}' is numeric: {df[col].dtype}")

    print(f"[DEBUG] Preprocessing completed. Final dataframe has {len(df)} rows.")
    return df


def loadIOTBOTNET(poisonedDataType=None):
    print("Starting loadIOTBOTNET()...")
    sample_size = 1

    dataset_paths = {
        "LF33": "./datasets/IOTBOTNET2020_POISONEDLF33",
        "LF66": "./datasets/IOTBOTNET2020_POISONEDLF66",
        "FN33": "./datasets/IOTBOTNET2020_POISONEDFN33",
        "FN66": "./datasets/IOTBOTNET2020_POISONEDFN66",
        None: "/home/cc/datasets/IOTBOTNET2020/DDoS"
    }
    DATASET_DIRECTORY = dataset_paths.get(poisonedDataType, dataset_paths[None])

    DATASET_URL = 'https://drive.google.com/uc?id=1Lz9KcI4ZZP9Svn7huV90MQUMZNLn96gX&export=download'

    # ensure_dataset_downloaded(DATASET_DIRECTORY, DATASET_URL)
    # memory_status("After ensure_dataset_downloaded")

    relevant_features = [
        'Src_Port', 'Pkt_Size_Avg', 'Bwd_Pkts/s', 'Pkt_Len_Mean', 'Dst_Port', 'Bwd_IAT_Max',
        'Flow_IAT_Mean', 'ACK_Flag_Cnt', 'Flow_Duration', 'Flow_IAT_Max', 'Flow_Pkts/s',
        'Fwd_Pkts/s', 'Bwd_IAT_Tot', 'Bwd_Header_Len', 'Bwd_IAT_Mean', 'Bwd_Seg_Size_Avg'
    ]

    categories = {
        'ddos': ['ddos_udp', 'ddos_tcp', 'ddos_http', 'os', 'service'],
        'dos': ['dos_udp', 'dos_tcp', 'dos_http'],
        'scan': ['OS', 'Service'],
        'theft': ['data_exfiltration', 'keylogging']
    }

    def flexible_find(folder, target_subfolder):
        folder = folder.lower()
        target_subfolder = target_subfolder.lower()

        for root, dirs, files in os.walk(DATASET_DIRECTORY):
            for d in dirs:
                if folder in d.lower() and target_subfolder in d.lower():
                    return os.path.join(root, d)
        return None

    Path("./tmp").mkdir(exist_ok=True)

    train_paths = []
    test_paths = []
    chunk_id = 0

    print("Searching and processing attack datasets...")
    for category, subcategories in categories.items():
        for sub in subcategories:
            dir_path = flexible_find(category, sub)
            if dir_path and os.path.exists(dir_path):
                csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
                for file in csv_files:
                    file_path = os.path.join(dir_path, file)
                    print("File found:", file_path)

                    try:
                        df = pd.read_csv(file_path)
                        original_len = len(df)
                        print(f"[INFO] Loaded CSV with {original_len} rows and {len(df.columns)} columns")

                        if original_len > 10000:
                            df = df.sample(frac=0.01, random_state=47)
                            print(f"[INFO] Sampled down to {len(df)} rows")

                        # Clean and preprocess the dataframe
                        df = clean_and_preprocess_dataframe(df, relevant_features)

                        if len(df) == 0:
                            print(f"[WARNING] No valid data remaining after preprocessing {file_path}")
                            continue

                        train_split, test_split = train_test_split(df, test_size=0.2, random_state=47)

                        def save_chunks(split_df, kind):
                            chunks = np.array_split(split_df, 4)
                            for i, chunk in enumerate(chunks):
                                # Reset index and ensure proper data types before saving
                                chunk = chunk.reset_index(drop=True)

                                # Final comprehensive check on feature columns before saving
                                for col in relevant_features:
                                    if col in chunk.columns:
                                        # Ensure float32 and handle any remaining issues
                                        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                                        chunk[col] = chunk[col].fillna(0.0)  # Fill any remaining NaN
                                        chunk[col] = chunk[col].astype(np.float32)

                                # Also ensure label is clean
                                if 'Label' in chunk.columns:
                                    chunk['Label'] = chunk['Label'].astype(str)  # Ensure string type

                                path = f"./tmp/{kind}_chunk_{chunk_id}_{i}.feather"
                                chunk.to_feather(path)

                                if kind == 'train':
                                    train_paths.append(path)
                                else:
                                    test_paths.append(path)
                                print(f"Saved {len(chunk)} {kind} samples -> {path}")

                                # Verify the saved file has correct dtypes
                                if i == 0:  # Check first chunk of each split
                                    verification_df = pd.read_feather(path)
                                    for col in relevant_features:
                                        if col in verification_df.columns:
                                            actual_dtype = verification_df[col].dtype
                                            if actual_dtype != np.float32:
                                                print(f"[WARNING] Saved file has wrong dtype for {col}: {actual_dtype}")
                                    print(f"[INFO] Verified saved chunk has correct dtypes")

                        save_chunks(train_split, "train")
                        save_chunks(test_split, "test")

                        chunk_id += 1
                        del df, train_split, test_split

                    except Exception as e:
                        print(f"[ERROR] Failed to process file {file_path}: {e}")
                        continue

    print(f"Saved {len(train_paths)} train chunks and {len(test_paths)} test chunks.")

    # Verify that we have some data
    if len(train_paths) == 0:
        raise ValueError("No training data was successfully processed!")

    return train_paths, test_paths, relevant_features