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

def loadIOTBOTNET(poisonedDataType=None):
    print("Starting loadIOTBOTNET()...")
    sample_size = 1

    dataset_paths = {
        "LF33": "./datasets/IOTBOTNET2020_POISONEDLF33",
        "LF66": "./datasets/IOTBOTNET2020_POISONEDLF66",
        "FN33": "./datasets/IOTBOTNET2020_POISONEDFN33",
        "FN66": "./datasets/IOTBOTNET2020_POISONEDFN66",
        None: "/home/cc/datasets/IOTBOTNET2020"
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
        'ddos': ['ddos_udp', 'ddos_tcp', 'ddos_http'],
        'dos': ['dos_udp', 'dos_tcp', 'dos_http'],
        'scan': ['os', 'service'],
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
                    df = pd.read_csv(file_path)
                    original_len = len(df)

                    if original_len > 10000:
                        df = df.sample(frac=0.01, random_state=47)

                    df.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
                    df.dropna(inplace=True)
                    if 'Label' in df.columns:
                        df['Label'] = df['Label'].apply(lambda x: 'Normal' if x == 'Normal' else 'Anomaly')

                    train_split, test_split = train_test_split(df, test_size=0.2, random_state=47)

                    def save_chunks(split_df, kind):
                        chunks = np.array_split(split_df, 4)
                        for i, chunk in enumerate(chunks):
                            path = f"./tmp/{kind}_chunk_{chunk_id}_{i}.feather"
                            chunk.reset_index(drop=True).to_feather(path)
                            if kind == 'train':
                                train_paths.append(path)
                            else:
                                test_paths.append(path)
                            print(f"Saved {len(chunk)} {kind} -> {path}")

                    save_chunks(train_split, "train")
                    save_chunks(test_split, "test")

                    chunk_id += 1
                    del df, train_split, test_split

    print(f"Saved {len(train_paths)} train chunks and {len(test_paths)} test chunks.")
    return train_paths, test_paths, relevant_features

