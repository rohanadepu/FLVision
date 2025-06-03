import os
import sys
import multiprocessing
import time
from pathlib import Path
import gc
import psutil
import pandas as pd
import torch
import random
import numpy as np

try:
    sys.path.append("..")
    from flight.topo import Topology
    from flight.runtime.fit import federated_fit
    from flight.strategies.impl.fedavg import FedAvg
    from NIDS import NIDSModule
    from dataset_iotbotnet import loadIOTBOTNET
except Exception as e:
    raise ImportError("Unable to import FloX libraries or model module.") from e


def print_memory_usage(stage):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    virtual_mem = psutil.virtual_memory()
    print(
        f"[Memory Check - {stage}] RSS: {mem_info.rss / (1024 * 1024):.2f} MB, VMS: {mem_info.vms / (1024 * 1024):.2f} MB, Shared: {mem_info.shared / (1024 * 1024):.2f} MB")
    print(
        f"[System Memory] Total: {virtual_mem.total / (1024 * 1024):.2f} MB, Available: {virtual_mem.available / (1024 * 1024):.2f} MB, Used: {virtual_mem.used / (1024 * 1024):.2f} MB, Percent: {virtual_mem.percent}%")


def ensure_numeric_features(df, feature_cols):
    """Ensure all feature columns are numeric and handle any conversion issues."""
    print(f"[DEBUG] Checking data types for feature columns...")

    for col in feature_cols:
        if col in df.columns:
            # Check current data type
            print(f"[DEBUG] Column '{col}' dtype: {df[col].dtype}")

            # Convert to numeric, forcing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Check for any NaN values after conversion
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"[WARNING] Column '{col}' has {nan_count} NaN values after numeric conversion")
                # Fill NaN with median for that column
                df[col].fillna(df[col].median(), inplace=True)
        else:
            print(f"[WARNING] Feature column '{col}' not found in DataFrame")

    return df


# Unified dataset that handles multiple nodes - this is what Flight framework expects
class FlightMultiNodeDataset:
    def __init__(self, node_datasets_dict, feature_cols):
        """
        Args:
            node_datasets_dict: Dictionary mapping node_id to list of file paths
            feature_cols: List of feature column names
        """
        self.node_datasets = {}
        self.feature_cols = feature_cols

        # Create PyTorch datasets for each node
        for node_id, paths in node_datasets_dict.items():
            self.node_datasets[node_id] = WorkerLazyDataset(paths, feature_cols)

        print(f"[DEBUG] Created unified dataset for nodes: {list(self.node_datasets.keys())}")

    def load(self, node):
        """Load method expected by the Flight framework"""
        node_id = node.idx if hasattr(node, 'idx') else node

        if node_id in self.node_datasets:
            print(f"[DEBUG] Loading dataset for node {node_id}")
            return self.node_datasets[node_id]
        else:
            available_nodes = list(self.node_datasets.keys())
            raise ValueError(f"No dataset found for node {node_id}. Available nodes: {available_nodes}")

    def __len__(self):
        # Return total length across all nodes
        return sum(len(dataset) for dataset in self.node_datasets.values())


# Lazy-loading WorkerLazyDataset - moved outside main() to make it picklable


# Lazy-loading WorkerLazyDataset - moved outside main() to make it picklable
class WorkerLazyDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, feature_cols):
        self.file_paths = file_paths
        self.feature_cols = feature_cols
        self.label_col = 'Label'
        self.lengths = []

        # Pre-validate the first file to catch data type issues early
        print(f"[DEBUG] Pre-validating data types in first file...")
        first_df = pd.read_feather(self.file_paths[0])
        first_df = ensure_numeric_features(first_df, self.feature_cols)
        print(f"[DEBUG] Data validation completed for first file")

        for path in self.file_paths:
            df = pd.read_feather(path, columns=[self.label_col])
            self.lengths.append(len(df))

        self.total_len = sum(self.lengths)
        print(f"[DEBUG] Total samples across files: {self.total_len}")

        self.cumulative_lengths = []
        running_total = 0
        for length in self.lengths:
            running_total += length
            self.cumulative_lengths.append(running_total)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        dataset_idx = 0
        while idx >= self.cumulative_lengths[dataset_idx]:
            dataset_idx += 1

        if dataset_idx > 0:
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]
        else:
            local_idx = idx

        df = pd.read_feather(self.file_paths[dataset_idx])

        # Ensure numeric conversion for features
        df = ensure_numeric_features(df, self.feature_cols)

        # Extract features and ensure they're all numeric
        feature_values = df.iloc[local_idx][self.feature_cols].values

        # Debug: Check what we actually got
        if idx % 1000 == 0:  # Only log occasionally
            print(f"[DEBUG] Raw feature_values dtype: {feature_values.dtype}")
            print(f"[DEBUG] Raw feature_values shape: {feature_values.shape}")
            print(f"[DEBUG] First few values: {feature_values[:3] if len(feature_values) > 0 else 'empty'}")

        # Handle different data types safely
        try:
            # If it's already numeric, check for NaN/Inf
            if np.issubdtype(feature_values.dtype, np.number):
                if np.isnan(feature_values).any() or np.isinf(feature_values).any():
                    if idx % 1000 == 0:
                        print(f"[WARNING] Found NaN/Inf values at idx {idx}, replacing with 0")
                    feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)
                feature_values = feature_values.astype(np.float32)
            else:
                # If it's not numeric, force conversion
                print(f"[WARNING] Non-numeric data at idx {idx}, dtype: {feature_values.dtype}")
                print(f"[DEBUG] Sample values: {feature_values[:5]}")

                # Convert each element individually to handle mixed types
                converted_values = []
                for i, val in enumerate(feature_values):
                    try:
                        # Try to convert to float
                        if pd.isna(val) or val == '' or val == 'nan':
                            converted_values.append(0.0)
                        else:
                            converted_values.append(float(val))
                    except (ValueError, TypeError):
                        print(f"[WARNING] Could not convert value '{val}' at feature index {i}, using 0.0")
                        converted_values.append(0.0)

                feature_values = np.array(converted_values, dtype=np.float32)

        except Exception as e:
            print(f"[ERROR] Failed to process features at idx {idx}: {e}")
            print(f"[DEBUG] feature_values: {feature_values}")
            print(f"[DEBUG] feature_values.dtype: {feature_values.dtype}")
            # Fallback: create zeros array
            feature_values = np.zeros(len(self.feature_cols), dtype=np.float32)

        features = torch.tensor(feature_values, dtype=torch.float32)

        label_value = df.iloc[local_idx][self.label_col]

        if label_value not in ['Normal', 'Anomaly']:
            print(f"[DEBUG] Unexpected label value: {label_value}")

        label = torch.tensor(0 if label_value == 'Normal' else 1, dtype=torch.long)

        if idx % 1000 == 0:
            print(f"[DEBUG] Sample idx={idx}, features shape={features.shape}, label={label.item()}")

        return features, label


def ensure_numeric_features(df, feature_cols):
    """Ensure all feature columns are numeric and handle any conversion issues."""
    print(f"[DEBUG] Checking data types for feature columns...")

    for col in feature_cols:
        if col in df.columns:
            # Check current data type
            print(f"[DEBUG] Column '{col}' dtype: {df[col].dtype}")

            # Convert to numeric, forcing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Check for any NaN values after conversion
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"[WARNING] Column '{col}' has {nan_count} NaN values after numeric conversion")
                # Fill NaN with median for that column
                df[col].fillna(df[col].median(), inplace=True)
        else:
            print(f"[WARNING] Feature column '{col}' not found in DataFrame")

    return df


def main():
    multiprocessing.set_start_method("spawn", force=True)
    print("Multiprocessing start method:", multiprocessing.get_start_method())

    print_memory_usage("Start")

    # Build topology
    topo = Topology()
    leader = topo.add_node(kind="leader")
    print(f"Leader added: idx={leader.idx}, kind={leader.kind}")
    topo.leader = leader

    for i in range(1):
        worker = topo.add_node(kind="worker")
        topo.add_edge(leader.idx, worker.idx)
        print(f"Worker {i} added: idx={worker.idx}, kind={worker.kind}")

    print_memory_usage("After Topology Creation")

    # Load dataset paths
    train_paths, test_paths, feature_names = loadIOTBOTNET()
    print(f"Loaded {len(train_paths)} train chunks and {len(test_paths)} test chunks.")

    print_memory_usage("After Dataset Load")

    # Manual federated split using file paths
    random.shuffle(train_paths)
    node_datasets = []
    print("ALL NODES:", topo.nodes())
    worker_nodes = list(topo.workers)
    print("ALL WORKER NODES:", worker_nodes)
    files_per_worker = len(train_paths) // len(worker_nodes)

    for idx, worker in enumerate(worker_nodes):
        assigned_files = train_paths[idx * files_per_worker: (idx + 1) * files_per_worker]
        node_datasets.append((worker.idx, assigned_files))
        print(f"Assigned {len(assigned_files)} files to worker {worker.idx}.")

    print_memory_usage("After Splitting")

    gc.collect()
    print(">>> Cleared any unused memory before federated_fit.")

    print_memory_usage("After GC")

    # Create a unified dataset that handles all nodes
    # Convert node_datasets to dictionary format for the unified dataset
    node_paths_dict = {}
    for node_id, paths in node_datasets:
        node_paths_dict[node_id] = paths

    print(f"[DEBUG] Creating unified dataset for nodes: {list(node_paths_dict.keys())}")

    # Create the unified dataset that Flight framework expects
    unified_dataset = FlightMultiNodeDataset(node_paths_dict, feature_names)

    print(f"[DEBUG] Unified dataset created successfully")
    print(f"[DEBUG] Unified dataset type: {type(unified_dataset)}")
    print(f"[DEBUG] Unified dataset has load method: {hasattr(unified_dataset, 'load')}")

    # Preflight feature and label check
    print("\n[Preflight Data Check]")
    # Test the unified dataset with each worker node
    for worker in topo.workers:
        try:
            print(f"[DEBUG] Testing dataset load for worker node {worker.idx}")
            pytorch_dataset = unified_dataset.load(worker)
            x, y = pytorch_dataset[0]
            print(f"Node {worker.idx}: Feature Shape = {x.shape}, Label = {y.item()}")
            print(f"Node {worker.idx}: Feature dtype = {x.dtype}, Label dtype = {y.dtype}")

            # Check for any NaN or infinite values
            if torch.isnan(x).any():
                print(f"[WARNING] Node {worker.idx}: Features contain NaN values")
            if torch.isinf(x).any():
                print(f"[WARNING] Node {worker.idx}: Features contain infinite values")

        except Exception as e:
            print(f"[ERROR] Failed to load sample from Node {worker.idx}: {e}")
            raise

    # Federated fit
    print("\n>>> Starting federated_fit with multiprocessing...")
    start_time = time.time()

    try:
        print(f"[DEBUG] Calling federated_fit with unified dataset...")
        print(f"[DEBUG] Dataset type: {type(unified_dataset)}")
        print(f"[DEBUG] Dataset has load method: {hasattr(unified_dataset, 'load')}")

        _, df = federated_fit(
            topo,
            NIDSModule(),
            unified_dataset,  # Single unified dataset
            5,
            strategy=FedAvg()
        )
        df["strategy"] = "fed-avg"
        print(">>> federated_fit completed successfully.")

    except Exception as e:
        print(f">>> ERROR during federated_fit(): {e}")
        print(f"[DEBUG] Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise

    duration = time.time() - start_time
    print(f"\n>>> federated_fit took {duration:.2f} seconds")

    # Save results
    train_history = df.reset_index(drop=True)
    Path("out").mkdir(exist_ok=True)
    train_history.to_feather(Path("out/federated_history.feather"))
    print(">>> Finished and saved training log to out/federated_history.feather")


if __name__ == "__main__":
    main()