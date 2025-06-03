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
    print(f"[Memory Check - {stage}] RSS: {mem_info.rss / (1024 * 1024):.2f} MB, VMS: {mem_info.vms / (1024 * 1024):.2f} MB, Shared: {mem_info.shared / (1024 * 1024):.2f} MB")
    print(f"[System Memory] Total: {virtual_mem.total / (1024 * 1024):.2f} MB, Available: {virtual_mem.available / (1024 * 1024):.2f} MB, Used: {virtual_mem.used / (1024 * 1024):.2f} MB, Percent: {virtual_mem.percent}%")


def main():
    multiprocessing.set_start_method("spawn", force=True)
    print("Multiprocessing start method:", multiprocessing.get_start_method())

    print_memory_usage("Start")

    # Build topology
    topo = Topology()
    leader = topo.add_node(kind="leader")
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
    all_nodes = topo.nodes()
    worker_nodes = [n for n in all_nodes if n.kind == 'worker']
    files_per_worker = len(train_paths) // len(worker_nodes)

    for idx, worker in enumerate(worker_nodes):
        assigned_files = train_paths[idx * files_per_worker: (idx + 1) * files_per_worker]
        node_datasets.append((worker.idx, assigned_files))
        print(f"Assigned {len(assigned_files)} files to worker {worker.idx}.")

    print_memory_usage("After Splitting")

    gc.collect()
    print(">>> Cleared any unused memory before federated_fit.")

    print_memory_usage("After GC")

    # Lazy-loading WorkerLazyDataset
    class WorkerLazyDataset(torch.utils.data.Dataset):
        def __init__(self, file_paths, feature_cols):
            self.file_paths = file_paths
            self.feature_cols = feature_cols
            self.label_col = 'Label'
            self.lengths = []

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
            features = torch.tensor(df.iloc[local_idx][self.feature_cols].values, dtype=torch.float32)
            label_value = df.iloc[local_idx][self.label_col]

            if label_value not in ['Normal', 'Anomaly']:
                print(f"[DEBUG] Unexpected label value: {label_value}")

            label = torch.tensor(0 if label_value == 'Normal' else 1, dtype=torch.long)

            if idx % 1000 == 0:
                print(f"[DEBUG] Sample idx={idx}, features shape={features.shape}, label={label.item()}")

            return features, label

    # Wrap node datasets
    final_node_datasets = []
    for node_id, paths in node_datasets:
        wrapped_dataset = WorkerLazyDataset(paths, feature_names)
        final_node_datasets.append((node_id, wrapped_dataset))

    # Preflight feature and label check
    print("\n[Preflight Data Check]")
    for node_id, dataset in final_node_datasets:
        x, y = dataset[0]
        print(f"Node {node_id}: Feature Shape = {x.shape}, Label = {y.item()}")

    # Federated fit
    print("\n>>> Starting federated_fit with multiprocessing...")
    start_time = time.time()

    try:
        _, df = federated_fit(
            topo,
            NIDSModule(),
            final_node_datasets,
            5,
            strategy=FedAvg()
        )
        df["strategy"] = "fed-avg"
        print(">>> federated_fit completed successfully.")

    except Exception as e:
        print(">>> ERROR during federated_fit():", e)
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

