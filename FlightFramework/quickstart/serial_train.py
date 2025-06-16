import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from NIDS import NIDSModule
import os
import numpy as np

# Define feature columns (must match your dataset)
FEATURE_COLS = [
    'Src_Port', 'Pkt_Size_Avg', 'Bwd_Pkts/s', 'Pkt_Len_Mean', 'Dst_Port', 'Bwd_IAT_Max',
    'Flow_IAT_Mean', 'ACK_Flag_Cnt', 'Flow_Duration', 'Flow_IAT_Max', 'Flow_Pkts/s',
    'Fwd_Pkts/s', 'Bwd_IAT_Tot', 'Bwd_Header_Len', 'Bwd_IAT_Mean', 'Bwd_Seg_Size_Avg'
]
LABEL_COL = 'Label'

# Load small feather files
train_files = sorted([f for f in Path("./tmp").glob("train_chunk_*.feather")])
print(f"Found {len(train_files)} train files.")

class LazyDataset(torch.utils.data.Dataset):
    def __init__(self, files, feature_cols):
        self.files = files
        self.feature_cols = feature_cols
        self.label_col = LABEL_COL
        self.lengths = []
        for f in files:
            df = pd.read_feather(f, columns=[self.label_col])
            self.lengths.append(len(df))
        self.total = sum(self.lengths)
        self.cumsum = [sum(self.lengths[:i+1]) for i in range(len(self.lengths))]

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        for file_idx, end in enumerate(self.cumsum):
            if idx < end:
                break
        else:
            raise IndexError(f"Index {idx} out of range (cumsum = {self.cumsum})")

        local_idx = idx if file_idx == 0 else idx - self.cumsum[file_idx - 1]
        #print(f"Reading index {idx} (local {local_idx}) from file {self.files[file_idx]}")
        df = pd.read_feather(self.files[file_idx])
        x = torch.tensor(df.iloc[local_idx][self.feature_cols].astype(np.float32).values, dtype=torch.float32)
        x = (x - x.mean()) / (x.std() + 1e-8)  # Normalize features
        yval = df.iloc[local_idx][LABEL_COL]
        y = torch.tensor(0 if yval == 'Normal' else 1, dtype=torch.long)
        return x, y

# Create dataset and dataloader
dataset = LazyDataset(train_files, FEATURE_COLS)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

# Create model and optimizer
model = NIDSModule()
optimizer = model.configure_optimizers()

# Training loop
print("\n>>> Starting local training (serial mode)...")
model.train()
for epoch in range(5):
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        preds = model(x)
        loss = torch.nn.functional.cross_entropy(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (preds.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {acc*100:.2f}%")

print("\n Training complete (no multiprocessing used).")
