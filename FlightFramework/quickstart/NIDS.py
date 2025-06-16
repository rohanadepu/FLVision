import torch
from torch import nn
from flight.nn import FloxModule


class NIDSModule(FloxModule):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=2, lr: float = 0.01):
        super().__init__()
        self.lr = lr
        self.last_accuracy = torch.tensor(0.0)

        self.linear_stack = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.linear_stack(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        preds = self(inputs)
        loss = torch.nn.functional.cross_entropy(preds, targets)

        # Track accuracy
        correct = (preds.argmax(dim=1) == targets).sum().item()
        total = targets.size(0)
        acc = correct / total
        self.last_accuracy = torch.tensor(acc)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=self.lr)

