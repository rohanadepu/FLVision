import torch

from flight.nn import FloxModule


class DebugModule(FloxModule):
    """
    A very lightweight ``FloxModule`` implementation that is used for lightweight debugging.
    """

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(1, 1))

    def forward(self, x):
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int  # type: ignore[override]
    ) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=1e-3)
