from abc import ABC, abstractmethod

from pandas import DataFrame

from flight.data import FederatedSubsets, FloxDataset, LocalDataset
from flight.topo import Node
from flight.nn.model import FloxModule


class Process(ABC):
    dataset: FloxDataset

    @abstractmethod
    def start(self, debug_mode: bool = False) -> tuple[FloxModule, DataFrame]:
        """Starts the FL process.

        Returns:
            The trained global module hosted on the leader of `topo`.
            The history metrics from training.
        """

    def fetch_worker_data(self, node: Node):
        match self.dataset:
            case FederatedSubsets():
                return self.dataset.load(node)
            case LocalDataset():
                return self.dataset.load(node)
            case _:
                cls_name = self.dataset.__class__.__name__
                raise TypeError(
                    f"`{cls_name}` is not a valid `FloxDataset`. Class member `dataset` must "
                    f"be a subclass of either `FederatedSubsets` or `LocalDataset`."
                )
