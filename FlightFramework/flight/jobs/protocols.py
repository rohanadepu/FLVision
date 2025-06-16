"""
This file provides protocols for the different types of jobs that are used in FLoX. More specifically, we
define protocols for:

1. aggregation jobs (``AggregableJob``)
2. local training jobs (``TrainableJob``)

These protocols can be used to define custom impl of aggregation jobs for highly-customized FLoX processes.
However, this is not necessary for the vast majority of imaginable cases.
Should users choose to do this, it is up to the user's discretion to do so safely and correctly.

All protocols presented here rely on the ``__call__`` method to define callable classes.
So, each protocol implementation is a separate class with a matching signature.

Notes:
    It is worth noting that the body of the ``__call__`` method for any protocol implementation here
    must act as a "pure function". This means any necessary Python dependencies (i.e., `import` statements)
    needed for the method to run must be included within the body of the ``__call__`` method. This is
    required for execution on *Globus Compute*.
"""

from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from flight.data import FloxDataset
    from flight.topo import Node
    from flight.nn import FloxModule
    from flight.nn.typing import Params
    from flight.runtime import Result
    from flight.runtime.transfer import BaseTransfer
    from flight.strategies import AggregatorStrategy, TrainerStrategy, WorkerStrategy


class LauncherFunction(t.Protocol):
    """
    Utility protocol that simply identifies any of callable that takes a ``Node``
    as its first argument.
    """

    def __call__(self, node: Node, *args, **kwargs) -> t.Any:
        pass


@t.runtime_checkable
class AggregableJob(t.Protocol):
    """
    A protocol that defines functions that are valid impl to be used for model aggregation in
    launching FLoX processes.

    Notes:
        FLoX provides default impl of this protocol via
        [AggregateJob][flight.jobs.aggregation.AggregateJob] and
        [DebugAggregateJob][flight.jobs.aggregation.DebugAggregateJob].
    """

    @staticmethod
    def __call__(
        node: Node,
        children: t.Iterable[Node],
        transfer: BaseTransfer,
        aggr_strategy: AggregatorStrategy,
        results: list[Result],
    ) -> Result:
        """
        AggrCallable

        Args:
            node (Node):
            transfer (BaseTransfer):
            aggr_strategy (AggregatorStrategy):
            results (list[Result]):

        Returns:
            ...
        """


@t.runtime_checkable
class TrainableJob(t.Protocol):
    """
    A protocol that defines functions that are valid impl to be used for local training in
    launching FLoX processes.

    Notes:
        FLoX provides default impl of this protocol via
        [LocalTrainJob][flight.jobs.local_training.LocalTrainJob] and
        [DebugLocalTrainJob][flight.jobs.local_training.DebugLocalTrainJob].
    """

    @staticmethod
    def __call__(
        node: Node,
        parent: Node,
        global_model: FloxModule,
        module_state_dict: Params,
        dataset: FloxDataset,
        transfer: BaseTransfer,
        worker_strategy: WorkerStrategy,
        trainer_strategy: TrainerStrategy,
        **train_hyper_params,
    ) -> Result:
        """

        Args:
            node ():
            parent ():
            module ():
            module_state_dict ():
            dataset ():
            transfer ():
            worker_strategy ():
            trainer_strategy ():
            **train_hyper_params ():

        Returns:

        """
