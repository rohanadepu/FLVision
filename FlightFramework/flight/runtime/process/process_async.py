from __future__ import annotations

import typing
from concurrent.futures import FIRST_COMPLETED, wait, Future
from copy import deepcopy

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from flight.topo.states import AggrState, NodeState, WorkerState
from flight.jobs import LocalTrainJob, DebugLocalTrainJob
from flight.runtime.process.process import Process

if typing.TYPE_CHECKING:
    from flight.data import FloxDataset
    from flight.topo import Topology, NodeID, Node
    from flight.nn.typing import Params
    from flight.nn import FloxModule
    from flight.runtime import Result
    from flight.runtime.runtime import Runtime
    from flight.strategies import (
        Strategy,
        ClientStrategy,
        AggregatorStrategy,
        WorkerStrategy,
        TrainerStrategy,
    )


class AsyncProcess(Process):
    """
    Asynchronous Federated Learning process. This code is very much still in 'beta' and not as robust as the
    [synchronous FL process][flight.runtime.process.process_sync.SyncProcess].

    Notes:
        Currently, this process is only compatible with two-tier ``Topology`` instances.
    """

    def __init__(
        self,
        runtime: Runtime,
        topo: Topology,
        num_global_rounds: int,
        module: FloxModule,
        dataset: FloxDataset,
        strategy: Strategy,
        *args,
    ):
        # assert that the topo is 2-tier
        if not topo.is_two_tier:
            raise ValueError(
                "Currently, FLoX only supports two-tier topologies for "
                "``AsyncProcess`` execution."
            )

        self.runtime = runtime
        self.topo = topo
        self.num_global_rounds = num_global_rounds
        self.global_model = module
        self.dataset = dataset
        self.strategy = strategy
        self.state_dict = None
        self.debug_mode = False
        self.params = self.global_model.state_dict()

        assert self.topo.leader is not None
        self.state = AggrState(
            self.topo.leader.idx, topo.children(topo.leader), self.global_model
        )

    def start(self, debug_mode: bool = False) -> tuple[FloxModule, DataFrame]:
        if debug_mode:
            self.debug_mode = debug_mode
            if self.global_model is None:
                from flight.runtime.process.debug_utils import DebugModule

                self.global_model = DebugModule()
                self.params = self.global_model.state_dict()
                self.state.global_model = self.global_model

        if not self.topo.two_tier:
            raise ValueError

        histories: list[DataFrame] = []
        worker_rounds: dict[NodeID, int] = {}
        worker_states: dict[NodeID, NodeState] = {}
        worker_state_dicts: dict[NodeID, Params] = {}

        for worker in self.topo.workers:
            worker_rounds[worker.idx] = 0
            worker_states[worker.idx] = WorkerState(worker.idx)
            worker_state_dicts[worker.idx] = deepcopy(self.global_model.state_dict())

        progress_bar = tqdm(
            total=self.num_global_rounds * self.topo.number_of_workers,
            desc="federated_fit::async",
        )
        futures = {
            self._worker_tasks(worker, self.topo.leader)
            for worker in self.topo.workers
        }

        while futures:
            dones, futures = wait(futures, return_when=FIRST_COMPLETED)
            if dones.intersection(futures):
                raise ValueError(
                    "Overlap between 'done' futures and 'to-be-done' Futures."
                )

            if len(dones) == 1:
                results = [dones.pop().result()]
            else:
                results = [done.result() for done in dones]

            for result in results:
                if worker_rounds[result.node_idx] >= self.num_global_rounds:
                    continue

                worker = self.topo[result.node_idx]
                worker_states[worker.idx] = result.node_state
                worker_state_dicts[worker.idx] = result.params

                result.history["round"] = worker_rounds[result.node_idx]
                histories.append(result.history)
                avg_params = self.strategy.aggr_strategy.aggregate_params(
                    self.state,
                    worker_states,
                    worker_state_dicts,
                    last_updated_node=worker.idx,
                )
                # avg_params = (
                #     self.global_model.state_dict()
                # )  # FIXME: Undo since this is just for testing

                self.global_model.load_state_dict(avg_params)
                self.params = avg_params

                # if not self.debug_mode:
                #     test_acc, test_loss = test_model(self.global_model)
                #     result.history["test/acc"] = test_acc
                #     result.history["test/loss"] = test_loss

                fut = self._worker_tasks(worker, self.topo.leader)
                futures.add(fut)
                worker_rounds[result.node_idx] += 1
                progress_bar.update()

        # TODO: Obviously fix this.
        return self.global_model, pd.concat(histories)

    def _worker_tasks(self, node: Node, parent: Node) -> Future[Result]:
        if self.debug_mode:
            job = DebugLocalTrainJob()
            dataset = None
        else:
            job = LocalTrainJob()
            dataset = self.runtime.proxy(self.dataset)

        return self.runtime.submit(
            job,
            node=node,
            parent=parent,
            global_model=self.runtime.proxy(deepcopy(self.global_model)),
            module_state_dict=self.runtime.proxy(self.params),
            worker_strategy=self.worker_strategy,
            trainer_strategy=self.trainer_strategy,
            dataset=dataset,
        )

    @property
    def client_strategy(self) -> ClientStrategy:
        return self.strategy.client_strategy

    @property
    def aggr_strategy(self) -> AggregatorStrategy:
        return self.strategy.aggr_strategy

    @property
    def worker_strategy(self) -> WorkerStrategy:
        return self.strategy.worker_strategy

    @property
    def trainer_strategy(self) -> TrainerStrategy:
        return self.strategy.trainer_strategy
