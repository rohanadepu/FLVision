from __future__ import annotations

import typing as t

from flight.jobs.protocols import TrainableJob

if t.TYPE_CHECKING:
    from flight.data import FloxDataset
    from flight.topo import Node
    from flight.nn import FloxModule
    from flight.nn.typing import Params
    from flight.runtime import Result
    from flight.runtime.transfer import BaseTransfer
    from flight.strategies import TrainerStrategy, WorkerStrategy


class LocalTrainJob(TrainableJob):
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
        # TODO: Get extra cached content from
        **train_hyper_params,
    ) -> Result:
        """Perform local training on a worker node.

        Args:
            node (Node):
            transfer (BaseTransfer): ...
            parent (Node):
            strategy (Strategy):
            module (FloxModule):
            module_state_dict (Params):
            dataset (Dataset | Subset | None):
            **train_hyper_params ():

        Returns:
            Local fitting results.
        """

        from torch.utils.data import DataLoader
        from copy import deepcopy
        from datetime import datetime

        from flight.topo.states import WorkerState
        from flight.nn.model_trainer import Trainer
        from flight.runtime import JobResult

        from proxystore.proxy import extract, Proxy

        training_start = datetime.now()

        # global_state_dict = global_model.state_dict()
        local_model = deepcopy(global_model)

        # local_model.to("mps")  # NOTE: Parameterize LATER

        if isinstance(module_state_dict, Proxy):
            module_state_dict = extract(module_state_dict)

        global_model.load_state_dict(module_state_dict)
        local_model.load_state_dict(module_state_dict)

        state = WorkerState(
            node.idx,
            global_model=global_model,
            local_model=local_model,
        )
        # print(f"{state=} (after state init)")
        state = worker_strategy.work_start(state)  # NOTE: Double-check.
        # print(f"{state=} (after `work_start`)")

        data = dataset.load(node)
        train_dataloader = DataLoader(
            data,
            batch_size=train_hyper_params.get("batch_size", 32),
            shuffle=train_hyper_params.get("shuffle", True),
        )

        optimizer = local_model.configure_optimizers()

        state, data = worker_strategy.before_training(state, data)
        # print(f"{state=} (after `before_training`)")

        # inputs, targets = next(iter(train_dataloader))
        # print(inputs)
        # print(local_model)
        # print(local_model(inputs))
        # history = pd.DataFrame.from_dict({})

        trainer = Trainer(trainer_strategy)
        history = trainer.fit(
            local_model,
            optimizer,
            train_dataloader,
            # TODO: Include `trainer_params` as an argument to
            #       this so users can easily customize Trainer.
            num_epochs=train_hyper_params.get("num_epochs", 5),
            node_state=state,
        )

        state = worker_strategy.after_training(state, optimizer)  # NOTE: Double-check.

        ################################################################################
        # TRAINING DATA POST-PROCESSING
        ################################################################################
        history["training_start"] = training_start
        history["training_end"] = datetime.now()
        history["node/idx"] = node.idx
        history["node/kind"] = node.kind.to_str()
        history["parent/idx"] = parent.idx
        history["parent/kind"] = parent.kind.to_str()

        assert state.local_model is not None
        local_params = state.local_model.state_dict()
        result = JobResult(state, node.idx, node.kind, local_params, history)

        result = worker_strategy.work_end(result)  # NOTE: Double-check.
        return transfer.report(result)

    @property
    def __name__(self) -> str:
        return "LocalTrainJob"


class DebugLocalTrainJob(TrainableJob):
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
    ):  # -> Result:
        """

        Args:
            node ():
            transfer ():
            parent ():
            strategy ():
            module (FloxModule): ...

        Returns:

        """

        from datetime import datetime

        import numpy as np
        import pandas

        from flight.topo.states import WorkerState
        from flight.runtime import JobResult

        local_module = global_model
        node_state = WorkerState(
            node.idx,
            global_model=local_module,
            local_model=local_module,
        )
        history = {
            "node/idx": [node.idx],
            "node/kind": [node.kind.to_str()],
            "parent/idx": [parent.idx],
            "parent/kind": [parent.kind.to_str()],
            "train/loss": [np.nan],
            "train/epoch": [np.nan],
            "train/batch_idx": [np.nan],
            "train/time": [datetime.now()],
            "mode": "debug",
        }
        history_df = pandas.DataFrame.from_dict(history)
        result = JobResult(
            node_state, node.idx, node.kind, global_model.state_dict(), history_df
        )
        # result = JobResult(node_state, node.idx, node.kind, np.array([0]), history_df)
        return transfer.report(result)

    @property
    def __name__(self) -> str:
        return "DebugLocalTrainJob"

