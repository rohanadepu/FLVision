import datetime
import typing as t

from pandas import DataFrame

import flight.strategies as strats
from flight.data import FloxDataset
from flight.topo import Topology
from flight.nn import FloxModule
from flight.nn.typing import Kind
from flight.runtime.launcher import (
    GlobusComputeLauncher,
    Launcher,
    LocalLauncher,
    ParslLauncher,
)
from flight.runtime.process.process import Process
from flight.runtime.process.process_async import AsyncProcess
from flight.runtime.process.process_sync import SyncProcess
from flight.runtime.process.process_sync_v2 import SyncProcessV2
from flight.runtime.runtime import Runtime
from flight.runtime.transfer import BaseTransfer, ProxyStoreTransfer, RedisTransfer


def create_launcher(kind: str, **launcher_cfg) -> Launcher:
    match kind:
        case "thread":
            return LocalLauncher(
                pool="thread", n_workers=launcher_cfg.get("max_workers", 3)
            )
        case "process":
            return LocalLauncher(
                pool="process", n_workers=launcher_cfg.get("max_workers", 3)
            )
        case "globus-compute":
            return GlobusComputeLauncher()
        case "parsl":
            return ParslLauncher(launcher_cfg)
        case _:
            raise ValueError("Illegal value for argument `kind`.")


def federated_fit(
    topo: Topology,
    module: FloxModule,
    datasets: FloxDataset,
    num_global_rounds: int,
    # Strategy arguments.
    strategy: strats.Strategy | str | None = None,
    client_strategy: strats.ClientStrategy | None = None,
    aggr_strategy: strats.AggregatorStrategy | None = None,
    worker_strategy: strats.WorkerStrategy | None = None,
    trainer_strategy: strats.TrainerStrategy | None = None,
    # Process arguments.
    kind: Kind = "sync",
    launcher_kind: str = "process",
    launcher_cfg: dict[str, t.Any] | None = None,
    debug_mode: bool = False,
    logging: bool = False,
    redis_ip_address: str = "127.0.0.1",
) -> tuple[FloxModule, DataFrame]:
    """

    Args:
        topo (Topology):
        module (FloxModule):
        datasets (FloxDataset):
        num_global_rounds (int):
        strategy (Strategy | str | None):
        client_strategy (strats.ClientStrategy): ...
        aggr_strategy (strats.AggregatorStrategy): ...
        worker_strategy (strats.WorkerStrategy): ...
        trainer_strategy (strats.TrainerStrategy): ...
        kind (Kind):
        launcher_kind (str):
        launcher_cfg (dict[str, t.Any] | None):
        debug_mode (bool): ...

    Returns:
        The trained global module hosted on the leader of `topo`.
        The history metrics from training.
    """
    launcher_cfg = dict() if launcher_cfg is None else launcher_cfg
    launcher = create_launcher(launcher_kind, **launcher_cfg)
    if isinstance(launcher, GlobusComputeLauncher):
        transfer = ProxyStoreTransfer(topo)
    elif isinstance(launcher, ParslLauncher):
        print(f"Yadu : setting RedisTransfer to {redis_ip_address}")
        transfer = RedisTransfer(ip_address=redis_ip_address)
    else:
        transfer = BaseTransfer()

    runtime = Runtime(launcher, transfer)
    parsed_strategy = parse_strategy_args(
        strategy=strategy,
        client_strategy=client_strategy,
        aggr_strategy=aggr_strategy,
        worker_strategy=worker_strategy,
        trainer_strategy=trainer_strategy,
    )

    # runner = runner_factory.build(kind, ...)
    # runner.start()
    process: Process
    match kind:
        case "sync":
            process = SyncProcess(
                runtime=runtime,
                topo=topo,
                num_global_rounds=num_global_rounds,
                module=module,
                dataset=datasets,
                strategy=parsed_strategy,
            )

        case "sync-v2":
            process = SyncProcessV2(
                runtime=runtime,
                topo=topo,
                global_rounds=num_global_rounds,
                module=module,
                dataset=datasets,
                strategy=parsed_strategy,
                logging=logging,
            )

        case "async":
            process = AsyncProcess(
                runtime=runtime,
                topo=topo,
                num_global_rounds=num_global_rounds,
                module=module,
                dataset=datasets,
                strategy=parsed_strategy,
            )
        case _:
            raise ValueError("Illegal value for the strategy `kind` parameter.")

    start_time = datetime.datetime.now()
    trained_module, history = process.start(debug_mode=debug_mode)
    try:
        history["train/rel_time"] = history["train/time"] - start_time
        history["train/rel_time"] = history["train/rel_time"].dt.total_seconds()
    except KeyError as err:
        print(history.head())

    if isinstance(runtime.launcher, ParslLauncher):
        runtime.launcher.executor.shutdown()

    return trained_module, history


def parse_strategy_args(
    strategy: strats.Strategy | str | None,
    client_strategy: strats.ClientStrategy | None,
    aggr_strategy: strats.AggregatorStrategy | None,
    worker_strategy: strats.WorkerStrategy | None,
    trainer_strategy: strats.TrainerStrategy | None,
    **kwargs,
) -> strats.Strategy:
    if isinstance(strategy, strats.Strategy):
        return strategy

    if isinstance(strategy, str):
        return strats.load_strategy(strategy, **kwargs)

    if strategy is not None:
        raise ValueError(
            "Argument ``strategy`` is not a legal value. Must be either a ``Strategy``, "
            "a supported string value, or ``None``. "
        )

    # If the user provided each individual strategy implementations, then we must first check and confirm
    # that none of those arguments are ``None``. If they are not, then we can package them as a single
    # ``Strategy`` and return that.
    strategies = [client_strategy, aggr_strategy, worker_strategy, trainer_strategy]
    for _name, _strategy in zip(["client", "aggr", "worker", "trainer"], strategies):
        if _strategy is None:
            cls_name = "aggregator" if _name == "aggr" else _name
            cls_name = cls_name.title()
            raise ValueError(
                f"Argument `{_name}_strategy` must be a class that implements protocol ``{cls_name}``."
            )

    # Explicit asserts to satisfy `mypy`.
    assert client_strategy is not None
    assert aggr_strategy is not None
    assert worker_strategy is not None
    assert trainer_strategy is not None

    return strats.Strategy(
        client_strategy=client_strategy,
        aggr_strategy=aggr_strategy,
        worker_strategy=worker_strategy,
        trainer_strategy=trainer_strategy,
    )
