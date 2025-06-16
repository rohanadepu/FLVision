from __future__ import annotations

import typing
from dataclasses import dataclass, field

from proxystore.proxy import Proxy

if typing.TYPE_CHECKING:
    from pandas import DataFrame

    from flight.topo import NodeID, NodeKind
    from flight.topo.states import NodeState
    from flight.nn.typing import Params

import random


@dataclass
class JobResult:
    """
    A simple dataclass that is returned by jobs executed on Aggregator and Worker
    nodes in a ``Topology``.

    Aggregators and Worker nodes have to return the same type of object to support hierarchical execution.
    """

    node_state: NodeState
    """The state of the ``Topology`` node based on its kind."""

    node_idx: NodeID
    """The ID of the ``Topology`` node."""

    node_kind: NodeKind
    """The kind of the ``Topology`` node."""

    params: Params
    """The ``Params`` of the PyTorch global_model (either aggregated or trained locally)."""

    history: DataFrame
    """The history of results."""

    cache: dict[str, typing.Any] = field(default_factory=dict)
    """Miscellaneous data to be returned as part of the ``JobResult``."""

    def __hash__(self):
        return random.randint(0, 1000000)


Result: typing.TypeAlias = JobResult | Proxy[JobResult]
