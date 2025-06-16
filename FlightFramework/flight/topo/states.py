from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

if t.TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from flight.topo import NodeID, Node
    from flight.nn import FloxModule


@dataclass
class NodeState:
    idx: NodeID
    """The ID of the ``Node`` that the ``NodeState`` corresponds with."""

    cache: dict[str, Any] = field(default_factory=dict)
    """
    A dictionary containing extra data. This can be used as a temporary "store" to 
    pass data between callbacks for custom ``Strategy`` objects."""

    def __post_init__(self):
        if type(self) is NodeState:
            raise NotImplementedError(
                "Cannot instantiate instance of ``NodeState`` (must instantiate "
                "instance of subclasses: ``AggrState`` or ``WorkerState``)."
            )

    def __iter__(self) -> Iterable[str]:
        """Returns an iterator through the state's cache."""
        return iter(self.cache)

    def __contains__(self, item) -> bool:
        """Returns `True` if `item` is stored in the state's cache, False otherwise."""
        return item in self.cache

    def __setitem__(self, key: str, value: Any) -> None:
        """Stores a piece of data (``value``) in ``self.extra_data`` using ``key``.

        Parameters:
            key (str): Key to store data in ``self.extra_data``.
            value (Any): Data to store in ``self.extra_data``.

        Examples:
            >>> state = WorkerState(...)
            >>> state["foo"] = "bar"
        """
        self.cache[key] = value

    def __getitem__(self, key: str) -> Any:
        """Retrieves the data in ``self.extra_data`` using ``key``.

        Parameters:
            key (str): Key to retrieve stored data in ``self.extra_data``.

        Examples:
            >>> state = WorkerState(...)
            >>> state["foo"] = "bar"  # Stores the data (see `__setitem__()`).
            >>> print(state["foo"])   # Gets the item.
            >>> # "foo"

        Throws:
            KeyError - Cannot retrieve data at ``key`` if it has not been set.
        """
        return self.cache[key]


class AggrState(NodeState):
    """State of an Aggregator node in a ``Topology``."""

    children: t.Iterable[Node]

    global_model: FloxModule | None

    # TODO: If there is no difference between ``AggrState`` and ``NodeState``, then do we need the former at all?
    def __init__(
        self,
        idx: NodeID,
        children: t.Iterable[Node],
        global_model: FloxModule | None,
    ) -> None:
        super().__init__(idx)
        self.children = children
        self.global_model = global_model


class WorkerState(NodeState):
    """State of a Worker node in a ``Topology``."""

    global_model: FloxModule | None = None
    """Global model."""

    local_model: FloxModule | None = None
    """Local model after local fitting/training."""

    def __init__(
        self,
        idx: NodeID,
        global_model: FloxModule | None = None,
        local_model: FloxModule | None = None,
    ):
        super().__init__(idx)
        self.global_model = global_model
        self.local_model = local_model

    def __repr__(self) -> str:
        template = "WorkerState(global_model={}, local_model={})"
        return template.format(self.global_model, self.local_model)
