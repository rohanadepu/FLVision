import functools
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

import matplotlib.pyplot as plt
import networkx as nx
import yaml
from matplotlib.axes import Axes

from flight.topo.node import Node, NodeID, NodeKind

REQUIRED_ATTRS: set[str] = {
    "kind",
    "globus_compute_endpoint",
    "proxystore_endpoint",
    "children",
}

# Valid programs for drawing using `graphviz`
PROGS = [
    "dot",
    "neato",
    "fdp",
    "sfdp",
    "circo",
    "twopi",
    "nop",
    "nop2",
    "osage",
    "patchwork",
]

"""
TODO: Make an `OmegaConf` standard.
"""


class Topology:
    topo: nx.DiGraph
    node_counter: int
    leader: Node | None

    def __init__(self, topo: nx.DiGraph | None = None, src: Path | str | None = None):
        """

        Args:
            topo (nx.DiGraph | None): The topology (defined as a NetworkX
                ``nx.DiGraph``) of the Topology. If none is provided, then the
                Topology is initialized in "interactive" mode. This means you can
                iteratively add nodes and edges to the Topology using
                ``Topology.add_node()`` and ``Topology.add_edge()``. This is *not*
                recommended. Defaults to ``None``.
            src (Path | str | None): This identifies the source file that was used to
                define the Topology network. This should only be used by the file
                constructor functions (e.g., `from_yaml()`) and should not be used
                by the user. Defaults to ``None``.
        """
        self.node_counter: int = 0
        self.src = src
        self.mode = "interactive" if src is None else "networkx"

        if topo is None:
            # By default (i.e., `topo is None`),
            self.topo = nx.DiGraph()
            self.node_counter += self.topo.number_of_nodes()
            self.leader = None
        else:
            self.topo = topo
            if not self.validate_topo():
                raise ValueError("Illegal topology!")

            found_leader = False
            for idx, data in self.topo.nodes(data=True):
                if data["kind"] is NodeKind.LEADER:
                    if not found_leader:
                        self.leader = Node(
                            idx=idx,
                            kind=data["kind"],
                            globus_compute_endpoint=data["globus_compute_endpoint"],
                            proxystore_endpoint=data["proxystore_endpoint"],
                            # children_idx: Sequence[NodeID] | None
                        )
                        found_leader = True
                    else:
                        raise ValueError(
                            "A legal Topology cannot have more than one leader."
                        )
            if not found_leader:
                raise ValueError("A legal Topology must have a leader.")

    def add_node(
        self,
        kind: NodeKind | str,
        globus_compute_endpoint_id: UUID | None = None,
        proxystore_endpoint_id: UUID | None = None,
    ) -> Node:
        if isinstance(kind, str):
            kind = NodeKind.from_str(kind)

        if kind is NodeKind.LEADER and self.leader is not None:
            raise ValueError("A leader node has already been established.")

        idx = self.node_counter
        self.node_counter += 1
        self.topo.add_node(
            idx,
            kind=kind,
            globus_compute_endpoint=globus_compute_endpoint_id,
            proxystore_endpoint=proxystore_endpoint_id,
        )
        return self[idx]

    def add_edge(self, u: NodeID, v: NodeID, **attrs) -> None:
        """

        Args:
            u (NodeID):
            v (NodeID):
            **attrs ():

        Throws:
            ValueError - Cannot add edges between nodes that do not already exist in 
            the ``Topology`` instance.
        """
        if any([u not in self.topo.nodes, v not in self.topo.nodes]):
            raise ValueError(
                "`Topology` does not support adding edges between nodes that do not "
                "already exist. Try adding each node first."
            )
        self.topo.add_edge(u, v, **attrs)

    def draw(
        self,
        color_by_kind: bool = True,
        with_labels: bool = True,
        label_color: str = "white",
        prog: str = "dot",
        node_kind_attrs: dict[NodeKind, dict[str, Any]] | None = None,
        show_axis_border: bool = False,
        ax: Axes | None = None,
    ) -> Axes:
        """
        Draws the topo using Matplotlib. The nodes are organized as a tree with the proper
        hierarchy based on depth from the Leader node (root).

        Args:
            color_by_kind (bool): Color nodes by kind, if True.
            with_labels (bool): Display labels of nodes, if True.
            label_color (str): Color of labels.
            prog (str): How the topology is organized. Leave alone for the default behavior
                of displaying it as a tree. This is passed into the `prog` argument for
                ``networkx.nx_agraph.graphviz_layout()``.
            node_kind_attrs (): Determines how node attributes should be plotted. By default,
                nodes will be colored and marked by kind.
            show_axis_border (bool): Show the border along the axis if True; defaults to False.
            ax (Axes | None): Axes object to draw onto. If none is provided, then one
                will be created.

        Returns:
            Axes object that was drawn onto.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if not show_axis_border:
            ax.axis("off")

        # TODO: We may want to remove this as a requirement. It produces nice "tree" positions
        # of the nodes. But it introduces a pretty restrictive dependency.
        if prog in PROGS:
            pos = nx.nx_pydot.pydot_layout(self.topo, prog=prog)
        else:
            pos = nx.spring_layout(self.topo)

        if not color_by_kind:
            nx.draw(self.topo, pos, with_labels=with_labels, ax=ax)
            return ax

        if self.leader is None:
            raise ValueError(
                "There is no leader in the Topology. This is likely because no topology "
                "has been created via interactive mode."
            )
        leader = [self.leader.idx]
        aggregators = list(aggr.idx for aggr in self.aggregators)
        workers = list(worker.idx for worker in self.workers)

        if node_kind_attrs is None:
            node_kind_attrs = {
                NodeKind.LEADER: {"color": "red", "shape": "D", "size": 300},
                NodeKind.AGGREGATOR: {"color": "green", "shape": "s", "size": 300},
                NodeKind.WORKER: {"color": "blue", "shape": "o", "size": 300},
            }

        kinds = [NodeKind.LEADER, NodeKind.AGGREGATOR, NodeKind.WORKER]
        node_sets = [leader, aggregators, workers]
        for kind, nodes in zip(kinds, node_sets):
            nx.draw_networkx_nodes(
                self.topo,
                pos,
                nodes,
                node_color=node_kind_attrs[kind]["color"],
                node_shape=node_kind_attrs[kind]["shape"],
                node_size=node_kind_attrs[kind]["size"],
                label=kind.to_str(),
                ax=ax,
            )

        nx.draw_networkx_edges(self.topo, pos)
        if with_labels:
            nx.draw_networkx_labels(self.topo, pos, font_color=label_color, ax=ax)

        return ax

    def validate_topo(self) -> bool:
        # STEP 1: Confirm that there only exists ONE leader in the Topology.
        leaders = []
        for idx, data in self.topo.nodes(data=True):
            if data["kind"] is NodeKind.LEADER:
                leaders.append(idx)
        if len(leaders) != 1:
            return False

        # STEP 2: Confirm that the Topology has a tree topology.
        if not nx.is_tree(self.topo):
            return False

        # TODO: We will also need to confirm that worker nodes are leaves in the tree.
        ...

        return True

    def parent(self, node: Node | NodeID | int) -> Node:
        if isinstance(node, Node):
            idx = node.idx
        else:
            idx = node.idx

        if idx == self.leader.idx:
            raise ValueError("Leader node has no parent.")

        parent_idx = list(self.topo.predecessors(idx))
        if len(parent_idx) != 1:
            raise ValueError(
                f"A node must have exactly 1 parent (except client); illegal topology -- {parent_idx=}"
            )
        return self[parent_idx[0]]

    def children(self, node: Node | NodeID | int) -> Iterator[Node]:
        if isinstance(node, Node):
            idx = node.idx
        else:
            idx = node
        gce = "globus_compute_endpoint"
        pse = "proxystore_endpoint"
        for child_idx in self.topo.successors(idx):
            yield self[child_idx]

    def get_kind(self, node: Node | NodeID | int) -> NodeKind:
        if isinstance(node, Node):
            idx = node.idx
        else:
            idx = node
        return self.topo.nodes[idx]["kind"]

    ####################################################################################

    @staticmethod
    def from_dict(content: dict[str, Any], src: Path | str | None = None) -> "Topology":
        """
        Imports a ``dict`` object to create a Topology network.

        Args:
            content (dict[str, Any]): Dictionary that defines the Topology network.
            src (Path | str | None): Identifies the source file used to define
                the Topology network. This should **not** be used by users. It is used by
                ``Topology`` class methods that are built on top of this method
                (e.g., ``Topology.from_yaml()``).

        Examples:
            >>> topo = {
            >>>     0: {
            >>>         'kind': 'leader',
            >>>         'globus_compute_endpoint': LEADER_GC_UUID,
            >>>         'proxystore_endpoint': LEADER_PS_UUID,
            >>>         'children': [1, 2]
            >>>     },
            >>>     1: {
            >>>         'kind': 'worker',
            >>>         'globus_compute_endpoint': w1_gc_endpoint,
            >>>         'proxystore_endpoint': w1_ps_endpoint,
            >>>         'children': []
            >>>     },
            >>>     2: {
            >>>         'kind': 'worker',
            >>>         'globus_compute_endpoint': w2_gc_endpoint,
            >>>         'proxystore_endpoint': w2_ps_endpoint,
            >>>         'children': []
            >>>     }
            >>> }
            >>> topo = Topology.from_dict(topo)
            >>> print(topo.number_of_workers) # outputs 2

        Returns:
            An instance of a Topology.
        """
        topo = nx.DiGraph()

        # STEP 1: Add the nodes with their attributes --- ignore children for now.
        for node_idx, values in content.items():
            for attr in REQUIRED_ATTRS:
                if attr not in values:
                    raise ValueError(
                        f"Node {node_idx} does not have required attribute: `{attr}`."
                    )

            topo.add_node(
                node_idx,
                kind=NodeKind.from_str(values["kind"]),
                globus_compute_endpoint=values["globus_compute_endpoint"],
                proxystore_endpoint=values["proxystore_endpoint"],
            )
            for extra_attr in set(values) - REQUIRED_ATTRS:
                topo.nodes[node_idx][extra_attr] = values[extra_attr]

        # STEP 2: Add the edges from the children attribute.
        for node_idx, values in content.items():
            for child in values["children"]:
                topo.add_edge(node_idx, child)

        return Topology(topo=topo, src=src)

    @staticmethod
    def from_json(path: Path | str) -> "Topology":
        """Imports a .json file as a Topology.

        Examples:
            >>> topo = Topology.from_json("my_topo.json")

        Args:
            path (Path | str): Must be a .json file defining a Topology topology.

        Returns:
            An instance of a Topology.
        """
        # TODO: Figure out how to address the issue of JSON requiring string keys for `from_json()`.
        with open(path) as f:
            content = json.load(f)
        return Topology.from_dict(content, src=path)

    @staticmethod
    def from_yaml(path: Path | str) -> "Topology":
        """Imports a `.yaml` file as a Topology.

        Examples:
            >>> topo = Topology.from_yaml("my_topo.yaml")

        Args:
            path (Path | str): Must be a .yaml file defining a Topology.

        Returns:
            An instance of a Topology.
        """
        with open(path) as f:
            content = yaml.safe_load(f)
        return Topology.from_dict(content, src=path)

    ####################################################################################

    @property
    def globus_compute_ready(self) -> bool:
        """
        True if the Topology instance has all necessary endpoints to be run across
        Globus Compute; False otherwise.
        """
        # TODO: The leader does NOT need a Globus Compute endpoint.
        key = "globus_compute_endpoint"
        for _idx, data in self.topo.nodes(data=True):
            value = data[key]
            if any([value is None, isinstance(value, UUID) is False]):
                return False
        return True

    @property
    def proxystore_ready(self) -> bool:
        """
        This property informs users of whether their `Topology` has all the necessary
        information to support data transmission over Proxystore (`True`) or not
        (`False`). Proxystore just requires that each node in the Topology has its own
        `proxystore_endpoint`.

        It is worth noting that Proxystore is necessary to transmit mid-to-large size
        model (roughly > 5MB in size) with Globus Compute.
        """
        key = "proxystore_endpoint"
        for _idx, data in self.topo.nodes(data=True):
            value = data[key]

            try:
                if any([value is None, UUID(value) is False]):
                    return False
            except ValueError:
                return False
        return True

    # @property
    # def leader(self):
    #     return self.nodes(by_kind=NodeKind.LEADER)

    @property
    def aggregators(self) -> Iterator[Node]:
        """
        The aggregator nodes of the Topology.

        Returns:
            Generator[Node]
        """
        return self.nodes(by_kind=NodeKind.AGGREGATOR)

    @property
    def workers(self) -> Iterator[Node]:
        """
        The worker nodes of the Topology.

        Returns:
            Generator[Node]
        """
        return self.nodes(by_kind=NodeKind.WORKER)

    @functools.cached_property
    def is_two_tier(self) -> bool:
        """
        Whether the topology is a two-tier topology or not.

        Notes:
            This must be ``True`` for asynchronous FL processes.

        Returns:
            ``True`` if the topology is two-tier, ``False`` otherwise.
        """
        assert self.leader is not None, "There must be a leader node in the topology."
        leader_id = self.leader.idx
        assert leader_id is not None, "Leader ID cannot be `None`."
        tree = nx.bfs_tree(self.topo, leader_id, depth_limit=1)
        return tree.number_of_nodes() == self.topo.number_of_nodes()

    @functools.cached_property
    def number_of_aggregators(self) -> int:
        """The number of aggregator nodes in the Topology."""
        return len(list(self.aggregators))

    @functools.cached_property
    def number_of_workers(self) -> int:
        """The number of worker nodes in the Topology."""
        return len(list(self.workers))

    @functools.cached_property
    def two_tier(self) -> bool:
        assert self.leader is not None
        for worker in self.workers:
            if not self.topo.has_edge(self.leader.idx, worker.idx):
                return False
        return True

    def nodes(self, by_kind: NodeKind | None = None) -> Iterator[Node]:
        for idx, data in self.topo.nodes(data=True):
            if by_kind is not None and data["kind"] != by_kind:
                continue
            yield Node(
                idx=idx,
                kind=data["kind"],
                globus_compute_endpoint=data["globus_compute_endpoint"],
                proxystore_endpoint=data["proxystore_endpoint"],
                # children_idx: Sequence[NodeID] | None
            )

    ####################################################################################

    def __repr__(self):
        return f"Topology(`{self.mode}`)"

    def __getitem__(self, idx: NodeID) -> Node:
        return Node(idx, **self.topo.nodes[idx])
