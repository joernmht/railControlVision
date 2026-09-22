"""Build a networkx multigraph from a topology and enumerate edge endpoints per node."""

from __future__ import annotations

import networkx as nx

from rail_vision_bench.schema.models import Port, Topology


def to_networkx(topology: Topology) -> nx.MultiGraph[str]:
    """Convert a topology into a multigraph keyed by element ids.

    Edges whose endpoints are not nodes of the topology are skipped; the
    validator reports them as dangling references.

    Args:
        topology: The nodes and edges.

    Returns:
        A multigraph with node attributes ``kind``/``label`` and, per edge
        (key = edge id), ``port_a``/``port_b``/``kind``.
    """
    graph: nx.MultiGraph[str] = nx.MultiGraph()
    for node in topology.nodes:
        graph.add_node(node.id, kind=node.kind, label=node.label)
    for edge in topology.edges:
        if edge.a.node in graph and edge.b.node in graph:
            graph.add_edge(
                edge.a.node,
                edge.b.node,
                key=edge.id,
                port_a=edge.a.port,
                port_b=edge.b.port,
                kind=edge.kind,
            )
    return graph


def endpoint_ports(topology: Topology) -> dict[str, list[tuple[Port, str]]]:
    """List the (port, edge id) endpoints attached to every referenced node.

    Both ends of every edge are counted, in edge order; node ids that do not
    exist in the topology appear as keys too, so degree and reference checks can
    share one pass.

    Args:
        topology: The nodes and edges.

    Returns:
        Mapping from node id to its endpoints.
    """
    endpoints: dict[str, list[tuple[Port, str]]] = {}
    for edge in topology.edges:
        for ref in (edge.a, edge.b):
            endpoints.setdefault(ref.node, []).append((ref.port, edge.id))
    return endpoints
