from __future__ import annotations

import networkx as nx
import numpy as np

from .validator import edge


def shuffled(values, rng: np.random.Generator):
    values = list(values)
    if values:
        order = rng.permutation(len(values))
        values = [values[int(index)] for index in order]
    return values


def apply_edits(
    graph: nx.Graph,
    removed: list[tuple[int, int]],
    added: list[tuple[int, int]],
) -> nx.Graph | None:
    removed = [edge(*item) for item in removed]
    added = [edge(*item) for item in added]
    if any(u == v for u, v in added) or len(set(added)) != len(added):
        return None
    if any(not graph.has_edge(*item) for item in removed):
        return None
    if any(graph.has_edge(*item) and item not in removed for item in added):
        return None
    result = graph.copy()
    result.remove_edges_from(removed)
    if any(result.has_edge(*item) for item in added):
        return None
    result.add_edges_from(added)
    return result

