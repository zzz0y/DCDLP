from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np


def _edge_set(edges: Iterable[Iterable[int]]) -> set[tuple[int, int]]:
    return {tuple(sorted((int(u), int(v)))) for u, v in edges}


def uniform_negative_sampling(
    num_nodes: int, forbidden_edges: Iterable[Iterable[int]], count: int, seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    forbidden = _edge_set(forbidden_edges)
    chosen: set[tuple[int, int]] = set()
    max_possible = num_nodes * (num_nodes - 1) // 2 - len(forbidden)
    if count > max_possible:
        raise ValueError(f"Requested {count} negatives but only {max_possible} non-edges exist")
    tries = 0
    while len(chosen) < count:
        u, v = map(int, rng.integers(0, num_nodes, size=2))
        edge = tuple(sorted((u, v)))
        if u != v and edge not in forbidden:
            chosen.add(edge)
        tries += 1
        if tries > max(10_000, count * 1000):
            raise RuntimeError("Uniform negative sampling exceeded retry budget")
    return np.asarray(sorted(chosen), dtype=np.int64)


def degree_corrected_sampling(
    graph: nx.Graph, forbidden_edges: Iterable[Iterable[int]], count: int, seed: int,
) -> np.ndarray:
    """Native degree-corrected sampler using degree-proportional endpoint draws."""
    rng = np.random.default_rng(seed)
    nodes = np.asarray(sorted(graph.nodes()), dtype=np.int64)
    weights = np.asarray([max(graph.degree(int(node)), 1) for node in nodes], dtype=float)
    weights /= weights.sum()
    forbidden = _edge_set(forbidden_edges)
    chosen: set[tuple[int, int]] = set()
    tries = 0
    while len(chosen) < count:
        u, v = map(int, rng.choice(nodes, size=2, replace=True, p=weights))
        edge = tuple(sorted((u, v)))
        if u != v and edge not in forbidden:
            chosen.add(edge)
        tries += 1
        if tries > max(20_000, count * 5000):
            raise RuntimeError("Degree-corrected sampling exceeded retry budget")
    return np.asarray(sorted(chosen), dtype=np.int64)


def grouped_negatives(
    positives: np.ndarray, pool: np.ndarray, negatives_per_positive: int, seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    replace = len(pool) < negatives_per_positive
    indices = rng.choice(len(pool), size=(len(positives), negatives_per_positive), replace=replace)
    return pool[indices]

