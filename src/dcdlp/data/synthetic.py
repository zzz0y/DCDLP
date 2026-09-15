from __future__ import annotations

import math

import networkx as nx
import numpy as np

from .loaders import GraphDataset, canonical_edges


MECHANISMS = {
    "pa-only": (2.0, 0.0, 0.0),
    "cn-only": (0.0, 2.0, 0.0),
    "mixed": (1.0, 1.0, 0.0),
    "feature": (0.0, 0.0, 2.0),
    "null": (0.0, 0.0, 0.0),
    "degree-dominant": (2.0, 0.5, 0.0),
    "cn-dominant": (0.5, 2.0, 0.0),
}


def _split_graph(graph: nx.Graph, features: np.ndarray, name: str, seed: int) -> GraphDataset:
    rng = np.random.default_rng(seed)
    edges = canonical_edges(graph.edges())
    rng.shuffle(edges)
    n_test = max(1, int(0.10 * len(edges)))
    n_valid = max(1, int(0.05 * len(edges)))
    test = edges[:n_test]
    valid = edges[n_test:n_test + n_valid]
    train = edges[n_test + n_valid:]
    dataset = GraphDataset(name, graph.number_of_nodes(), features, train, valid, test, edges)
    dataset.validate()
    return dataset


def make_smoke_dataset(seed: int = 0) -> GraphDataset:
    graph = nx.barabasi_albert_graph(48, 2, seed=seed)
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(48, 16)).astype(np.float32)
    return _split_graph(graph, features, "smoke", seed)


def make_pa_tc_dataset(
    mechanism: str = "mixed", seed: int = 0, num_nodes: int = 5000,
    average_degree: float = 10.0, feature_dim: int = 32,
    candidate_size: int = 50_000,
) -> GraphDataset:
    if mechanism not in MECHANISMS:
        raise ValueError(f"Unknown mechanism {mechanism}; choose from {sorted(MECHANISMS)}")
    alpha, beta, gamma = MECHANISMS[mechanism]
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(num_nodes, feature_dim)).astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    initial_n = min(30, num_nodes)
    graph = nx.random_labeled_tree(initial_n, seed=seed)
    graph.add_nodes_from(range(num_nodes))
    target_edges = int(num_nodes * average_degree / 2)
    while graph.number_of_edges() < target_edges:
        size = min(candidate_size, max(1000, target_edges - graph.number_of_edges()))
        left = rng.integers(0, num_nodes, size=size)
        right = rng.integers(0, num_nodes, size=size)
        candidates = []
        for u, v in zip(left, right):
            u, v = int(u), int(v)
            if u != v and not graph.has_edge(u, v):
                candidates.append((min(u, v), max(u, v)))
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            continue
        degree_signal = np.asarray([
            math.log1p(graph.degree(u)) + math.log1p(graph.degree(v)) for u, v in candidates
        ])
        cn_signal = np.asarray([math.log1p(len(list(nx.common_neighbors(graph, u, v)))) for u, v in candidates])
        feature_signal = np.asarray([float(features[u] @ features[v]) for u, v in candidates])
        signals = []
        for signal in (degree_signal, cn_signal, feature_signal):
            std = signal.std()
            signals.append((signal - signal.mean()) / (std if std > 1e-8 else 1.0))
        logits = alpha * signals[0] + beta * signals[1] + gamma * signals[2]
        logits -= logits.max()
        probabilities = np.exp(np.clip(logits, -40, 0))
        probabilities /= probabilities.sum()
        graph.add_edge(*candidates[int(rng.choice(len(candidates), p=probabilities))])
    return _split_graph(graph, features, f"synthetic-{mechanism}", seed)
