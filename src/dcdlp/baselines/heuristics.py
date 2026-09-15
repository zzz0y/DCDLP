from __future__ import annotations

import networkx as nx
import numpy as np


def heuristic_scores(graph: nx.Graph, pairs, method: str) -> np.ndarray:
    degrees = dict(graph.degree())
    scores = []
    for raw_u, raw_v in pairs:
        u, v = int(raw_u), int(raw_v)
        common = set(graph.neighbors(u)) & set(graph.neighbors(v))
        if method in {"degree", "pa"}:
            value = degrees[u] * degrees[v]
        elif method == "cn":
            value = len(common)
        elif method == "aa":
            value = sum(1.0 / np.log(max(degrees[node], 2)) for node in common)
        elif method == "ra":
            value = sum(1.0 / max(degrees[node], 1) for node in common)
        elif method == "jaccard":
            union = set(graph.neighbors(u)) | set(graph.neighbors(v))
            value = len(common) / len(union) if union else 0.0
        else:
            raise ValueError(f"Unknown heuristic: {method}")
        scores.append(value)
    return np.asarray(scores, dtype=float)

