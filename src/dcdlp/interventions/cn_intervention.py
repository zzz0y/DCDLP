from __future__ import annotations

import networkx as nx
import numpy as np

from .candidate_search import apply_edits, shuffled
from .validator import EditLog, InterventionError, edge, validate_intervention


def _single_cn_step(
    graph: nx.Graph, u: int, v: int, direction: int,
    forbidden: set[tuple[int, int]], rng: np.random.Generator,
) -> tuple[nx.Graph, list[tuple[int, int]], list[tuple[int, int]]]:
    base_cn = len(set(graph.neighbors(u)) & set(graph.neighbors(v)))
    orientations = shuffled([(u, v), (v, u)], rng)
    if direction > 0:
        for anchor, endpoint in orientations:
            candidates_w = set(graph.neighbors(anchor)) - set(graph.neighbors(endpoint)) - {endpoint}
            for w in shuffled(candidates_w, rng):
                for x in shuffled(graph.neighbors(endpoint), rng):
                    for a, b in shuffled(graph.edges(), rng):
                        if a == w:
                            y = b
                        elif b == w:
                            y = a
                        else:
                            continue
                        removed = [edge(endpoint, x), edge(w, y)]
                        added = [edge(endpoint, w), edge(x, y)]
                        if any(item in forbidden for item in added):
                            continue
                        candidate = apply_edits(graph, removed, added)
                        if candidate is None:
                            continue
                        if dict(candidate.degree()) != dict(graph.degree()):
                            continue
                        cn = len(set(candidate.neighbors(u)) & set(candidate.neighbors(v)))
                        if cn == base_cn + 1:
                            return candidate, removed, added
    else:
        common = set(graph.neighbors(u)) & set(graph.neighbors(v))
        for w in shuffled(common, rng):
            for anchor, endpoint in orientations:
                if not graph.has_edge(endpoint, w):
                    continue
                for x, y in shuffled(graph.edges(), rng):
                    removed = [edge(endpoint, w), edge(x, y)]
                    added = [edge(endpoint, x), edge(w, y)]
                    if any(item in forbidden for item in added):
                        continue
                    candidate = apply_edits(graph, removed, added)
                    if candidate is None:
                        continue
                    if dict(candidate.degree()) != dict(graph.degree()):
                        continue
                    cn = len(set(candidate.neighbors(u)) & set(candidate.neighbors(v)))
                    if cn == base_cn - 1:
                        return candidate, removed, added
    raise InterventionError("NO_CANDIDATE", "No degree-preserving two-switch was found")


def intervene_cn(
    graph: nx.Graph, u: int, v: int, delta: int, seed: int = 0,
    forbidden_edges: set[tuple[int, int]] | None = None, pair_id: str = "",
) -> tuple[nx.Graph, EditLog]:
    if delta == 0 or abs(delta) > 2:
        raise ValueError("CN delta must be one of -2, -1, 1, 2")
    if graph.has_edge(u, v):
        graph = graph.copy()
        graph.remove_edge(u, v)
    original = graph.copy()
    forbidden = {edge(*item) for item in (forbidden_edges or set())} | {edge(u, v)}
    rng = np.random.default_rng(seed)
    current = original
    for _ in range(abs(delta)):
        current, _, _ = _single_cn_step(current, u, v, 1 if delta > 0 else -1, forbidden, rng)
    original_edges = {edge(*item) for item in original.edges()}
    current_edges = {edge(*item) for item in current.edges()}
    kind = "cn_plus" if delta > 0 else "cn_minus"
    log = EditLog(pair_id or f"{u}-{v}", u, v, kind, delta,
                  sorted(original_edges - current_edges), sorted(current_edges - original_edges), seed=seed)
    validate_intervention(original, current, log, forbidden)
    log.valid = True
    return current, log

