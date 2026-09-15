from __future__ import annotations

import networkx as nx
import numpy as np

from .candidate_search import apply_edits, shuffled
from .validator import EditLog, InterventionError, common_neighbor_set, edge, validate_intervention


def _single_degree_step(
    graph: nx.Graph, u: int, v: int, endpoint: int, direction: int,
    forbidden: set[tuple[int, int]], rng: np.random.Generator,
) -> tuple[nx.Graph, tuple[int, int], tuple[int, int]]:
    base_cn = common_neighbor_set(graph, u, v)
    other = v if endpoint == u else u
    if direction > 0:
        for x, y in shuffled(graph.edges(), rng):
            for donor, retained in ((x, y), (y, x)):
                if graph.degree(donor) <= 1 or retained in {u, v}:
                    continue
                added = edge(endpoint, retained)
                if graph.has_edge(*added) or graph.has_edge(other, retained) or added in forbidden:
                    continue
                candidate = apply_edits(graph, [edge(donor, retained)], [added])
                if candidate is not None and common_neighbor_set(candidate, u, v) == base_cn:
                    return candidate, edge(donor, retained), added
    else:
        if graph.degree(endpoint) <= 1:
            raise InterventionError("DISCONNECT_ENDPOINT", "Endpoint degree cannot be reduced")
        for retained in shuffled(graph.neighbors(endpoint), rng):
            if graph.has_edge(other, retained):
                continue
            for recipient in shuffled(graph.nodes(), rng):
                added = edge(recipient, retained)
                if recipient in {endpoint, retained} or graph.has_edge(*added) or added in forbidden:
                    continue
                candidate = apply_edits(graph, [edge(endpoint, retained)], [added])
                if candidate is not None and common_neighbor_set(candidate, u, v) == base_cn:
                    return candidate, edge(endpoint, retained), added
    raise InterventionError("NO_CANDIDATE", "No CN-set-preserving edge transfer was found")


def intervene_degree(
    graph: nx.Graph, u: int, v: int, endpoint: str, delta: int, seed: int = 0,
    forbidden_edges: set[tuple[int, int]] | None = None, pair_id: str = "",
) -> tuple[nx.Graph, EditLog]:
    if endpoint not in {"u", "v"}:
        raise ValueError("endpoint must be 'u' or 'v'")
    if delta == 0 or abs(delta) > 2:
        raise ValueError("Degree delta must be one of -2, -1, 1, 2")
    if graph.has_edge(u, v):
        graph = graph.copy()
        graph.remove_edge(u, v)
    original = graph.copy()
    forbidden = {edge(*item) for item in (forbidden_edges or set())} | {edge(u, v)}
    rng = np.random.default_rng(seed)
    current = original
    target = u if endpoint == "u" else v
    for _ in range(abs(delta)):
        current, _, _ = _single_degree_step(current, u, v, target, 1 if delta > 0 else -1, forbidden, rng)
    original_edges = {edge(*item) for item in original.edges()}
    current_edges = {edge(*item) for item in current.edges()}
    kind = f"degree_{endpoint}_{'plus' if delta > 0 else 'minus'}"
    log = EditLog(pair_id or f"{u}-{v}", u, v, kind, delta,
                  sorted(original_edges - current_edges), sorted(current_edges - original_edges), seed=seed)
    validate_intervention(original, current, log, forbidden)
    log.valid = True
    return current, log

