from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx


FAILURE_CODES = {
    "NO_CANDIDATE", "DUPLICATE_EDGE", "FORBIDDEN_EDGE", "DISCONNECT_ENDPOINT",
    "CN_CONSTRAINT_FAILED", "DEGREE_CONSTRAINT_FAILED", "MAX_TRIES",
}


class InterventionError(RuntimeError):
    def __init__(self, code: str, message: str = "") -> None:
        if code not in FAILURE_CODES:
            raise ValueError(f"Unknown failure code: {code}")
        self.code = code
        super().__init__(f"{code}: {message}" if message else code)


def edge(u: int, v: int) -> tuple[int, int]:
    return min(int(u), int(v)), max(int(u), int(v))


def common_neighbor_set(graph: nx.Graph, u: int, v: int) -> set[int]:
    return set(graph.neighbors(u)) & set(graph.neighbors(v))


@dataclass
class EditLog:
    pair_id: str
    u: int
    v: int
    intervention_type: str
    requested_delta: int
    removed_edges: list[tuple[int, int]] = field(default_factory=list)
    added_edges: list[tuple[int, int]] = field(default_factory=list)
    valid: bool = False
    failure_code: str = ""
    seed: int = 0


def validate_intervention(
    original: nx.Graph,
    counterfactual: nx.Graph,
    log: EditLog,
    forbidden_edges: set[tuple[int, int]] | None = None,
) -> None:
    forbidden_edges = {edge(*item) for item in (forbidden_edges or set())}
    u, v = log.u, log.v
    if isinstance(counterfactual, (nx.MultiGraph, nx.MultiDiGraph)):
        raise InterventionError("DUPLICATE_EDGE", "Counterfactual must be a simple graph")
    if list(nx.selfloop_edges(counterfactual)):
        raise InterventionError("DUPLICATE_EDGE", "Self loops are not allowed")
    if counterfactual.has_edge(u, v):
        raise InterventionError("FORBIDDEN_EDGE", "Target edge was added")
    if any(edge(*item) in forbidden_edges for item in log.added_edges):
        raise InterventionError("FORBIDDEN_EDGE", "An added edge is a held-out positive")
    expected_removed = {edge(*item) for item in original.edges()} - {edge(*item) for item in counterfactual.edges()}
    expected_added = {edge(*item) for item in counterfactual.edges()} - {edge(*item) for item in original.edges()}
    if expected_removed != set(map(lambda item: edge(*item), log.removed_edges)):
        raise InterventionError("CN_CONSTRAINT_FAILED", "Removed-edge log does not match graph diff")
    if expected_added != set(map(lambda item: edge(*item), log.added_edges)):
        raise InterventionError("CN_CONSTRAINT_FAILED", "Added-edge log does not match graph diff")
    if original.degree(u) > 0 and counterfactual.degree(u) == 0:
        raise InterventionError("DISCONNECT_ENDPOINT", "u became isolated")
    if original.degree(v) > 0 and counterfactual.degree(v) == 0:
        raise InterventionError("DISCONNECT_ENDPOINT", "v became isolated")

    original_cn = common_neighbor_set(original, u, v)
    counterfactual_cn = common_neighbor_set(counterfactual, u, v)
    if log.intervention_type.startswith("cn_"):
        if dict(original.degree()) != dict(counterfactual.degree()):
            raise InterventionError("DEGREE_CONSTRAINT_FAILED", "CN intervention changed a node degree")
        if len(counterfactual_cn) - len(original_cn) != log.requested_delta:
            raise InterventionError("CN_CONSTRAINT_FAILED", "CN delta does not match request")
        expected_edits = 2 * abs(log.requested_delta)
        if len(log.removed_edges) != expected_edits or len(log.added_edges) != expected_edits:
            raise InterventionError("CN_CONSTRAINT_FAILED", "CN intervention is not a minimal 2-switch sequence")
    elif log.intervention_type.startswith("degree_"):
        if counterfactual_cn != original_cn:
            raise InterventionError("CN_CONSTRAINT_FAILED", "Degree intervention changed the CN set")
        endpoint = u if "_u_" in log.intervention_type else v
        if counterfactual.degree(endpoint) - original.degree(endpoint) != log.requested_delta:
            raise InterventionError("DEGREE_CONSTRAINT_FAILED", "Endpoint degree delta does not match request")
        if len(log.removed_edges) != abs(log.requested_delta) or len(log.added_edges) != abs(log.requested_delta):
            raise InterventionError("DEGREE_CONSTRAINT_FAILED", "Degree intervention is not a minimal edge transfer")
    else:
        raise ValueError(f"Unknown intervention type {log.intervention_type}")

