from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import networkx as nx
import numpy as np

from dcdlp.interventions.validator import edge


BRANCHES = ("degree", "cn", "residual")
SCORES = ("degree", "cn", "residual", "interaction", "total")


def _scalar(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) != 1:
        raise ValueError(f"Expected one score, received shape {array.shape}")
    return float(array[0])


def branch_response(
    original: Mapping[str, object],
    counterfactual: Mapping[str, object],
    intervention_kind: str,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    """Return score-level response metrics for one strictly paired intervention."""
    if intervention_kind not in {"degree", "cn"}:
        raise ValueError("intervention_kind must be degree or cn")
    deltas = {
        "degree": _scalar(counterfactual["score_degree"]) - _scalar(original["score_degree"]),
        "cn": _scalar(counterfactual["score_cn"]) - _scalar(original["score_cn"]),
        "residual": _scalar(counterfactual["score_residual"]) - _scalar(original["score_residual"]),
        "interaction": (
            _scalar(counterfactual["score_interaction"])
            - _scalar(original["score_interaction"])
        ),
        "total": _scalar(counterfactual["logit"]) - _scalar(original["logit"]),
    }
    absolute = {name: abs(value) for name, value in deltas.items()}
    target = intervention_kind
    main_denominator = sum(absolute[name] for name in BRANCHES)
    full_denominator = main_denominator + absolute["interaction"]
    non_targets = [name for name in BRANCHES if name != target]
    output: dict[str, float] = {}
    for name in SCORES:
        output[f"delta_{name}"] = deltas[name]
        output[f"abs_delta_{name}"] = absolute[name]
    output.update({
        "target_abs_delta": absolute[target],
        "non_target_abs_delta": sum(absolute[name] for name in non_targets),
        "routing_selectivity_main_branches": (
            absolute[target] / (main_denominator + epsilon)
        ),
        "routing_selectivity_with_interaction": (
            absolute[target] / (full_denominator + epsilon)
        ),
        "interaction_share": absolute["interaction"] / (full_denominator + epsilon),
    })
    return output


def summarize_values(
    values: Iterable[float],
    *,
    seed: int = 0,
    bootstrap_samples: int = 2_000,
    confidence: float = 0.95,
) -> dict[str, float | int | str]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {
            "status": "MISSING", "n": 0, "mean": np.nan, "std": np.nan,
            "median": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        }
    rng = np.random.default_rng(seed)
    estimates = np.empty(bootstrap_samples, dtype=float)
    for index in range(bootstrap_samples):
        estimates[index] = rng.choice(array, size=len(array), replace=True).mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "status": "AVAILABLE",
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
    }


def paired_effect(
    a14_values: Iterable[float],
    a5_values: Iterable[float],
    *,
    seed: int = 0,
    bootstrap_samples: int = 2_000,
    tie_tolerance: float = 1e-12,
) -> dict[str, float | int | str]:
    a14 = np.asarray(list(a14_values), dtype=float)
    a5 = np.asarray(list(a5_values), dtype=float)
    if a14.shape != a5.shape:
        raise ValueError("Paired arrays must have exactly the same shape")
    finite = np.isfinite(a14) & np.isfinite(a5)
    delta = a14[finite] - a5[finite]
    if not len(delta):
        return {
            "status": "MISSING", "n": 0, "mean_delta": np.nan,
            "median_delta": np.nan, "ci_low": np.nan, "ci_high": np.nan,
            "wins": 0, "ties": 0, "losses": 0, "paired_cliffs_delta": np.nan,
        }
    summary = summarize_values(
        delta, seed=seed, bootstrap_samples=bootstrap_samples
    )
    wins = int(np.sum(delta > tie_tolerance))
    losses = int(np.sum(delta < -tie_tolerance))
    ties = int(len(delta) - wins - losses)
    return {
        "status": "AVAILABLE",
        "n": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "ci_low": summary["ci_low"],
        "ci_high": summary["ci_high"],
        "wins": wins,
        "ties": ties,
        "losses": losses,
        # A paired ordinal effect: +1 means every paired A14 value wins.
        "paired_cliffs_delta": float((wins - losses) / len(delta)),
    }


@dataclass(frozen=True)
class RandomRewire:
    graph: nx.Graph
    removed_edges: tuple[tuple[int, int], ...]
    added_edges: tuple[tuple[int, int], ...]


def _valid_added_edge(
    graph: nx.Graph,
    candidate: tuple[int, int],
    target: tuple[int, int],
    forbidden: set[tuple[int, int]],
    removed_now: set[tuple[int, int]],
) -> bool:
    u, v = candidate
    return (
        u != v
        and candidate != target
        and candidate not in forbidden
        and (not graph.has_edge(u, v) or candidate in removed_now)
    )


def random_rewire_like(
    graph: nx.Graph,
    u: int,
    v: int,
    edit_count: int,
    *,
    seed: int,
    forbidden_edges: set[tuple[int, int]] | None = None,
    max_attempts: int = 2_000,
) -> RandomRewire:
    """Create an edit-count-matched, unconstrained simple-graph rewire.

    The control deliberately does not preserve degree or CN, but it does
    preserve edge count and obey all held-out-edge safety constraints.
    """
    if edit_count < 1:
        raise ValueError("edit_count must be positive")
    target = edge(u, v)
    forbidden = {edge(*item) for item in (forbidden_edges or set())}
    original_edges = {edge(*item) for item in graph.edges()}
    rng = np.random.default_rng(seed)
    edge_list = sorted(original_edges)
    nodes = np.asarray(sorted(graph.nodes()), dtype=int)
    if len(edge_list) < 1 or len(nodes) < 3:
        raise RuntimeError("MISSING: graph has too few edges for random rewire")

    for restart in range(32):
        current = graph.copy()
        for _ in range(max_attempts):
            if len(original_edges - {edge(*item) for item in current.edges()}) == edit_count:
                current_edges = {edge(*item) for item in current.edges()}
                removed = original_edges - current_edges
                added = current_edges - original_edges
                if len(added) == edit_count:
                    return RandomRewire(
                        current, tuple(sorted(removed)), tuple(sorted(added))
                    )
                break
            current_edges = sorted({edge(*item) for item in current.edges()})
            removable = current_edges[int(rng.integers(len(current_edges)))]
            if not current.has_edge(*removable):
                continue
            raw_a, raw_b = rng.choice(nodes, 2, replace=False)
            added = edge(int(raw_a), int(raw_b))
            if not _valid_added_edge(
                current, added, target, forbidden, {removable}
            ):
                continue
            candidate = current.copy()
            candidate.remove_edge(*removable)
            candidate.add_edge(*added)
            if list(nx.selfloop_edges(candidate)):
                continue
            current = candidate
        rng = np.random.default_rng(seed + restart + 1)
        edge_list = sorted(original_edges)
    raise RuntimeError("MISSING: could not construct a valid random rewire control")
