from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np


def pair_features(graph: nx.Graph, pairs: np.ndarray, node_features: np.ndarray | None = None) -> dict[str, np.ndarray]:
    degrees = dict(graph.degree())
    output: dict[str, list[float]] = {
        "degree_u": [], "degree_v": [], "degree_score": [], "degree_product": [],
        "cn": [], "aa": [], "ra": [], "jaccard": [], "shortest_path": [],
        "feature_similarity": [],
    }
    for raw_u, raw_v in np.asarray(pairs, dtype=np.int64):
        u, v = int(raw_u), int(raw_v)
        du, dv = degrees.get(u, 0), degrees.get(v, 0)
        common = set(graph.neighbors(u)) & set(graph.neighbors(v))
        output["degree_u"].append(du)
        output["degree_v"].append(dv)
        output["degree_score"].append(np.log1p(du) + np.log1p(dv))
        output["degree_product"].append((1 + du) * (1 + dv))
        output["cn"].append(len(common))
        output["aa"].append(sum(1.0 / np.log(max(degrees[w], 2)) for w in common))
        output["ra"].append(sum(1.0 / max(degrees[w], 1) for w in common))
        union = set(graph.neighbors(u)) | set(graph.neighbors(v))
        output["jaccard"].append(len(common) / len(union) if union else 0.0)
        try:
            output["shortest_path"].append(nx.shortest_path_length(graph, u, v))
        except nx.NetworkXNoPath:
            output["shortest_path"].append(np.inf)
        if node_features is None:
            output["feature_similarity"].append(np.nan)
        else:
            a, b = node_features[u], node_features[v]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            output["feature_similarity"].append(float(a @ b / denom) if denom > 0 else 0.0)
    return {key: np.asarray(value) for key, value in output.items()}


def conditional_inputs(stats: dict[str, np.ndarray]) -> np.ndarray:
    du, dv = stats["degree_u"], stats["degree_v"]
    lu, lv = np.log1p(du), np.log1p(dv)
    return np.column_stack([lu, lv, np.log1p(du * dv), np.abs(lu - lv)])


def normalized_cn(stats: dict[str, np.ndarray]) -> np.ndarray:
    """Symmetric degree-normalized common-neighbor statistic."""
    denominator = np.sqrt(
        np.maximum(
            np.asarray(stats["degree_u"], dtype=float)
            * np.asarray(stats["degree_v"], dtype=float),
            1.0,
        )
    )
    return np.asarray(stats["cn"], dtype=float) / denominator


def canonicalize_endpoint_degrees(
    stats: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return a shallow copy whose endpoint degrees have canonical order."""
    output = dict(stats)
    degree_u = np.asarray(stats["degree_u"])
    degree_v = np.asarray(stats["degree_v"])
    output["degree_u"] = np.minimum(degree_u, degree_v)
    output["degree_v"] = np.maximum(degree_u, degree_v)
    return output


@dataclass
class ConditionalCNRegressor:
    seed: int = 0
    symmetric: bool = False
    model: object | None = None
    fit_metadata: dict[str, object] = field(default_factory=dict)

    def _inputs(self, stats: dict[str, np.ndarray]) -> np.ndarray:
        selected = canonicalize_endpoint_degrees(stats) if self.symmetric else stats
        return conditional_inputs(selected)

    def fit(
        self,
        stats: dict[str, np.ndarray],
        *,
        metadata: dict[str, object] | None = None,
    ) -> "ConditionalCNRegressor":
        from sklearn.ensemble import HistGradientBoostingRegressor

        estimator = HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, random_state=self.seed,
        )
        self.model = estimator.fit(
            self._inputs(stats), np.log1p(stats["cn"])
        )
        self.fit_metadata = {
            "residualizer_type": type(estimator).__name__,
            "residualizer_parameters": estimator.get_params(deep=False),
            "fit_seed": int(self.seed),
            "fit_sample_count": int(len(stats["cn"])),
            "symmetric_endpoint_degrees": bool(self.symmetric),
            **(metadata or {}),
        }
        return self

    def predict(self, stats: dict[str, np.ndarray]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ConditionalCNRegressor must be fitted before predict")
        return np.asarray(self.model.predict(self._inputs(stats)))

    def residual(self, stats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        expected = self.predict(stats)
        return expected, np.log1p(stats["cn"]) - expected


def assign_quadrants(degree_score: np.ndarray, cn_residual: np.ndarray, degree_threshold: float) -> np.ndarray:
    high_degree = degree_score >= degree_threshold
    high_cn = cn_residual >= 0.0
    return np.where(high_degree, np.where(high_cn, "HH", "HL"), np.where(high_cn, "LH", "LL"))
