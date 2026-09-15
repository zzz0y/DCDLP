from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np


def canonical_edges(edges: Iterable[Iterable[int]]) -> np.ndarray:
    source = edges if isinstance(edges, np.ndarray) else list(edges)
    array = np.asarray(source, dtype=np.int64).reshape(-1, 2)
    if len(array) == 0:
        return array
    array = np.sort(array, axis=1)
    # Link-prediction positives represent pairs of distinct nodes.  Removing
    # source-data self-loops here also keeps the intervention validator and
    # message-passing graph on the same simple-graph contract.
    array = array[array[:, 0] != array[:, 1]]
    return np.unique(array, axis=0)


def normalized_edge_rows(edges: Iterable[Iterable[int]]) -> np.ndarray:
    """Normalize undirected endpoints while preserving row order and multiplicity."""
    source = edges if isinstance(edges, np.ndarray) else list(edges)
    array = np.asarray(source)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Edge tensor must have shape [N, 2], got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Edge tensor must use an integer dtype")
    return np.sort(array.astype(np.int64, copy=False), axis=1)


def _row_keys(rows: np.ndarray) -> np.ndarray:
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    return rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).reshape(-1)


def _has_row_overlap(left: np.ndarray, right: np.ndarray) -> bool:
    if len(left) == 0 or len(right) == 0:
        return False
    return bool(np.intersect1d(_row_keys(left), _row_keys(right)).size)


def _validated_years(years: np.ndarray, edge_count: int, split: str) -> np.ndarray:
    array = np.asarray(years)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{split}_year must use an integer dtype")
    if array.ndim not in {1, 2} or array.size != edge_count:
        raise ValueError(f"{split}_year must contain one year per positive edge")
    return array.astype(np.int64, copy=False).reshape(-1)


@dataclass
class GraphDataset:
    name: str
    num_nodes: int
    features: np.ndarray
    train_pos: np.ndarray
    valid_pos: np.ndarray
    test_pos: np.ndarray
    all_positive: np.ndarray
    valid_neg: np.ndarray | None = None
    test_neg: np.ndarray | None = None
    train_year: np.ndarray | None = None
    valid_year: np.ndarray | None = None
    test_year: np.ndarray | None = None

    def train_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(self.num_nodes))
        graph.add_edges_from(map(tuple, self.train_pos.tolist()))
        return graph

    def validate(self) -> None:
        if not isinstance(self.num_nodes, (int, np.integer)) or self.num_nodes <= 0:
            raise ValueError("num_nodes must be a positive integer")
        features = np.asarray(self.features)
        if features.ndim != 2:
            raise ValueError(f"Feature tensor must have shape [num_nodes, feature_dim], got {features.shape}")
        if features.shape[0] != self.num_nodes:
            raise ValueError("Feature row count must equal num_nodes")

        split_names = ("train", "valid", "test")
        splits = [normalized_edge_rows(value) for value in (self.train_pos, self.valid_pos, self.test_pos)]
        all_positive = normalized_edge_rows(self.all_positive)
        for split, edges in zip((*split_names, "all_positive"), (*splits, all_positive)):
            if edges.size and (edges.min() < 0 or edges.max() >= self.num_nodes):
                raise ValueError(f"{split} edge contains a node index outside [0, num_nodes)")
            if edges.size and np.any(edges[:, 0] == edges[:, 1]):
                raise ValueError(f"{split} positive edges must not contain self-loops")

        year_values = (self.train_year, self.valid_year, self.test_year)
        any_year = any(value is not None for value in year_values)
        all_years = all(value is not None for value in year_values)
        if any_year and not all_years:
            raise ValueError("train_year, valid_year, and test_year must be provided together")
        years = None
        if all_years:
            years = [
                _validated_years(value, len(edges), split)
                for value, edges, split in zip(year_values, splits, split_names)
            ]

        # Static datasets remain strictly pair-disjoint.  ogbl-collab is a
        # temporal multigraph: the same pair may be positive in different
        # years, but the exact normalized (u, v, year) event may not be reused.
        if self.name == "ogbl-collab" and years is not None:
            identities = [np.column_stack([edges, year]) for edges, year in zip(splits, years)]
            for split, identity in zip(split_names, identities):
                if len(np.unique(identity, axis=0)) != len(identity):
                    raise ValueError(f"Duplicate temporal positive edge within {split} split")
            if (_has_row_overlap(identities[0], identities[1]) or
                    _has_row_overlap(identities[0], identities[2]) or
                    _has_row_overlap(identities[1], identities[2])):
                raise ValueError("Temporal positive edge leakage across train/valid/test splits")
        else:
            for split, edges in zip(split_names, splits):
                if len(np.unique(edges, axis=0)) != len(edges):
                    raise ValueError(f"Duplicate positive edge within {split} split")
            if self.name != "ogbl-collab" and (
                _has_row_overlap(splits[0], splits[1]) or
                _has_row_overlap(splits[0], splits[2]) or
                _has_row_overlap(splits[1], splits[2])
            ):
                raise ValueError("Positive edge leakage across train/valid/test splits")

        expected_union = canonical_edges(np.vstack(splits))
        canonical_all = canonical_edges(all_positive)
        if len(canonical_all) != len(all_positive):
            raise ValueError("all_positive contains duplicate undirected pairs")
        if not np.array_equal(expected_union, canonical_all):
            raise ValueError("all_positive does not equal the union of positive splits")

        self._validate_negatives("valid", self.valid_neg, splits[1])
        self._validate_negatives("test", self.test_neg, splits[2])

    def _validate_negatives(self, split: str, negatives: np.ndarray | None, positives: np.ndarray) -> None:
        if negatives is None:
            return
        array = np.asarray(negatives)
        if array.ndim < 2 or array.shape[-1] != 2:
            raise ValueError(f"{split}_neg must have shape [..., 2], got {array.shape}")
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{split}_neg must use an integer dtype")
        flat = np.sort(array.astype(np.int64, copy=False).reshape(-1, 2), axis=1)
        if flat.size and (flat.min() < 0 or flat.max() >= self.num_nodes):
            raise ValueError(f"{split}_neg contains a node index outside [0, num_nodes)")
        if array.ndim > 2 and array.shape[0] == len(positives):
            shaped = np.sort(array.astype(np.int64, copy=False), axis=-1)
            positive_shape = (len(positives),) + (1,) * (array.ndim - 2) + (2,)
            if np.any(np.all(shaped == positives.reshape(positive_shape), axis=-1)):
                raise ValueError(f"{split}_neg conflicts with its corresponding positive edge")
        elif _has_row_overlap(flat, positives):
            raise ValueError(f"{split}_neg conflicts with a {split} positive edge")


def save_dataset(dataset: GraphDataset, path: Path) -> None:
    dataset.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    valid_is_aligned = (
        dataset.valid_neg is not None and np.asarray(dataset.valid_neg).ndim > 2 and
        np.asarray(dataset.valid_neg).shape[0] == len(dataset.valid_pos)
    )
    test_is_aligned = (
        dataset.test_neg is not None and np.asarray(dataset.test_neg).ndim > 2 and
        np.asarray(dataset.test_neg).shape[0] == len(dataset.test_pos)
    )
    payload = dict(
        name=dataset.name,
        num_nodes=dataset.num_nodes,
        features=dataset.features.astype(np.float32),
        train_pos=(normalized_edge_rows(dataset.train_pos) if dataset.train_year is not None
                   else canonical_edges(dataset.train_pos)),
        valid_pos=(normalized_edge_rows(dataset.valid_pos) if dataset.valid_year is not None or valid_is_aligned
                   else canonical_edges(dataset.valid_pos)),
        test_pos=(normalized_edge_rows(dataset.test_pos) if dataset.test_year is not None or test_is_aligned
                  else canonical_edges(dataset.test_pos)),
        all_positive=canonical_edges(dataset.all_positive),
    )
    if dataset.valid_neg is not None:
        payload["valid_neg"] = np.asarray(dataset.valid_neg, dtype=np.int64)
    if dataset.test_neg is not None:
        payload["test_neg"] = np.asarray(dataset.test_neg, dtype=np.int64)
    for key in ("train_year", "valid_year", "test_year"):
        value = getattr(dataset, key)
        if value is not None:
            payload[key] = np.asarray(value, dtype=np.int64).reshape(-1)
    np.savez_compressed(path, **payload)


def _load_npz(path: Path) -> GraphDataset:
    data = np.load(path, allow_pickle=False)
    dataset = GraphDataset(
        name=str(data["name"]),
        num_nodes=int(data["num_nodes"]),
        features=np.asarray(data["features"], dtype=np.float32),
        train_pos=np.asarray(data["train_pos"], dtype=np.int64),
        valid_pos=np.asarray(data["valid_pos"], dtype=np.int64),
        test_pos=np.asarray(data["test_pos"], dtype=np.int64),
        all_positive=np.asarray(data["all_positive"], dtype=np.int64),
        valid_neg=np.asarray(data["valid_neg"], dtype=np.int64) if "valid_neg" in data.files else None,
        test_neg=np.asarray(data["test_neg"], dtype=np.int64) if "test_neg" in data.files else None,
        train_year=np.asarray(data["train_year"], dtype=np.int64) if "train_year" in data.files else None,
        valid_year=np.asarray(data["valid_year"], dtype=np.int64) if "valid_year" in data.files else None,
        test_year=np.asarray(data["test_year"], dtype=np.int64) if "test_year" in data.files else None,
    )
    dataset.validate()
    return dataset


def _from_pyg_planetoid(name: str, root: Path, seed: int) -> GraphDataset:
    try:
        import torch
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import RandomLinkSplit
    except ImportError as exc:
        raise RuntimeError("Planetoid preparation requires torch-geometric") from exc

    pyg_names = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
    pyg = Planetoid(str(root), pyg_names[name])[0]
    splitter = RandomLinkSplit(
        num_val=0.05, num_test=0.10, is_undirected=True,
        add_negative_train_samples=False, split_labels=True,
    )
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    train, valid, test = splitter(pyg)
    torch.random.set_rng_state(generator_state)

    def positive_edges(obj) -> np.ndarray:
        return canonical_edges(obj.pos_edge_label_index.t().cpu().numpy())

    train_pos = canonical_edges(train.edge_index.t().cpu().numpy())
    valid_pos = positive_edges(valid)
    test_pos = positive_edges(test)
    all_pos = canonical_edges(np.vstack([train_pos, valid_pos, test_pos]))
    return GraphDataset(
        name=name, num_nodes=int(pyg.num_nodes), features=pyg.x.cpu().numpy(),
        train_pos=train_pos, valid_pos=valid_pos, test_pos=test_pos,
        all_positive=all_pos,
    )


def _from_ogb(name: str, root: Path) -> GraphDataset:
    try:
        from ogb.linkproppred import PygLinkPropPredDataset
    except ImportError as exc:
        raise RuntimeError("OGB preparation requires ogb and torch-geometric") from exc
    dataset = PygLinkPropPredDataset(name=name, root=str(root))
    graph = dataset[0]
    split = dataset.get_edge_split()

    def as_numpy(value) -> np.ndarray:
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    def get_edges(part: dict) -> tuple[np.ndarray, np.ndarray | None]:
        if "edge" in part:
            raw_edges = as_numpy(part["edge"])
        else:
            raw_edges = np.column_stack([
                as_numpy(part["source_node"]), as_numpy(part["target_node"]),
            ])
        if name == "ogbl-collab" and "year" in part:
            return normalized_edge_rows(raw_edges), as_numpy(part["year"]).astype(np.int64).reshape(-1)
        return canonical_edges(raw_edges), None

    (train, train_year), (valid, valid_year), (test, test_year) = (
        get_edges(split[key]) for key in ("train", "valid", "test")
    )
    valid_neg = canonical_edges(as_numpy(split["valid"]["edge_neg"])) if "edge_neg" in split["valid"] else None
    test_neg = canonical_edges(as_numpy(split["test"]["edge_neg"])) if "edge_neg" in split["test"] else None
    features = graph.x.cpu().numpy() if graph.x is not None else np.ones((graph.num_nodes, 1), np.float32)
    return GraphDataset(
        name=name, num_nodes=int(graph.num_nodes), features=features,
        train_pos=train, valid_pos=valid, test_pos=test,
        all_positive=canonical_edges(np.vstack([train, valid, test])),
        valid_neg=valid_neg, test_neg=test_neg,
        train_year=train_year, valid_year=valid_year, test_year=test_year,
    )


def _read_edge_file(path: Path, preserve_order: bool = False) -> np.ndarray:
    edges = np.loadtxt(path, delimiter="\t", dtype=np.int64)
    rows = edges.reshape(-1, 2)
    return normalized_edge_rows(rows) if preserve_order else canonical_edges(rows)


def _from_heart_small(name: str) -> GraphDataset:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("HeaRT feature loading requires PyTorch") from exc
    heart_root = Path(__file__).resolve().parents[3] / "third_party" / "HeaRT" / "dataset" / name
    required = [
        heart_root / "train_pos.txt", heart_root / "valid_pos.txt", heart_root / "test_pos.txt",
        heart_root / "gnn_feature", heart_root / "heart_valid_samples.npy",
        heart_root / "heart_test_samples.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("HeaRT official data is incomplete: " + ", ".join(missing))
    train = _read_edge_file(heart_root / "train_pos.txt")
    # HeaRT negative candidates are row-aligned with these two files.  Keep
    # their original row order while still normalizing undirected endpoints.
    valid = _read_edge_file(heart_root / "valid_pos.txt", preserve_order=True)
    test = _read_edge_file(heart_root / "test_pos.txt", preserve_order=True)
    feature_payload = torch.load(heart_root / "gnn_feature", map_location="cpu")
    features = feature_payload["entity_embedding"] if isinstance(feature_payload, dict) else feature_payload
    valid_neg = np.load(heart_root / "heart_valid_samples.npy", mmap_mode="r")
    test_neg = np.load(heart_root / "heart_test_samples.npy", mmap_mode="r")
    dataset = GraphDataset(
        name=name, num_nodes=int(features.shape[0]), features=features.cpu().numpy(),
        train_pos=train, valid_pos=valid, test_pos=test,
        all_positive=canonical_edges(np.vstack([train, valid, test])),
        valid_neg=np.asarray(valid_neg), test_neg=np.asarray(test_neg),
    )
    dataset.validate()
    return dataset


def load_dataset(name: str, root: Path, protocol: str = "standard", seed: int = 0) -> GraphDataset:
    prepared = root / "processed" / f"{name}_{protocol}_seed{seed}.npz"
    if prepared.exists():
        return _load_npz(prepared)
    # HeaRT and OGB provide fixed official splits.  Reuse the seed-0 prepared
    # artifact for every model seed instead of materializing identical large
    # NPZ files (and repeatedly parsing the raw OGB dataset).
    if protocol in {"heart", "ogb"}:
        shared = root / "processed" / f"{name}_{protocol}_seed0.npz"
        if shared.exists():
            return _load_npz(shared)
    if name == "smoke":
        from .synthetic import make_smoke_dataset
        return make_smoke_dataset(seed)
    if name.startswith("synthetic-"):
        from .synthetic import make_pa_tc_dataset
        mechanism = name.removeprefix("synthetic-")
        return make_pa_tc_dataset(mechanism=mechanism, seed=seed)
    if name in {"cora", "citeseer", "pubmed"}:
        if protocol == "heart":
            return _from_heart_small(name)
        return _from_pyg_planetoid(name, root / "raw", seed)
    if name.startswith("ogbl-"):
        return _from_ogb(name, root / "raw")
    raise ValueError(f"Unknown dataset: {name}")
