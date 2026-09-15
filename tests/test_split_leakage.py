import numpy as np
import pytest

from dcdlp.data.loaders import (
    GraphDataset, canonical_edges, load_dataset, normalized_edge_rows, save_dataset,
)
from dcdlp.data.synthetic import make_smoke_dataset


def test_positive_splits_are_disjoint_and_train_graph_is_clean():
    dataset = make_smoke_dataset(0)
    dataset.validate()
    train = {tuple(edge) for edge in dataset.train_pos.tolist()}
    held_out = {tuple(edge) for edge in np.vstack([dataset.valid_pos, dataset.test_pos]).tolist()}
    assert train.isdisjoint(held_out)
    graph_edges = {tuple(sorted(edge)) for edge in dataset.train_graph().edges()}
    assert graph_edges.isdisjoint(held_out)


def test_canonical_edges_removes_self_loops():
    edges = canonical_edges([[0, 0], [0, 1], [1, 0], [2, 2]])
    np.testing.assert_array_equal(edges, np.asarray([[0, 1]], dtype=np.int64))


def test_dataset_validation_rejects_positive_self_loops():
    dataset = make_dataset("cora", [[0, 0], [0, 1]], [[1, 2]], [[2, 3]])
    with pytest.raises(ValueError, match="self-loops"):
        dataset.validate()


def make_dataset(name, train, valid, test, years=None, valid_neg=None):
    splits = [np.asarray(value, dtype=np.int64).reshape(-1, 2) for value in (train, valid, test)]
    kwargs = {}
    if years is not None:
        kwargs = dict(zip(("train_year", "valid_year", "test_year"), [np.asarray(x) for x in years]))
    return GraphDataset(
        name=name,
        num_nodes=6,
        features=np.zeros((6, 3), dtype=np.float32),
        train_pos=splits[0],
        valid_pos=splits[1],
        test_pos=splits[2],
        all_positive=canonical_edges(np.vstack(splits)),
        valid_neg=valid_neg,
        **kwargs,
    )


def test_static_positive_pair_overlap_still_raises():
    dataset = make_dataset("cora", [[0, 1]], [[1, 0]], [[2, 3]])
    with pytest.raises(ValueError, match="Positive edge leakage"):
        dataset.validate()


def test_ogbl_collab_same_pair_in_different_years_is_allowed():
    dataset = make_dataset(
        "ogbl-collab", [[0, 1]], [[1, 0]], [[2, 3]],
        years=([2017], [2018], [2019]),
    )
    dataset.validate()
    assert {tuple(sorted(edge)) for edge in dataset.train_graph().edges()} == {(0, 1)}


def test_ogbl_collab_years_survive_processed_npz_round_trip(tmp_path):
    dataset = make_dataset(
        "ogbl-collab", [[0, 1], [1, 0]], [[1, 0]], [[2, 3]],
        years=([2016, 2017], [2018], [2019]),
    )
    output = tmp_path / "processed" / "ogbl-collab_ogb_seed0.npz"
    save_dataset(dataset, output)
    loaded = load_dataset("ogbl-collab", tmp_path, "ogb", 0)
    np.testing.assert_array_equal(loaded.train_pos, normalized_edge_rows(dataset.train_pos))
    np.testing.assert_array_equal(loaded.train_year, dataset.train_year)
    assert len(loaded.train_graph().edges()) == 1


def test_row_aligned_negative_candidates_preserve_positive_order(tmp_path):
    valid = np.asarray([[4, 5], [1, 2]], dtype=np.int64)
    valid_neg = np.asarray([[[0, 3]], [[0, 4]]], dtype=np.int64)
    dataset = make_dataset(
        "cora", [[0, 1]], valid, [[2, 3]], valid_neg=valid_neg,
    )
    output = tmp_path / "processed" / "cora_heart_seed0.npz"
    save_dataset(dataset, output)
    loaded = load_dataset("cora", tmp_path, "heart", 0)
    np.testing.assert_array_equal(loaded.valid_pos, valid)
    np.testing.assert_array_equal(loaded.valid_neg, valid_neg)


def test_ogbl_collab_same_temporal_edge_across_splits_raises():
    dataset = make_dataset(
        "ogbl-collab", [[0, 1]], [[1, 0]], [[2, 3]],
        years=([2018], [2018], [2019]),
    )
    with pytest.raises(ValueError, match="Temporal positive edge leakage"):
        dataset.validate()


def test_ogbl_collab_without_year_only_skips_cross_split_pair_check():
    dataset = make_dataset("ogbl-collab", [[0, 1]], [[1, 0]], [[2, 3]])
    dataset.validate()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"train_pos": np.asarray([[0, 1], [1, 0]]),
          "train_year": np.asarray([2017, 2017])}, "Duplicate temporal positive edge"),
        ({"valid_pos": np.asarray([[0, 6]])}, "outside"),
        ({"valid_neg": np.asarray([[2, 1]])}, "conflicts"),
    ],
)
def test_ogbl_collab_keeps_other_integrity_checks(changes, message):
    dataset = make_dataset(
        "ogbl-collab", [[0, 1]], [[1, 2]], [[2, 3]],
        years=([2017], [2018], [2019]), valid_neg=np.asarray([[4, 5]]),
    )
    for key, value in changes.items():
        setattr(dataset, key, value)
    with pytest.raises(ValueError, match=message):
        dataset.validate()
