import numpy as np

from dcdlp.data.negative_sampling import degree_corrected_sampling, uniform_negative_sampling
from dcdlp.data.synthetic import make_smoke_dataset


def test_negative_samplers_never_return_true_edges_or_self_loops():
    dataset = make_smoke_dataset(2)
    truth = {tuple(edge) for edge in dataset.all_positive.tolist()}
    for negatives in (
        uniform_negative_sampling(dataset.num_nodes, dataset.all_positive, 100, 9),
        degree_corrected_sampling(dataset.train_graph(), dataset.all_positive, 100, 9),
    ):
        assert len({tuple(edge) for edge in negatives.tolist()}) == 100
        assert all(u != v and (u, v) not in truth for u, v in negatives)


def test_negative_sampling_is_reproducible():
    dataset = make_smoke_dataset(3)
    first = uniform_negative_sampling(dataset.num_nodes, dataset.all_positive, 20, 5)
    second = uniform_negative_sampling(dataset.num_nodes, dataset.all_positive, 20, 5)
    np.testing.assert_array_equal(first, second)

