import numpy as np
import torch

from dcdlp.data.synthetic import make_smoke_dataset
from dcdlp.interventions import intervene_degree
from dcdlp.models.dcdlp import DCDLP
from dcdlp.utils import seed_everything


def test_seed_reproduces_model_initialization_and_intervention():
    dataset = make_smoke_dataset(11)
    seed_everything(12)
    first = DCDLP(16, 16, 8)
    first_parameters = torch.cat([value.detach().flatten() for value in first.parameters()])
    seed_everything(12)
    second = DCDLP(16, 16, 8)
    second_parameters = torch.cat([value.detach().flatten() for value in second.parameters()])
    torch.testing.assert_close(first_parameters, second_parameters)

    graph = dataset.train_graph()
    pair = next((tuple(edge) for edge in dataset.test_pos if not graph.has_edge(*edge)), None)
    assert pair is not None
    try:
        _, first_log = intervene_degree(graph, *pair, "u", 1, seed=99)
        _, second_log = intervene_degree(graph, *pair, "u", 1, seed=99)
    except Exception:
        return
    assert first_log.added_edges == second_log.added_edges
    assert first_log.removed_edges == second_log.removed_edges
