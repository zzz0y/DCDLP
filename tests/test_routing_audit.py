import networkx as nx
import numpy as np
import pytest

from dcdlp.evaluation.routing_audit import (
    branch_response,
    paired_effect,
    random_rewire_like,
)


def _output(degree, cn, residual, interaction):
    return {
        "score_degree": np.asarray([degree]),
        "score_cn": np.asarray([cn]),
        "score_residual": np.asarray([residual]),
        "score_interaction": np.asarray([interaction]),
        "logit": np.asarray([degree + cn + residual + interaction]),
    }


def test_branch_response_reports_interaction_separately():
    result = branch_response(
        _output(1, 2, 3, 4), _output(3, 3, 3, 5), "degree"
    )
    assert result["delta_degree"] == 2
    assert result["delta_interaction"] == 1
    assert result["routing_selectivity_main_branches"] == pytest.approx(2 / 3)
    assert result["routing_selectivity_with_interaction"] == pytest.approx(2 / 4)
    assert result["interaction_share"] == pytest.approx(1 / 4)


def test_paired_effect_is_pairwise_not_cartesian():
    result = paired_effect([2.0, 1.0, 4.0], [1.0, 1.0, 5.0], bootstrap_samples=100)
    assert result["wins"] == 1
    assert result["ties"] == 1
    assert result["losses"] == 1
    assert result["paired_cliffs_delta"] == 0


def test_random_rewire_obeys_safety_and_edit_count():
    graph = nx.cycle_graph(10)
    result = random_rewire_like(
        graph, 0, 5, 2, seed=7, forbidden_edges={(1, 8)}
    )
    assert len(result.removed_edges) == 2
    assert len(result.added_edges) == 2
    assert not list(nx.selfloop_edges(result.graph))
    assert not result.graph.has_edge(0, 5)
    assert not result.graph.has_edge(1, 8)
    assert result.graph.number_of_edges() == graph.number_of_edges()


def test_random_rewire_supports_one_edge_degree_control():
    graph = nx.cycle_graph(10)
    result = random_rewire_like(graph, 0, 5, 1, seed=11)
    assert len(result.removed_edges) == 1
    assert len(result.added_edges) == 1
