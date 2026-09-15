import networkx as nx

from dcdlp.interventions import intervene_degree
from dcdlp.interventions.validator import common_neighbor_set


def graph_for_degree_edits():
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from([(0, 2), (1, 2), (0, 4), (3, 5), (3, 6), (5, 7), (6, 7)])
    return graph


def test_degree_plus_preserves_target_common_neighbor_set():
    graph = graph_for_degree_edits()
    cf, log = intervene_degree(graph, 0, 1, "u", 1, seed=3)
    assert log.valid
    assert cf.degree(0) == graph.degree(0) + 1
    assert common_neighbor_set(cf, 0, 1) == common_neighbor_set(graph, 0, 1)


def test_degree_minus_preserves_target_common_neighbor_set():
    graph = graph_for_degree_edits()
    cf, log = intervene_degree(graph, 0, 1, "u", -1, seed=7)
    assert log.valid
    assert cf.degree(0) == graph.degree(0) - 1
    assert common_neighbor_set(cf, 0, 1) == common_neighbor_set(graph, 0, 1)

