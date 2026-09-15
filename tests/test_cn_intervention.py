import networkx as nx

from dcdlp.interventions import intervene_cn
from dcdlp.interventions.validator import common_neighbor_set


def plus_graph():
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from([(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 7)])
    return graph


def minus_graph():
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from([(0, 2), (1, 2), (3, 4), (0, 5), (1, 6), (3, 7), (4, 7), (5, 6)])
    return graph


def test_cn_plus_preserves_complete_degree_vector():
    graph = plus_graph()
    cf, log = intervene_cn(graph, 0, 1, 1, seed=1)
    assert log.valid
    assert dict(graph.degree()) == dict(cf.degree())
    assert len(common_neighbor_set(cf, 0, 1)) - len(common_neighbor_set(graph, 0, 1)) == 1


def test_cn_minus_preserves_complete_degree_vector():
    graph = minus_graph()
    cf, log = intervene_cn(graph, 0, 1, -1, seed=4)
    assert log.valid
    assert dict(graph.degree()) == dict(cf.degree())
    assert len(common_neighbor_set(cf, 0, 1)) - len(common_neighbor_set(graph, 0, 1)) == -1

