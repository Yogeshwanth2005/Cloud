import networkx as nx
import pytest

from aco.causal.uncertainty import edge_uncertainty, node_uncertainty

VARS = ["a", "b", "c"]


def test_empty_graph_is_maximally_uncertain():
    assert edge_uncertainty(nx.DiGraph(), VARS) == pytest.approx(1.0)


def test_discovering_one_edge_moves_uncertainty_by_one_pair_not_to_zero():
    # The cliff this replaces: the old measure averaged over the edges that had
    # been discovered, so its denominator grew with its numerator and the first
    # discovered edge took it from its no-edges default of 1.0 straight to ~0.
    # Measuring over the candidate pairs fixes the denominator, which is what
    # makes this a gradient an optimizer can respond to.
    g = nx.DiGraph()
    g.add_edge("a", "b", pval=0.0)
    # 3 variables -> 6 ordered pairs; one is now certain, five are not.
    assert edge_uncertainty(g, VARS) == pytest.approx(5 / 6)


def test_a_weak_edge_counts_as_partly_uncertain():
    g = nx.DiGraph()
    g.add_edge("a", "b", pval=0.5)
    assert edge_uncertainty(g, VARS) == pytest.approx((0.5 + 5) / 6)


def test_node_uncertainty_covers_only_pairs_touching_that_node():
    g = nx.DiGraph()
    g.add_edge("a", "b", pval=0.0)
    # Pairs touching "a": a->b, a->c, b->a, c->a -> one certain, three not.
    assert node_uncertainty(g, "a", VARS) == pytest.approx(3 / 4)
    # "c" touches four pairs, none of them discovered.
    assert node_uncertainty(g, "c", VARS) == pytest.approx(1.0)


def test_uncertainty_falls_monotonically_as_edges_are_established():
    # The property Task 4 depends on: a better causal model must produce a
    # strictly smaller number, so it can tighten the optimizer's ambiguity set.
    g = nx.DiGraph()
    readings = [edge_uncertainty(g, VARS)]
    for u, v in [("a", "b"), ("b", "c"), ("a", "c")]:
        g.add_edge(u, v, pval=0.0)
        readings.append(edge_uncertainty(g, VARS))

    assert readings == sorted(readings, reverse=True)
    assert readings[0] > readings[-1]
