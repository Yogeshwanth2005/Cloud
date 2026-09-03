import networkx as nx

from aco.interventions.voi import score_intervention, select_best_intervention


def test_score_intervention_is_positive_when_uncertainty_is_high():
    graph = nx.DiGraph()
    graph.add_edge("poa_irradiance", "dc_power", pval=0.2)
    score = score_intervention(
        graph, "poa_irradiance", current_uncertainty={"poa_irradiance": 0.2},
        expected_uncertainty_reduction=0.15, name="high_res_sampling", magnitude=1.0,
    )
    assert score > 0


def test_select_best_intervention_returns_none_when_all_negative():
    graph = nx.DiGraph()
    graph.add_edge("poa_irradiance", "dc_power", pval=0.01)  # already well-known -> little to gain
    result = select_best_intervention(
        graph, node_candidates=["poa_irradiance"], uncertainty_estimates={"poa_irradiance": 0.01},
    )
    assert result is None
