import networkx as nx
import numpy as np
import pandas as pd

from aco.causal.graph import fit_observational_graph


def _ar1_irradiance(rng, n, mean=500, std=100, phi=0.8):
    """Autocorrelated (AR(1)) irradiance series with the same marginal mean/std
    as an i.i.d. draw from N(mean, std).

    Real solar irradiance is smooth/autocorrelated over time, not i.i.d. noise
    from one timestep to the next -- and that temporal structure isn't just
    cosmetic realism here: PCMCI+'s orientation rules need *some* genuine
    time-asymmetry to determine the direction of a contemporaneous (lag-0)
    link. Two variables that are purely i.i.d. across time and linked only at
    lag 0 are fundamentally non-identifiable by constraint-based causal
    discovery (swapping which variable "causes" the other produces the same
    joint distribution), so PCMCI+ correctly reports such a link as
    unoriented ('o-o') rather than guessing -- see fit_observational_graph's
    docstring. Giving irradiance realistic AR(1) persistence resolves that
    ambiguity the same way the module's own multi-lag test does (via a
    genuine lag-1 pathway), letting these tests exercise a link PCMCI+ can
    actually orient.
    """
    noise_std = std * np.sqrt(1 - phi ** 2)
    irradiance = np.empty(n)
    irradiance[0] = rng.normal(mean, std)
    for t in range(1, n):
        irradiance[t] = mean + phi * (irradiance[t - 1] - mean) + rng.normal(0, noise_std)
    return irradiance


def test_fit_observational_graph_recovers_known_edge():
    rng = np.random.default_rng(0)
    n = 500
    irradiance = _ar1_irradiance(rng, n)
    # dc_power is causally driven by irradiance at lag 0, plus noise
    dc_power = 0.2 * irradiance + rng.normal(0, 5, n)
    df = pd.DataFrame({"poa_irradiance": irradiance, "dc_power": dc_power})
    graph = fit_observational_graph(df, var_names=["poa_irradiance", "dc_power"], tau_max=1)
    assert graph.has_edge("poa_irradiance", "dc_power")
    assert nx.is_directed_acyclic_graph(graph)
    assert not any(u == v for u, v in graph.edges())


def test_fit_observational_graph_skips_unoriented_contemporaneous_link():
    # Regression test for the bidirectional-edge bug: two variables that are
    # each i.i.d. across time (no autocorrelation) and linked only at lag 0
    # give tigramite no temporal asymmetry to work with, so PCMCI+ cannot
    # determine which one causes the other -- it marks the contemporaneous
    # link 'o-o' (unoriented) even though p_matrix reports it as highly
    # significant in *both* directions (tigramite symmetrizes p_matrix[:,:,0]).
    #
    # This is exactly the flagship irradiance/dc_power DGP this module used
    # before its synthetic data was made autocorrelated to give PCMCI+ real
    # orientation evidence (see _ar1_irradiance). Before the fix, the naive
    # "p_matrix < 0.05" check added both poa_irradiance -> dc_power AND
    # dc_power -> poa_irradiance here, producing a 2-cycle (non-DAG) with a
    # reversed causal claim. The fix must skip this pair entirely rather than
    # asserting a direction PCMCI+ itself declined to determine.
    rng = np.random.default_rng(0)
    n = 500
    irradiance = rng.normal(500, 100, n)  # i.i.d. -- no autocorrelation
    dc_power = 0.2 * irradiance + rng.normal(0, 5, n)  # lag-0 only, real but unorientable
    df = pd.DataFrame({"poa_irradiance": irradiance, "dc_power": dc_power})

    graph = fit_observational_graph(df, var_names=["poa_irradiance", "dc_power"], tau_max=1)

    assert not graph.has_edge("poa_irradiance", "dc_power")
    assert not graph.has_edge("dc_power", "poa_irradiance")
    assert nx.is_directed_acyclic_graph(graph)
    assert not any(u == v for u, v in graph.edges())


def test_fit_observational_graph_keeps_strongest_lag_not_last_lag():
    # dc_power is driven by irradiance at both lag 0 (strong) and lag 1 (weaker
    # but still statistically significant). A DiGraph only has one edge slot
    # per (u, v) pair, so the edge-construction loop must keep the
    # lowest-pval (most significant) lag's attributes -- not whichever lag
    # happens to be processed last -- or the strong lag-0 relationship gets
    # silently clobbered by the weaker lag-1 one.
    rng = np.random.default_rng(1)
    n = 5000
    irradiance = rng.normal(500, 100, n)
    irr_lag1 = np.roll(irradiance, 1)
    irr_lag1[0] = irradiance[0]
    dc_power = 0.6 * irradiance + 0.2 * irr_lag1 + rng.normal(0, 3, n)
    df = pd.DataFrame({"poa_irradiance": irradiance, "dc_power": dc_power})

    graph = fit_observational_graph(df, var_names=["poa_irradiance", "dc_power"], tau_max=1)

    assert graph.has_edge("poa_irradiance", "dc_power")
    edge = graph["poa_irradiance"]["dc_power"]
    assert edge["lag"] == 0
    assert edge["weight"] > 0.9  # lag-0 link strength, not lag-1's ~0.3
    assert nx.is_directed_acyclic_graph(graph)
    assert not any(u == v for u, v in graph.edges())


def test_update_graph_with_intervention_sharpens_pval():
    # Simulate pre- and post-intervention data: pre has weak/noisy signal (small n),
    # post has the same relationship but much cleaner (large n). The test verifies
    # that update_graph_with_intervention replaces the weak pre-intervention estimate
    # with the sharper (lower pval) post-intervention estimate.
    from aco.causal.graph import update_graph_with_intervention

    rng = np.random.default_rng(1)
    irr_pre = _ar1_irradiance(rng, 30)
    pre = pd.DataFrame({
        "poa_irradiance": irr_pre,
        "dc_power": 0.2 * irr_pre + rng.normal(0, 20, 30),  # weak but real signal pre-intervention
    })
    pre_graph = fit_observational_graph(pre, var_names=["poa_irradiance", "dc_power"], tau_max=1)

    irr = _ar1_irradiance(rng, 500)
    post = pd.DataFrame({
        "poa_irradiance": irr,
        "dc_power": 0.2 * irr + rng.normal(0, 2, 500),  # clean signal post-intervention
    })
    updated = update_graph_with_intervention(
        pre_graph, "poa_irradiance", pre, post, var_names=["poa_irradiance", "dc_power"], tau_max=1,
    )
    assert updated.has_edge("poa_irradiance", "dc_power")
    assert updated["poa_irradiance"]["dc_power"]["pval"] <= pre_graph.get_edge_data(
        "poa_irradiance", "dc_power", {"pval": 1.0}
    )["pval"]


def test_update_graph_with_intervention_severs_incoming_edges_to_intervened_var():
    # A hard intervention forces intervened_var to a value chosen by the
    # orchestrator, decoupling it from any of its natural causes for the whole
    # post-intervention window -- it cannot have a causal parent there, no
    # matter what an observational refit of post_df finds. Construct post_df
    # with a real, PCMCI+-detectable lag-1 edge dc_power -> poa_irradiance
    # (unambiguous by time order, per fit_observational_graph's docstring) to
    # prove the naive "just refit and merge" approach WOULD keep this edge,
    # and that update_graph_with_intervention must not.
    from aco.causal.graph import fit_observational_graph, update_graph_with_intervention

    rng = np.random.default_rng(2)
    n = 500
    dc_power = rng.normal(50, 10, n)
    dc_lag1 = np.roll(dc_power, 1)
    dc_lag1[0] = dc_power[0]
    poa_irradiance = 0.5 * dc_lag1 + rng.normal(0, 1, n)
    post = pd.DataFrame({"poa_irradiance": poa_irradiance, "dc_power": dc_power})

    # Sanity check: an observational fit of this data really does find the
    # (here, spurious-if-treated-as-causal) edge into the would-be intervened var.
    post_only_graph = fit_observational_graph(post, var_names=["poa_irradiance", "dc_power"], tau_max=1)
    assert post_only_graph.has_edge("dc_power", "poa_irradiance")

    pre_graph = nx.DiGraph()
    pre_graph.add_nodes_from(["poa_irradiance", "dc_power"])

    updated = update_graph_with_intervention(
        pre_graph, "poa_irradiance", post, post, var_names=["poa_irradiance", "dc_power"], tau_max=1,
    )
    assert not updated.has_edge("dc_power", "poa_irradiance")
