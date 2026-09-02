import networkx as nx
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

# Fixed vocabulary every later module imports, so names never drift between
# phases. `power_mw` was added per the Phase 1.5 decision record -- it is the
# fleet's real metered output, kept distinct from PVDAQ's measured `ac_power`,
# not an alias for it. Tier 1 (PVDAQ) fits use the five physical names; Tier 2
# (fleet) fits use `power_mw` plus the operational/cloud names. No single
# dataframe is expected to carry all eleven at once.
NODE_SCHEMA = [
    "poa_irradiance", "module_temp", "ambient_temp", "dc_power", "ac_power", "power_mw",
    "curtailment_frac", "sampling_rate_hz", "cpu_rate_sum", "cost", "risk",
]


def fit_observational_graph(df, var_names: list, tau_max: int = 3) -> nx.DiGraph:
    """Run PCMCI+ over df[var_names] and return a time-lagged causal graph.

    Each edge carries `weight` (partial-correlation link strength), `pval`
    (the PCMCI+ p-value), and `lag` (the time lag, 0 for contemporaneous).

    An edge i -> j is added only where p_matrix[i, j, tau] < 0.05 AND
    PCMCI+'s own orientation mark (results["graph"][i, j, tau]) is '-->',
    i.e. PCMCI+ has actually determined the direction i causes j.

    This distinction matters at tau == 0 (contemporaneous links): tigramite
    symmetrizes p_matrix[:, :, 0] (p_matrix[i, j, 0] == p_matrix[j, i, 0]
    always), so a naive p_matrix-only check would add both i -> j and j -> i
    for any significant contemporaneous relationship -- even when PCMCI+
    itself marks the link 'o-o' (unoriented/undetermined, e.g. because there
    weren't enough variables/conditioning sets to apply its orientation
    rules). The orientation marks are NOT symmetrized: a directed link shows
    as '-->' from cause to effect and '<--' from effect to cause, and an
    unresolved link shows as 'o-o' on both sides -- only the '-->' side is
    used to add an edge, so an 'o-o' pair contributes no edge at all.

    At tau > 0 (lagged links), the direction is unambiguous by construction
    (an effect at a time lag can't run backward in time): empirically, this
    installed tigramite version (5.2.10.1) only ever marks a significant
    lagged link '-->', never '<--' or 'o-o', so the same combined check is
    also correct -- and harmlessly conservative -- there.
    """
    values = df[var_names].to_numpy()
    dataframe = pp.DataFrame(values, var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
    results = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=0.05)

    graph = nx.DiGraph()
    graph.add_nodes_from(var_names)
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    edge_marks = results["graph"]
    n_vars = len(var_names)
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                # Autodependencies (a variable's own lagged autocorrelation)
                # are not cross-variable causal relationships and must never
                # become a self-loop edge, at any lag.
                continue
            for tau in range(tau_max + 1):
                if p_matrix[i, j, tau] < 0.05 and edge_marks[i, j, tau] == "-->":
                    # A DiGraph has one edge slot per (u, v) pair, so if this pair is
                    # significant at multiple lags, keep only the most significant
                    # (lowest pval) one -- otherwise whichever lag is processed last
                    # (highest tau) silently wins, even if it's the weakest link.
                    # Same merge rule Task 3.2's update_graph_with_intervention uses.
                    if (
                        not graph.has_edge(var_names[i], var_names[j])
                        or p_matrix[i, j, tau] < graph[var_names[i]][var_names[j]]["pval"]
                    ):
                        graph.add_edge(
                            var_names[i], var_names[j],
                            weight=float(val_matrix[i, j, tau]),
                            pval=float(p_matrix[i, j, tau]),
                            lag=tau,
                        )
    return graph


def update_graph_with_intervention(graph, intervened_var, pre_df, post_df, var_names, tau_max=3):
    """Refits PCMCI+ on post-intervention data and merges edges into the pre-intervention graph.

    An edge whose pval improves (drops) after the intervention has its weight/pval
    replaced by the post-intervention estimate; edges unaffected by data volume
    around intervened_var are left as-is. Returns a new graph (does not mutate input).

    Args:
        graph: The pre-intervention causal graph (nx.DiGraph)
        intervened_var: The variable that was intervened upon (str)
        pre_df: Pre-intervention data (pd.DataFrame)
        post_df: Post-intervention data (pd.DataFrame)
        var_names: Variable names to fit (list[str])
        tau_max: Maximum lag to consider (int, default=3)

    Returns:
        A new nx.DiGraph with merged edges
    """
    post_graph = fit_observational_graph(post_df, var_names=var_names, tau_max=tau_max)
    merged = graph.copy()
    for u, v, data in post_graph.edges(data=True):
        if merged.has_edge(u, v):
            if data["pval"] <= merged[u][v].get("pval", 1.0):
                merged[u][v].update(data)
        else:
            merged.add_edge(u, v, **data)
    return merged
