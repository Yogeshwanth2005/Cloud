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
    An edge exists wherever p_matrix[i, j, tau] < 0.05.
    """
    values = df[var_names].to_numpy()
    dataframe = pp.DataFrame(values, var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
    results = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=0.05)

    graph = nx.DiGraph()
    graph.add_nodes_from(var_names)
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    n_vars = len(var_names)
    for i in range(n_vars):
        for j in range(n_vars):
            for tau in range(tau_max + 1):
                if i == j and tau == 0:
                    continue
                if p_matrix[i, j, tau] < 0.05:
                    graph.add_edge(
                        var_names[i], var_names[j],
                        weight=float(val_matrix[i, j, tau]),
                        pval=float(p_matrix[i, j, tau]),
                        lag=tau,
                    )
    return graph
