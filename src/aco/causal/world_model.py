import warnings

import networkx as nx
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


class CausalWorldModel:
    """Per-node structural equation model over a Phase-3 causal graph.

    One GradientBoostingRegressor per node with in-edges, regressed on its
    direct parents. `do()` clamps intervened nodes to fixed values and
    propagates them through descendants in topological order, giving
    interventional predictions without a full probabilistic SCM library.
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.models = {}

    def fit(self, df: pd.DataFrame) -> None:
        for node in nx.topological_sort(self.graph):
            parents = list(self.graph.predecessors(node))
            if not parents:
                continue
            reg = GradientBoostingRegressor(random_state=0)
            reg.fit(df[parents], df[node])
            self.models[node] = (parents, reg)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for node in nx.topological_sort(self.graph):
            if node not in self.models:
                continue
            parents, reg = self.models[node]
            out[node] = reg.predict(out[parents])
        return out

    def do(self, df: pd.DataFrame, interventions: dict) -> pd.DataFrame:
        out = df.copy()
        for var, value in interventions.items():
            out[var] = value
        for node in nx.topological_sort(self.graph):
            if node in interventions or node not in self.models:
                continue
            parents, reg = self.models[node]
            out[node] = reg.predict(out[parents])
        return out

    def estimate_uncertainty_reduction(
        self, df: pd.DataFrame, node: str, magnitude: float, var_names: list,
        tau_max: int = 1, n_probe: int = 100,
    ) -> float:
        """Estimate the expected reduction in causal uncertainty from probing
        `node` (proposal Section 8.2's VoI input, formerly a caller-supplied
        placeholder). Simulates the probe by scaling `node`'s whole recent
        trajectory by `(1 + magnitude)` and propagating it through `do()` --
        a uniform rescale, not per-row noise, so the series' own autocorrelation
        (what PCMCI+ needs to orient a same-lag link at all -- see
        fit_observational_graph's docstring) survives the probe. Because the
        simulated response comes straight from the fitted regressor, it carries
        none of the real sensor noise the natural data does, so a real causal
        edge typically comes back sharper (lower pval) than the natural fit
        finds with the same amount of data. The reduction is the drop in
        `node`'s edges' average pval after refitting on that simulated batch
        via `update_graph_with_intervention`, which is what actually severs
        edges into `node` before scoring the gain.
        """
        from aco.causal.graph import fit_observational_graph, update_graph_with_intervention

        if node not in self.graph or node not in df.columns:
            return 0.0

        sample = df[var_names].dropna()
        if len(sample) < 5:
            return 0.0
        # A contiguous, time-ordered tail window, not a random row sample: PCMCI+
        # reads row order as time order, so shuffling would destroy the very lag
        # structure it needs to orient edges.
        sample = sample.tail(min(n_probe, len(sample))).reset_index(drop=True)

        probe_values = sample[node].to_numpy() * (1 + magnitude)
        simulated = self.do(sample, {node: probe_values})

        # A near-constant window (little natural variation, or a probe magnitude
        # too small to move a node off its own GBR leaf) makes ParCorr's
        # correlation undefined -- a legitimate "no signal" outcome the pval-based
        # scoring below already handles, not a bug, so the expected scipy warning
        # is suppressed rather than left to alarm callers of a research API.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="An input array is constant")
            pre_graph = fit_observational_graph(sample, var_names=var_names, tau_max=tau_max)
            updated = update_graph_with_intervention(
                pre_graph, node, simulated, var_names=var_names, tau_max=tau_max,
            )

        def _avg_pval(g):
            pvals = [d["pval"] for u, v, d in g.edges(data=True) if u == node or v == node]
            return sum(pvals) / len(pvals) if pvals else 1.0

        return max(0.0, _avg_pval(pre_graph) - _avg_pval(updated))
