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
