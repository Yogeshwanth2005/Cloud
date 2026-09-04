"""Residual causal uncertainty over a discovered graph (proposal Sections 8.4
and 10.1).

Measured over the *candidate* edge set -- every ordered pair of variables --
rather than over the edges that happen to have been discovered so far.

The fixed denominator is the whole point. An average taken over discovered
edges has a denominator that grows with its numerator, so it reads 1.0 while
nothing is known and drops to nearly 0 the moment a single edge is found:
measured on the standard fixture, 1.0000 at 300 rows, 0.0199 at 301, 0.0000
from 303 on. That is a cliff, and a cliff cannot steer anything -- it made the
value-of-information signal binary and would make the optimizer's ambiguity set
(Section 6.4) a step function. Dividing by the number of pairs that *could*
carry an edge makes each discovery move the measure by 1/n_pairs, which is a
gradient the rest of the system can respond to.
"""

# An undiscovered pair contributes the maximum. This is a modelling choice
# rather than a p-value: "we have not established this edge" is treated as full
# residual uncertainty about it, which is what makes an empty graph read 1.0
# and a fully determined one read 0.0.
UNDISCOVERED_UNCERTAINTY = 1.0


def _pair_uncertainty(graph, u, v) -> float:
    if graph.has_edge(u, v):
        return float(graph[u][v].get("pval", UNDISCOVERED_UNCERTAINTY))
    return UNDISCOVERED_UNCERTAINTY


def edge_uncertainty(graph, var_names: list) -> float:
    """Mean residual uncertainty over every ordered pair of `var_names`.

    Returns 1.0 when nothing is known and approaches 0.0 as edges are
    established with low p-values. Pairs absent from `graph` count as fully
    uncertain, so adding a variable the model knows nothing about correctly
    raises the measure.
    """
    pairs = [(u, v) for u in var_names for v in var_names if u != v]
    if not pairs:
        return UNDISCOVERED_UNCERTAINTY
    return sum(_pair_uncertainty(graph, u, v) for u, v in pairs) / len(pairs)


def node_uncertainty(graph, node: str, var_names: list) -> float:
    """Mean residual uncertainty over the ordered pairs touching `node`.

    This is what the value-of-information criterion scores a probe against:
    how much is still unknown about the relationships the probed variable
    participates in, in both directions.
    """
    pairs = [(u, v) for u in var_names for v in var_names
             if u != v and node in (u, v)]
    if not pairs:
        return UNDISCOVERED_UNCERTAINTY
    return sum(_pair_uncertainty(graph, u, v) for u, v in pairs) / len(pairs)
