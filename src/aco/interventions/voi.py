"""Value-of-Information scoring for candidate interventions (proposal Section 8.2).

Approximates expected information gain as the reduction in a Phase-3 graph
edge's pval, following the standard "uncertainty reduction as VoI proxy"
pattern in active causal discovery -- exact Shannon gain over the full graph
posterior is intractable at this variable count.
"""
from aco.interventions.library import INTERVENTIONS

# Converts a unit of expected uncertainty reduction into the same cost units
# INTERVENTIONS' cost_fn values are denominated in, so information gain and
# operational cost can be netted against each other in score_intervention.
# This is a research/tuning knob, not a physical constant -- Task 5.2 Step 5's
# empirical check against the world model's actual post-intervention accuracy
# gain is what validates (or should revise) this exchange rate.
INFO_VALUE_SCALE = 5.0


def score_intervention(graph, node, current_uncertainty, expected_uncertainty_reduction, name, magnitude):
    """Net value of probing `node` via intervention `name` at `magnitude`.

    Positive means "worth executing" per proposal Section 8.2: expected
    information gain (uncertainty reduction, scaled to cost units and spread
    across every graph edge the node touches) minus the intervention's cost.
    """
    n_edges_touched = graph.degree(node) if node in graph else 0
    info_value = expected_uncertainty_reduction * max(n_edges_touched, 1) * INFO_VALUE_SCALE
    cost = INTERVENTIONS[name]["cost_fn"](magnitude)
    return info_value - cost


def select_best_intervention(graph, node_candidates, uncertainty_estimates):
    """Try every (node, intervention) pair at a mid-range magnitude, return the
    best-scoring one, or None if no candidate has positive net value -- "do
    nothing" is itself a valid, and often correct, decision.
    """
    best = None
    best_score = 0.0
    for node in node_candidates:
        uncertainty = uncertainty_estimates.get(node, 0.0)
        for name, spec in INTERVENTIONS.items():
            magnitude = spec["max_magnitude"] / 2
            score = score_intervention(
                graph, node, uncertainty_estimates, expected_uncertainty_reduction=uncertainty,
                name=name, magnitude=magnitude,
            )
            if score > best_score:
                best_score = score
                best = (node, name, magnitude)
    return best
