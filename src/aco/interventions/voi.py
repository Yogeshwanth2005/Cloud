"""Value-of-Information under risk constraints for candidate interventions
(proposal Section 6.2).

Approximates expected information gain as the reduction in a Phase-3 graph
edge's pval, following the standard "uncertainty reduction as VoI proxy"
pattern in active causal discovery -- exact Shannon gain over the full graph
posterior is intractable at this variable count. That reduction is estimated
by a `CausalWorldModel` (`select_best_intervention`'s `world_model` argument),
not supplied by the caller, per Section 8.2. `score_intervention` nets the
scaled information gain against both the intervention's operational cost and
its risk (fraction of its own pre-registered safety bound consumed).
"""
from aco.interventions.library import INTERVENTIONS

# Converts a unit of expected uncertainty reduction into the same cost units
# INTERVENTIONS' cost_fn values are denominated in, so information gain and
# operational cost can be netted against each other in score_intervention.
# This is a research/tuning knob, not a physical constant -- Task 5.2 Step 5's
# empirical check against the world model's actual post-intervention accuracy
# gain is what validates (or should revise) this exchange rate.
INFO_VALUE_SCALE = 5.0

# Converts "fraction of an intervention's own pre-registered safety bound
# (max_magnitude) consumed" into the same cost units as INFO_VALUE_SCALE, so
# risk can be netted against information gain per proposal Section 6.2
# ("Value-of-Information under Risk Constraints"). Same research/tuning-knob
# status as INFO_VALUE_SCALE -- not a physical constant.
RISK_SCALE = 5.0

# How many future slots a sharpened causal model is expected to keep paying
# off over. Proposal Section 6.2 admits an intervention when its "long-term
# benefit (reduced future risk or cost through better causal knowledge)
# exceeds their short-term penalty" -- so the two sides of the comparison are
# asymmetric by design: the probe's cost is paid once over the slots it is
# held, while the resulting model improves every decision taken until the
# graph next changes.
#
# Without this the comparison is incoherent rather than merely mis-tuned.
# Uncertainty reduction is bounded above by 1.0, so information value could
# never exceed `INFO_VALUE_SCALE` (5.0), while any probe long enough for
# PCMCI+ to identify anything at all -- measured at ~120 slots -- costs at
# least 108. No admissible intervention could ever have positive net value.
#
# 1000 slots is ~3.5 days at 5-minute ticks. Research/tuning knob, not a
# physical constant, same status as INFO_VALUE_SCALE and RISK_SCALE.
PLANNING_HORIZON_SLOTS = 1000


def score_intervention(graph, node, expected_uncertainty_reduction, name, magnitude,
                       duration_slots=1):
    """Net value of probing `node` via intervention `name` at `magnitude`.

    Positive means "worth executing" per proposal Section 6.2: expected
    information gain (uncertainty reduction, scaled to cost units and spread
    across every graph edge the node touches) minus the intervention's
    operational cost and its risk -- the fraction of its own pre-registered
    safety bound (`max_magnitude`) this magnitude consumes. Risk is netted
    out separately from cost because they aren't the same thing: cost_fn is
    an operational dollar cost, while risk tracks how much of the safety
    margin that makes probing a live fleet defensible gets used up, and the
    two scale very differently across the registered interventions (e.g.
    high_res_logging's cost_fn is tiny relative to its 9.0 max_magnitude).
    """
    spec = INTERVENTIONS[name]
    n_edges_touched = graph.degree(node) if node in graph else 0
    # Amortised over the horizon the improved model is useful for -- see
    # PLANNING_HORIZON_SLOTS. Cost below is not amortised: it is paid once.
    info_value = (
        expected_uncertainty_reduction
        * max(n_edges_touched, 1)
        * INFO_VALUE_SCALE
        * PLANNING_HORIZON_SLOTS
    )
    # Cost accrues every slot the intervention is held -- a curtailment held for
    # three hours is three hours of lost generation, not a one-off charge.
    cost = spec["cost_fn"](magnitude) * duration_slots
    # Risk consumes two pre-registered safety margins at once: how far the
    # magnitude goes toward its bound, and how long it is held toward its own.
    risk = (
        (magnitude / spec["max_magnitude"])
        * (duration_slots / spec["max_duration_slots"])
        * RISK_SCALE
    )
    return info_value - cost - risk


def select_best_intervention(world_model, df, graph, node_candidates, var_names, tau_max=1,
                             duration_slots=1):
    """Try every (node, intervention) pair at a mid-range magnitude, return the
    best-scoring one, or None if no candidate has positive net value -- "do
    nothing" is itself a valid, and often correct, decision.

    `expected_uncertainty_reduction` is estimated by `world_model` (proposal
    Section 8.2: "the Causal World Model estimates the expected reduction in
    causal uncertainty") rather than supplied by the caller.
    """
    best = None
    best_score = 0.0
    for node in node_candidates:
        for name, spec in INTERVENTIONS.items():
            # Only interventions that actually manipulate `node` are candidates
            # for probing it. Without this an irradiance probe could be carried
            # out by changing the power factor, and the closed loop would then
            # treat irradiance as having been intervened on when nothing
            # touched it.
            if spec["target_var"] != node:
                continue
            # An intervention that cannot be held for the whole observation
            # window would leave that window partly unclamped, so the graph
            # update could not treat it as interventional data at all.
            if duration_slots > spec["max_duration_slots"]:
                continue
            magnitude = spec["max_magnitude"] / 2
            reduction = world_model.estimate_uncertainty_reduction(
                df, node, magnitude, var_names, tau_max=tau_max,
            )
            score = score_intervention(
                graph, node, expected_uncertainty_reduction=reduction, name=name,
                magnitude=magnitude, duration_slots=duration_slots,
            )
            if score > best_score:
                best_score = score
                best = (node, name, magnitude)
    return best
