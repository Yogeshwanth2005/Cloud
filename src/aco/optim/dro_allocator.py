"""Per-slot convex resource allocation with a CVaR risk constraint
(proposal Section 6.4/8.3).

Uses the Rockafellar-Uryasev convex formulation of CVaR so the risk
constraint stays linear in the decision variable and the whole slot solve
remains a single convex program `cvxpy` can solve exactly. This gives the
"distributionally robust" ambiguity handling the proposal asks for via
empirical-distribution CVaR rather than a full Wasserstein-ball DRO --
documented simplification, see PROJECT_OVERVIEW.md.
"""
import cvxpy as cp
import numpy as np

# The proposal's per-slot objective is "minimize allocation cost subject to
# budget/demand/CVaR" -- but with no term rewarding service, that objective's
# trivial optimum is to allocate nothing (cost 0). Serving compute demand has
# implicit value (the whole point of dispatching available power) that the
# proposal doesn't quantify, so this constant stands in for it: netted against
# cost_per_unit, it must dominate any realistic per-unit cost so the optimizer
# actually deploys available power to serve demand -- preferring cheap
# resources when budget or CVaR forces a trade-off -- rather than defaulting
# to zero. Research/tuning knob, not a physical constant, same status as
# INFO_VALUE_SCALE / RISK_SCALE in aco.interventions.voi.
DEMAND_FULFILLMENT_VALUE = 1000.0


def solve_slot(
    available_power_mw: float,
    compute_demand: list[float],
    cost_per_unit: list[float],
    risk_samples: list[list[float]],
    cvar_alpha: float,
    cvar_limit: float,
) -> dict:
    """Allocate available power to serve compute demand at least cost, within
    a power budget and a CVaR bound.

    `risk_samples` is a list of scenarios, each an n-length vector of
    per-unit risk realizations; CVaR is computed on `risk_samples @ x`,
    the scenario-wise risk of the resulting allocation.

    Internally solves `minimize sum((cost_per_unit - DEMAND_FULFILLMENT_VALUE)
    * x)` rather than raw cost -- see `DEMAND_FULFILLMENT_VALUE` -- but the
    returned `objective` reports the real, interpretable operational cost
    `sum(cost_per_unit * x)`.

    Returns `{"allocation": list[float], "objective": float, "cvar": float,
    "status": str}`.
    """
    n = len(compute_demand)
    x = cp.Variable(n, nonneg=True)
    risk_matrix = np.array(risk_samples)
    n_samples = risk_matrix.shape[0]
    var = cp.Variable()
    excess = cp.Variable(n_samples, nonneg=True)

    constraints = [
        cp.sum(x) <= available_power_mw,
        x <= np.array(compute_demand),
        excess >= risk_matrix @ x - var,
    ]
    cvar_expr = var + cp.sum(excess) / ((1 - cvar_alpha) * n_samples)
    constraints.append(cvar_expr <= cvar_limit)

    net_cost = np.array(cost_per_unit) - DEMAND_FULFILLMENT_VALUE
    objective = cp.Minimize(cp.sum(cp.multiply(net_cost, x)))
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if x.value is None:
        return {"allocation": [0.0] * n, "objective": float("inf"), "cvar": float("inf"), "status": problem.status}

    return {
        "allocation": x.value.tolist(),
        "objective": float(np.dot(np.array(cost_per_unit), x.value)),
        "cvar": float(cvar_expr.value),
        "status": problem.status,
    }
