"""Safe intervention library (proposal Section 8.1).

Each intervention is pre-registered with a safety bound (`max_magnitude`) so
the orchestrator can never exceed it -- this is what makes probing a live PV
fleet defensible rather than an open-ended actuation.
"""


def _apply_curtailment(state, magnitude):
    new_state = dict(state)
    new_state["power_mw"] = state["power_mw"] * (1 - magnitude)
    return new_state


def _apply_sampling(state, magnitude):
    new_state = dict(state)
    new_state["sampling_rate_hz"] = state.get("sampling_rate_hz", 1.0) * (1 + magnitude)
    return new_state


def _apply_setpoint(state, magnitude):
    new_state = dict(state)
    new_state["power_factor"] = max(0.8, min(1.0, state.get("power_factor", 1.0) - magnitude))
    return new_state


def _apply_logging(state, magnitude):
    new_state = dict(state)
    new_state["logging_resolution_hz"] = state.get("logging_resolution_hz", 1.0) * (1 + magnitude)
    return new_state


# `target_var` is the causal-graph variable each intervention actually
# manipulates -- the state key its `apply` writes. It is what makes an
# intervention a *causal* intervention on a named node rather than an
# undirected perturbation: `select_best_intervention` will only probe a node
# with an intervention that can reach it, and the closed loop hands this name
# (not the node the VoI layer was curious about) to
# `update_graph_with_intervention`, which severs incoming edges on the strength
# of the claim that this variable was set rather than caused.
INTERVENTIONS = {
    "curtailment": {
        "apply": _apply_curtailment, "cost_fn": lambda m: 5.0 * m,
        "max_magnitude": 0.3, "target_var": "power_mw",
    },
    "high_res_sampling": {
        "apply": _apply_sampling, "cost_fn": lambda m: 0.5 * m,
        "max_magnitude": 4.0, "target_var": "sampling_rate_hz",
    },
    "setpoint_change": {
        "apply": _apply_setpoint, "cost_fn": lambda m: 2.0 * m,
        "max_magnitude": 0.1, "target_var": "power_factor",
    },
    "high_res_logging": {
        "apply": _apply_logging, "cost_fn": lambda m: 0.2 * m,
        "max_magnitude": 9.0, "target_var": "logging_resolution_hz",
    },
}


def apply_intervention(name: str, state: dict, magnitude: float) -> tuple[dict, float]:
    """Apply a pre-registered intervention, enforcing its safety bound.

    Returns (new_state, cost). Raises ValueError if magnitude exceeds the
    intervention's max_magnitude.
    """
    spec = INTERVENTIONS[name]
    if magnitude > spec["max_magnitude"]:
        raise ValueError(f"{name} magnitude {magnitude} exceeds safe max {spec['max_magnitude']}")
    new_state = spec["apply"](state, magnitude)
    cost = spec["cost_fn"](magnitude)
    return new_state, cost
