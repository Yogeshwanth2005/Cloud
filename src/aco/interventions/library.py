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


INTERVENTIONS = {
    "curtailment": {"apply": _apply_curtailment, "cost_fn": lambda m: 5.0 * m, "max_magnitude": 0.3},
    "high_res_sampling": {"apply": _apply_sampling, "cost_fn": lambda m: 0.5 * m, "max_magnitude": 4.0},
    "setpoint_change": {"apply": _apply_setpoint, "cost_fn": lambda m: 2.0 * m, "max_magnitude": 0.1},
    "high_res_logging": {"apply": _apply_logging, "cost_fn": lambda m: 0.2 * m, "max_magnitude": 9.0},
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
