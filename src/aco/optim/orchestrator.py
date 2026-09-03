"""Lyapunov drift-plus-penalty online policy tying VoI intervention choice
(proposal Section 6.2) to per-slot CVaR-constrained allocation (Section
6.4/8.3) into one closed loop (Section 7).
"""
from aco.interventions.library import apply_intervention
from aco.interventions.voi import select_best_intervention
from aco.optim.dro_allocator import solve_slot

# Per-slot decay applied to the virtual queue when a slot doesn't violate its
# CVaR limit, so the queue relaxes back toward zero rather than only ever
# holding steady -- standard drift-plus-penalty queue mechanics.
QUEUE_DECAY = 0.01


class ActiveOrchestrator:
    """Maintains a virtual queue of CVaR-constraint backlog across slots and,
    each slot, selects the best-scoring intervention (if any), applies it,
    and solves the resulting resource allocation.
    """

    def __init__(self, V: float, cvar_alpha: float, cvar_limit: float):
        self.V = V
        self.cvar_alpha = cvar_alpha
        self.cvar_limit = cvar_limit
        self._queue = 0.0

    def step(
        self, site_states: dict, world_model, df, graph,
        node_candidates: list, var_names: list, tau_max: int = 1,
    ) -> dict:
        site_ids = list(site_states.keys())

        best = select_best_intervention(world_model, df, graph, node_candidates, var_names, tau_max=tau_max)
        intervention_result = None
        if best is not None:
            node, name, magnitude = best
            for sid in site_ids:
                site_states[sid], _cost = apply_intervention(name, site_states[sid], magnitude)
            intervention_result = best

        available_power = sum(site_states[sid]["power_mw"] for sid in site_ids)
        demand = [site_states[sid]["compute_demand"] for sid in site_ids]
        cost = [site_states[sid]["cost_per_unit"] * self.V for sid in site_ids]
        risk_samples = [[site_states[sid]["risk_sample"][0] for sid in site_ids]]

        solved = solve_slot(available_power, demand, cost, risk_samples, self.cvar_alpha, self.cvar_limit)
        violation = max(0.0, solved["cvar"] - self.cvar_limit)
        self._queue = max(0.0, self._queue + violation - QUEUE_DECAY)

        return {
            "allocation": dict(zip(site_ids, solved["allocation"])),
            "intervention": intervention_result,
            "queue_backlog": self._queue,
        }
