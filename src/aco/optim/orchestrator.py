"""Lyapunov drift-plus-penalty online policy tying VoI intervention choice
(proposal Section 6.2) to per-slot CVaR-constrained allocation (Section
6.4/8.3) into one closed loop (Section 7).

The loop is closed in the sense of proposal Section 8.4: after an intervention
is executed, the observations that follow it are fed back into the Active
Causal Semantic Event Graph and the Causal World Model is refreshed, so the
orchestrator's causal model actually improves as a result of having acted.

The contract the rest of the pipeline relies on:

    one .step() == one new observation, taken as the last row of `df`

The orchestrator counts those observations itself rather than differencing
`len(df)`, so an intervention's window still fills when the caller feeds a
fixed-size sliding frame (where `len(df)` never grows) or rebases it smaller.
"""
import pandas as pd

from aco.causal.graph import update_graph_with_intervention
from aco.interventions.library import INTERVENTIONS, apply_intervention
from aco.interventions.voi import select_best_intervention
from aco.optim.dro_allocator import solve_slot

# Per-slot decay applied to the virtual queue when a slot doesn't violate its
# CVaR limit, so the queue relaxes back toward zero rather than only ever
# holding steady -- standard drift-plus-penalty queue mechanics.
QUEUE_DECAY = 0.01

# Post-intervention observations required before the learning loop closes. One
# tick yields one row, which cannot support a PCMCI+ refit, so the orchestrator
# holds the intervention open until a usable observation window exists.
DEFAULT_MIN_POST_OBS = 50


class ActiveOrchestrator:
    """Maintains a virtual queue of CVaR-constraint backlog across slots and,
    each slot:

    1. records the new observation against any intervention still in flight,
       and once its window is full, folds it back into the causal graph and
       refits the world model (Section 8.4);
    2. selects and applies the next intervention -- but only when none is in
       flight, so every window stays attributable to exactly one intervention;
    3. solves the resulting resource allocation.
    """

    def __init__(self, V: float, cvar_alpha: float, cvar_limit: float,
                 min_post_obs: int = DEFAULT_MIN_POST_OBS):
        self.V = V
        self.cvar_alpha = cvar_alpha
        self.cvar_limit = cvar_limit
        self.min_post_obs = min_post_obs
        # The graph the orchestrator has learned so far. None until the first
        # causal update, after which it supersedes the caller's prior.
        self.graph = None
        self._queue = 0.0
        # {"node", "target_var", "post": [one-row frames]} while an
        # intervention is being observed; None otherwise.
        self._pending = None

    def step(
        self, site_states: dict, world_model, df, graph,
        node_candidates: list, var_names: list, tau_max: int = 1,
    ) -> dict:
        site_ids = list(site_states.keys())
        active_graph = self.graph if self.graph is not None else graph

        causal_update = None
        self._record_observation(df)
        learned = self._close_learning_loop(world_model, df, active_graph, var_names, tau_max)
        if learned is not None:
            active_graph, causal_update = learned
            self.graph = active_graph

        # Probe only when nothing is still being observed. Attribution needs a
        # clean window: a second intervention landing mid-window makes the
        # observations unattributable to the first. Skipping selection outright
        # also avoids paying for select_best_intervention during the wait.
        best = None
        if self._pending is None:
            best = select_best_intervention(
                world_model, df, active_graph, node_candidates, var_names, tau_max=tau_max
            )
        intervention_result = None
        if best is not None:
            node, name, magnitude = best
            for sid in site_ids:
                site_states[sid], _cost = apply_intervention(name, site_states[sid], magnitude)
            intervention_result = best
            self._pending = {
                "node": node,
                # What the intervention actually manipulates, which is what
                # licenses severing this variable's incoming edges later --
                # not `node`, which is only what the VoI layer was curious about.
                "target_var": INTERVENTIONS[name]["target_var"],
                "post": [],
            }

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
            "graph": active_graph,
            "causal_update": causal_update,
        }

    def _record_observation(self, df) -> None:
        """Attribute this slot's new observation to the intervention in flight.

        Takes `df`'s last row -- see the module docstring's contract. Counting
        observations here rather than differencing `len(df)` is what keeps the
        window filling under a sliding or rebased frame.
        """
        if self._pending is not None and len(df) > 0:
            self._pending["post"].append(df.tail(1))

    def _close_learning_loop(self, world_model, df, graph, var_names, tau_max):
        """Fold a completed intervention's observations back into the graph and
        world model (proposal Section 8.4).

        Returns `(updated_graph, update_record)`, or None while no intervention
        is in flight or its window is still filling. This uses the *observed*
        post-intervention data, unlike
        `CausalWorldModel.estimate_uncertainty_reduction`, which refits on a
        simulated probe to score a candidate before executing it.
        """
        if self._pending is None or len(self._pending["post"]) < self.min_post_obs:
            return None

        post_df = pd.concat(self._pending["post"], ignore_index=True)
        target_var = self._pending["target_var"]
        updated = update_graph_with_intervention(
            graph, target_var, post_df, var_names=var_names, tau_max=tau_max,
        )

        # Refresh the twin onto the updated structure. models is cleared first
        # so a node that lost all its parents in the update doesn't keep a
        # stale regressor fitted against edges that no longer exist.
        world_model.graph = updated
        world_model.models = {}
        world_model.fit(df)

        record = {
            "node": self._pending["node"],
            "target_var": target_var,
            "n_post_obs": len(post_df),
        }
        self._pending = None
        return updated, record
