import pytest

from aco.interventions.library import INTERVENTIONS, apply_intervention


def test_curtailment_reduces_power_and_has_cost():
    state = {"power_mw": 10.0}
    new_state, cost = apply_intervention("curtailment", state, magnitude=0.2)
    assert new_state["power_mw"] == 8.0
    assert cost > 0


def test_intervention_rejects_unsafe_magnitude():
    state = {"power_mw": 10.0}
    with pytest.raises(ValueError):
        apply_intervention("curtailment", state, magnitude=0.9)  # library caps curtailment at 0.3


def test_every_intervention_declares_the_state_variable_it_manipulates():
    # An intervention that doesn't declare its target can be paired with a
    # causal node it cannot touch -- e.g. "probe poa_irradiance" carried out by
    # changing the power factor. update_graph_with_intervention would then
    # sever edges into a variable that was never actually intervened on, which
    # is Pearl's mutilated graph applied to an intervention that never happened.
    baseline = {"power_mw": 10.0, "sampling_rate_hz": 1.0,
                "power_factor": 1.0, "logging_resolution_hz": 1.0}
    for name, spec in INTERVENTIONS.items():
        target = spec["target_var"]
        new_state, _cost = apply_intervention(name, baseline, spec["max_magnitude"] / 2)
        changed = sorted(k for k, v in new_state.items() if baseline.get(k) != v)
        assert changed == [target], f"{name} declares target_var={target!r} but changes {changed}"


def test_every_intervention_declares_a_duration_safety_bound():
    # Proposal Section 8.1: "temporary, limited-duration". An intervention with
    # no duration bound cannot be held for an attributable observation window
    # without leaving the safe envelope it was pre-registered under.
    for name, spec in INTERVENTIONS.items():
        assert isinstance(spec["max_duration_slots"], int)
        assert spec["max_duration_slots"] >= 1, name
