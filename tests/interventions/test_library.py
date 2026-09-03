import pytest

from aco.interventions.library import apply_intervention


def test_curtailment_reduces_power_and_has_cost():
    state = {"power_mw": 10.0}
    new_state, cost = apply_intervention("curtailment", state, magnitude=0.2)
    assert new_state["power_mw"] == 8.0
    assert cost > 0


def test_intervention_rejects_unsafe_magnitude():
    state = {"power_mw": 10.0}
    with pytest.raises(ValueError):
        apply_intervention("curtailment", state, magnitude=0.9)  # library caps curtailment at 0.3
