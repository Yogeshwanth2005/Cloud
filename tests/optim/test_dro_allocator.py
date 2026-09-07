import pytest

from aco.optim.dro_allocator import solve_slot


def test_solve_slot_respects_power_budget_and_cvar():
    result = solve_slot(
        available_power_mw=10.0,
        compute_demand=[6.0, 6.0],
        cost_per_unit=[1.0, 2.0],
        risk_samples=[[0.1, 0.2], [0.3, 0.1], [0.2, 0.4]],
        cvar_alpha=0.9,
        cvar_limit=1.5,
    )
    assert result["status"] == "optimal"
    assert sum(result["allocation"]) <= 10.0 + 1e-6
    assert result["cvar"] <= 1.5 + 1e-6


def test_solve_slot_prefers_cheaper_resource():
    result = solve_slot(
        available_power_mw=20.0,
        compute_demand=[5.0, 5.0],
        cost_per_unit=[1.0, 5.0],
        risk_samples=[[0.0, 0.0]],
        cvar_alpha=0.9,
        cvar_limit=100.0,
    )
    assert result["allocation"][0] == pytest.approx(5.0, abs=1e-4)
