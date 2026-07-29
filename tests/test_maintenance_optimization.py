from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from aeromaintain.optimization import (
    Engine,
    OptimizationError,
    Scenario,
    build_baseline_schedule,
    evaluate_retrospective,
    evaluate_schedule,
    generate_scenario,
    maintenance,
    optimize_run,
    scenario_variant,
    solve_cp_sat,
)
from aeromaintain.optimization.maintenance import _no_solution_result


def _config() -> dict:
    return {
        "metadata": {"synthetic": True, "seed": 42},
        "planning": {
            "fleet_size": 20,
            "horizon_days": 30,
            "operating_demand_fraction": 0.8,
        },
        "generation": {
            "cycles_per_day": {"min": 1, "max": 4},
            "maintenance_duration_days": {"min": 2, "max": 5},
            "technicians_required": {"min": 2, "max": 6},
            "kit_quantity": {"min": 1, "max": 2},
        },
        "resources": {
            "teams": {
                "team_A": {"technicians": 6},
                "team_B": {"technicians": 6},
            },
            "parts": {
                "types": ["kit_A", "kit_B", "kit_C"],
                "initial_units_per_type": 4,
                "replenishment": {"days": [10, 20], "units_per_type": 3},
            },
        },
        "cost_units": {
            "planned_maintenance": 100,
            "emergency_maintenance": 500,
            "unused_predicted_cycle": 1,
            "low_risk_deferral": 150,
        },
    }


def _predictions(count: int = 25) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": range(1, count + 1),
            "cycle": [80 + index for index in range(count)],
            "prediction": [50.5 + index for index in range(count)],
            "interval_low": [float(count - index) for index in range(count)],
            "interval_high": [80.0 + index for index in range(count)],
            "risk_band": ["critical"] * count,
        }
    )


def _engine(
    engine_id: int,
    *,
    safe_due_day: int = 0,
    due_in_horizon: bool = True,
    cycles_per_day: int = 1,
    duration_days: int = 2,
    technicians: int = 2,
    kit_type: str = "kit_A",
) -> Engine:
    return Engine(
        engine_id=engine_id,
        observed_cycle=90,
        prediction=10.0,
        interval_low=float(safe_due_day + 1),
        interval_high=20.0,
        risk_band="critical",
        cycles_per_day=cycles_per_day,
        duration_days=duration_days,
        technicians=technicians,
        kit_type=kit_type,
        kit_quantity=1,
        point_rul_cycles=10,
        lower_rul_cycles=safe_due_day + 1,
        safe_due_day=safe_due_day,
        due_in_horizon=due_in_horizon,
    )


def _scenario(*engines: Engine, demand: float = 0.0) -> Scenario:
    return Scenario(
        schema_version=1,
        generator_version="test",
        seed=42,
        horizon_days=5,
        operating_demand_fraction=demand,
        engines=tuple(engines),
        teams=(("team_A", 6), ("team_B", 6)),
        bays=("bay_1", "bay_2"),
        part_types=("kit_A", "kit_B", "kit_C"),
        initial_parts_per_type=4,
        replenishment_days=(2, 4),
        replenishment_units_per_type=3,
        planned_maintenance_cost=100,
        emergency_maintenance_cost=500,
        unused_predicted_cycle_cost=1,
        low_risk_deferral_cost=150,
    )


def test_scenario_is_deterministic_risk_sorted_and_truth_free() -> None:
    predictions = _predictions()
    first = generate_scenario(predictions, _config())
    second = generate_scenario(predictions.sample(frac=1, random_state=7), _config())

    assert first == second
    assert len(first.engines) == 20
    assert [engine.engine_id for engine in first.engines] == list(range(25, 5, -1))
    assert first.to_dict()["metadata"]["truth_fields_present"] is False
    for engine in first.engines:
        expected_due = max(
            0, math.floor(engine.lower_rul_cycles / engine.cycles_per_day) - 1
        )
        assert engine.safe_due_day == expected_due


@pytest.mark.parametrize("truth_column", ["rul_true", "true_rul", "actual_rul"])
def test_scenario_rejects_truth_fields(truth_column: str) -> None:
    predictions = _predictions()
    predictions[truth_column] = 99

    with pytest.raises(OptimizationError, match="forbidden truth"):
        generate_scenario(predictions, _config())


def test_scenario_rejects_invalid_inputs() -> None:
    duplicate = _predictions()
    duplicate.loc[1, "unit_id"] = duplicate.loc[0, "unit_id"]
    with pytest.raises(OptimizationError, match="one row per engine"):
        generate_scenario(duplicate, _config())

    invalid_interval = _predictions()
    invalid_interval.loc[0, "interval_high"] = -1
    with pytest.raises(OptimizationError, match="ordered and non-negative"):
        generate_scenario(invalid_interval, _config())


def test_baselines_share_resources_and_common_evaluator() -> None:
    scenario = _scenario(
        _engine(1, technicians=4),
        _engine(2, technicians=4, kit_type="kit_B"),
        _engine(3, safe_due_day=4, due_in_horizon=False),
    )
    for policy in ("reactive", "fixed_90", "predicted_rul_30"):
        schedule = build_baseline_schedule(scenario, policy)
        metrics = evaluate_schedule(scenario, schedule)

        assert metrics["policy"] == policy
        assert metrics["operating_capacity_shortfall_cycles"] == 0
        assert {job["engine_id"] for job in schedule["jobs"]} == {1, 2, 3}


def test_hand_solvable_cp_sat_fixture_starts_due_job_on_day_zero() -> None:
    scenario = _scenario(_engine(1))

    schedule = solve_cp_sat(scenario, time_limit_seconds=5)
    metrics = evaluate_schedule(scenario, schedule)

    assert schedule["solver_status"] == "OPTIMAL"
    assert schedule["lexicographic_optimality"] == "proven"
    assert schedule["jobs"][0]["start_day"] == 0
    assert schedule["jobs"][0]["end_day"] == 2
    assert metrics["due_deferrals"] == 0
    assert metrics["late_days"] == 0


def test_cp_sat_enforces_team_bay_parts_operations_and_horizon() -> None:
    scenario = _scenario(
        _engine(1, technicians=4, cycles_per_day=1),
        _engine(2, technicians=4, cycles_per_day=1),
        _engine(3, technicians=4, cycles_per_day=2, kit_type="kit_B"),
        demand=0.5,
    )

    schedule = solve_cp_sat(scenario, time_limit_seconds=5)
    metrics = evaluate_schedule(scenario, schedule)

    assert schedule["solver_status"] == "OPTIMAL"
    assert metrics["schedule_available"] is True
    assert all(
        job["end_day"] <= scenario.horizon_days
        for job in schedule["jobs"]
        if job["status"] == "scheduled"
    )


def test_no_solution_status_never_contains_a_plausible_schedule() -> None:
    scenario = _scenario(_engine(1), demand=1.0)
    infeasible = solve_cp_sat(
        scenario,
        time_limit_seconds=5,
        require_all_due=True,
    )
    unknown = _no_solution_result(
        status="UNKNOWN",
        stage1_status="UNKNOWN",
        solve_time_seconds=0.0,
    )

    assert infeasible["solver_status"] == "INFEASIBLE"
    assert infeasible["jobs"] == []
    assert evaluate_schedule(scenario, infeasible)["schedule_available"] is False
    assert unknown["jobs"] == []
    assert evaluate_schedule(scenario, unknown)["solver_status"] == "UNKNOWN"


def test_capacity_variants_change_only_capacity_contract() -> None:
    base = _scenario(_engine(1))
    constrained = scenario_variant(base, "constrained")
    expanded = scenario_variant(base, "expanded")

    assert constrained.engines == base.engines == expanded.engines
    assert len(constrained.bays) == 1
    assert constrained.operating_demand_fraction == 0.9
    assert len(expanded.bays) == 3
    assert expanded.operating_demand_fraction == 0.7
    with pytest.raises(OptimizationError, match="Unknown capacity"):
        scenario_variant(base, "imaginary")


def test_retrospective_truth_is_separate_and_requires_exact_fleet() -> None:
    scenario = _scenario(_engine(1), _engine(2))
    schedule = build_baseline_schedule(scenario, "reactive")

    retrospective = evaluate_retrospective(scenario, schedule, {1: 3, 2: 9})

    assert retrospective == {
        "unplanned_failures": 2,
        "emergency_cost_units": 1000,
    }
    with pytest.raises(OptimizationError, match="match the scenario fleet"):
        evaluate_retrospective(scenario, schedule, {1: 3})


def test_evaluator_rejects_resource_violation_and_unknown_policy() -> None:
    scenario = _scenario(_engine(1))
    invalid = {
        "policy": "invalid",
        "solver_status": "NOT_APPLICABLE",
        "lexicographic_optimality": "not_applicable",
        "solve_time_seconds": 0.0,
        "jobs": [
            {
                "engine_id": 1,
                "status": "scheduled",
                "start_day": 4,
                "end_day": 6,
                "team_id": "team_A",
                "bay_id": "bay_1",
                "reason": "invalid",
            }
        ],
    }
    with pytest.raises(OptimizationError, match="resource invariant"):
        evaluate_schedule(scenario, invalid)
    with pytest.raises(OptimizationError, match="Unknown baseline"):
        build_baseline_schedule(scenario, "imaginary")


def test_operating_demand_can_force_explicit_deferral() -> None:
    engine = _engine(1, due_in_horizon=False)
    scenario = replace(_scenario(engine), operating_demand_fraction=1.0)

    schedule = build_baseline_schedule(scenario, "fixed_90")

    assert schedule["jobs"][0]["status"] == "deferred"
    assert schedule["jobs"][0]["reason"] == "resource_infeasible_within_horizon"


def test_optimize_run_persists_hashed_truth_free_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    run_id = "fixture"
    (tmp_path / "runs" / run_id).mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "scenario.yaml").write_text(
        "fixture: true\n", encoding="utf-8"
    )
    predictions = _predictions(20)
    truth = {engine_id: 20 for engine_id in range(1, 21)}
    monkeypatch.setattr(
        maintenance,
        "_load_verified_official_predictions",
        lambda *args, **kwargs: (
            predictions,
            truth,
            {
                "model_lock_sha256": "a",
                "predictions_sha256": "b",
                "evaluation_manifest_sha256": "c",
            },
        ),
    )
    monkeypatch.setattr(
        maintenance,
        "_load_scenario_config",
        lambda *args, **kwargs: _config(),
    )

    def quick_solver(scenario, **kwargs):
        return build_baseline_schedule(scenario, "predicted_rul_30") | {
            "policy": "cp_sat",
            "solver_status": "FEASIBLE",
            "stage1_status": "FEASIBLE",
            "lexicographic_optimality": "unproven",
        }

    monkeypatch.setattr(maintenance, "solve_cp_sat", quick_solver)

    result = optimize_run(tmp_path, run_id=run_id)
    scenario_payload = json.loads(
        (result.output_dir / "scenario.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (result.output_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert scenario_payload["metadata"]["truth_fields_present"] is False
    assert "rul_true" not in json.dumps(scenario_payload)
    assert manifest["status"] == "optimization_complete"
    assert len(result.policy_comparison) == 4
    assert len(result.capacity_comparison) == 3
    with pytest.raises(OptimizationError, match="Refusing to overwrite"):
        optimize_run(tmp_path, run_id=run_id)
