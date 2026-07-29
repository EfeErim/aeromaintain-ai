"""Synthetic maintenance scenarios, policies, metrics, and CP-SAT scheduling."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from ortools.sat.python import cp_model

from aeromaintain.data.pipeline import sha256_file

SCENARIO_SCHEMA_VERSION = 1
GENERATOR_VERSION = "fd001-synthetic-maintenance-v1"
DECISION_COLUMNS = (
    "unit_id",
    "cycle",
    "prediction",
    "interval_low",
    "interval_high",
    "risk_band",
)
TRUTH_COLUMNS = frozenset(
    {"rul_true", "true_rul", "actual_rul", "interval_contains_true_rul"}
)
CAPACITY_SCENARIOS = {
    "constrained": (1, 0.90),
    "base": (2, 0.80),
    "expanded": (3, 0.70),
}
SOLVER_STATUSES_WITH_SOLUTION = frozenset({"OPTIMAL", "FEASIBLE"})


class OptimizationError(RuntimeError):
    """Raised when the M3 decision contract cannot be satisfied."""


@dataclass(frozen=True)
class Engine:
    """Prediction and synthetic planning fields for one engine."""

    engine_id: int
    observed_cycle: int
    prediction: float
    interval_low: float
    interval_high: float
    risk_band: str
    cycles_per_day: int
    duration_days: int
    technicians: int
    kit_type: str
    kit_quantity: int
    point_rul_cycles: int
    lower_rul_cycles: int
    safe_due_day: int
    due_in_horizon: bool


@dataclass(frozen=True)
class Scenario:
    """Truth-free synthetic inputs shared by every maintenance policy."""

    schema_version: int
    generator_version: str
    seed: int
    horizon_days: int
    operating_demand_fraction: float
    engines: tuple[Engine, ...]
    teams: tuple[tuple[str, int], ...]
    bays: tuple[str, ...]
    part_types: tuple[str, ...]
    initial_parts_per_type: int
    replenishment_days: tuple[int, ...]
    replenishment_units_per_type: int
    planned_maintenance_cost: int
    emergency_maintenance_cost: int
    unused_predicted_cycle_cost: int
    low_risk_deferral_cost: int

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""
        payload = asdict(self)
        payload["teams"] = [
            {"team_id": name, "technicians": technicians}
            for name, technicians in self.teams
        ]
        payload["metadata"] = {
            "synthetic": True,
            "warning": "All operational, resource, and cost fields are synthetic.",
            "truth_fields_present": False,
        }
        return payload


@dataclass(frozen=True)
class OptimizationResult:
    """CLI-safe output summary for a completed M3 run."""

    run_id: str
    output_dir: Path
    policy_comparison: tuple[dict[str, Any], ...]
    capacity_comparison: tuple[dict[str, Any], ...]


def _load_scenario_config(project_root: Path) -> dict[str, Any]:
    path = project_root / "configs" / "scenario.yaml"
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise OptimizationError(f"Cannot load scenario config: {path}") from exc
    if not isinstance(config, dict):
        raise OptimizationError("Scenario config must be a mapping")
    return config


def _assert_truth_free_columns(frame: pd.DataFrame) -> None:
    columns = {str(column).casefold() for column in frame.columns}
    forbidden = sorted(columns & TRUTH_COLUMNS)
    if forbidden:
        raise OptimizationError(
            "Optimizer input contains forbidden truth fields: " + ", ".join(forbidden)
        )
    missing = [column for column in DECISION_COLUMNS if column not in frame.columns]
    if missing:
        raise OptimizationError(
            "Optimizer input is missing required fields: " + ", ".join(missing)
        )


def generate_scenario(
    predictions: pd.DataFrame,
    config: dict[str, Any],
    *,
    capacity_name: str = "base",
) -> Scenario:
    """Select the riskiest engines and generate deterministic synthetic fields."""
    _assert_truth_free_columns(predictions)
    if capacity_name not in CAPACITY_SCENARIOS:
        raise OptimizationError(f"Unknown capacity scenario: {capacity_name}")
    planning = config["planning"]
    generation = config["generation"]
    resources = config["resources"]
    costs = config["cost_units"]
    seed = int(config["metadata"]["seed"])
    horizon = int(planning["horizon_days"])
    fleet_size = int(planning["fleet_size"])
    if predictions["unit_id"].duplicated().any():
        raise OptimizationError("Optimizer input must contain one row per engine")
    numeric = predictions.loc[
        :, ["cycle", "prediction", "interval_low", "interval_high"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise OptimizationError("Optimizer input contains non-finite values")
    if (predictions["interval_low"] < 0).any() or (
        predictions["interval_high"] < predictions["interval_low"]
    ).any():
        raise OptimizationError("Optimizer intervals must be ordered and non-negative")

    selected = (
        predictions.sort_values(["interval_low", "unit_id"], kind="mergesort")
        .head(fleet_size)
        .reset_index(drop=True)
    )
    if len(selected) != fleet_size:
        raise OptimizationError(f"Expected at least {fleet_size} prediction rows")
    rng = np.random.default_rng(seed)
    cycles_range = generation["cycles_per_day"]
    duration_range = generation["maintenance_duration_days"]
    technician_range = generation["technicians_required"]
    quantity_range = generation["kit_quantity"]
    cycles_per_day = rng.integers(
        int(cycles_range["min"]),
        int(cycles_range["max"]) + 1,
        size=fleet_size,
    )
    duration_days = rng.integers(
        int(duration_range["min"]),
        int(duration_range["max"]) + 1,
        size=fleet_size,
    )
    technicians = rng.integers(
        int(technician_range["min"]),
        int(technician_range["max"]) + 1,
        size=fleet_size,
    )
    kit_types = rng.choice(tuple(resources["parts"]["types"]), size=fleet_size)
    kit_quantities = rng.integers(
        int(quantity_range["min"]),
        int(quantity_range["max"]) + 1,
        size=fleet_size,
    )
    engines: list[Engine] = []
    for index, row in selected.iterrows():
        point = math.floor(float(row["prediction"]))
        lower = math.floor(float(row["interval_low"]))
        due = max(0, math.floor(lower / int(cycles_per_day[index])) - 1)
        engines.append(
            Engine(
                engine_id=int(row["unit_id"]),
                observed_cycle=int(row["cycle"]),
                prediction=float(row["prediction"]),
                interval_low=float(row["interval_low"]),
                interval_high=float(row["interval_high"]),
                risk_band=str(row["risk_band"]),
                cycles_per_day=int(cycles_per_day[index]),
                duration_days=int(duration_days[index]),
                technicians=int(technicians[index]),
                kit_type=str(kit_types[index]),
                kit_quantity=int(kit_quantities[index]),
                point_rul_cycles=point,
                lower_rul_cycles=lower,
                safe_due_day=due,
                due_in_horizon=due < horizon,
            )
        )

    bay_count, demand_fraction = CAPACITY_SCENARIOS[capacity_name]
    team_rows = tuple(
        (str(name), int(value["technicians"]))
        for name, value in sorted(resources["teams"].items())
    )
    parts = resources["parts"]
    return Scenario(
        schema_version=SCENARIO_SCHEMA_VERSION,
        generator_version=GENERATOR_VERSION,
        seed=seed,
        horizon_days=horizon,
        operating_demand_fraction=demand_fraction,
        engines=tuple(engines),
        teams=team_rows,
        bays=tuple(f"bay_{index}" for index in range(1, bay_count + 1)),
        part_types=tuple(str(item) for item in parts["types"]),
        initial_parts_per_type=int(parts["initial_units_per_type"]),
        replenishment_days=tuple(int(day) for day in parts["replenishment"]["days"]),
        replenishment_units_per_type=int(parts["replenishment"]["units_per_type"]),
        planned_maintenance_cost=int(costs["planned_maintenance"]),
        emergency_maintenance_cost=int(costs["emergency_maintenance"]),
        unused_predicted_cycle_cost=int(costs["unused_predicted_cycle"]),
        low_risk_deferral_cost=int(costs["low_risk_deferral"]),
    )


def scenario_variant(scenario: Scenario, capacity_name: str) -> Scenario:
    """Change only bays and demand for one fixed capacity sensitivity case."""
    if capacity_name not in CAPACITY_SCENARIOS:
        raise OptimizationError(f"Unknown capacity scenario: {capacity_name}")
    bay_count, demand_fraction = CAPACITY_SCENARIOS[capacity_name]
    return replace(
        scenario,
        bays=tuple(f"bay_{index}" for index in range(1, bay_count + 1)),
        operating_demand_fraction=demand_fraction,
    )


def _empty_job(engine: Engine, reason: str) -> dict[str, Any]:
    return {
        "engine_id": engine.engine_id,
        "status": "deferred",
        "start_day": None,
        "end_day": None,
        "team_id": None,
        "bay_id": None,
        "reason": reason,
    }


def _active_on(job: dict[str, Any], day: int) -> bool:
    return job["status"] == "scheduled" and int(job["start_day"]) <= day < int(
        job["end_day"]
    )


def _assignment_feasible(
    scenario: Scenario,
    jobs: list[dict[str, Any]],
    engine: Engine,
    start_day: int,
    team_id: str,
    bay_id: str,
) -> bool:
    end_day = start_day + engine.duration_days
    if start_day < 0 or end_day > scenario.horizon_days:
        return False
    team_capacity = dict(scenario.teams)[team_id]
    total_cycles = sum(item.cycles_per_day for item in scenario.engines)
    minimum_cycles = math.ceil(total_cycles * scenario.operating_demand_fraction)
    for day in range(start_day, end_day):
        team_used = sum(
            next(
                item.technicians
                for item in scenario.engines
                if item.engine_id == job["engine_id"]
            )
            for job in jobs
            if job["team_id"] == team_id and _active_on(job, day)
        )
        if team_used + engine.technicians > team_capacity:
            return False
        if any(job["bay_id"] == bay_id and _active_on(job, day) for job in jobs):
            return False
        outage = sum(
            next(
                item.cycles_per_day
                for item in scenario.engines
                if item.engine_id == job["engine_id"]
            )
            for job in jobs
            if _active_on(job, day)
        )
        if total_cycles - outage - engine.cycles_per_day < minimum_cycles:
            return False
    starts = [
        (
            int(job["start_day"]),
            next(
                item for item in scenario.engines if item.engine_id == job["engine_id"]
            ),
        )
        for job in jobs
        if job["status"] == "scheduled"
    ]
    starts.append((start_day, engine))
    for day in range(scenario.horizon_days):
        consumed = sum(
            item.kit_quantity
            for start, item in starts
            if start <= day and item.kit_type == engine.kit_type
        )
        available = scenario.initial_parts_per_type + sum(
            scenario.replenishment_units_per_type
            for replenish_day in scenario.replenishment_days
            if replenish_day <= day
        )
        if consumed > available:
            return False
    return True


def _schedule_greedily(
    scenario: Scenario,
    policy: str,
    trigger_days: dict[int, int | None],
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    ordered = sorted(
        scenario.engines, key=lambda engine: (engine.safe_due_day, engine.engine_id)
    )
    for engine in ordered:
        trigger = trigger_days[engine.engine_id]
        if trigger is None or trigger >= scenario.horizon_days:
            jobs.append(_empty_job(engine, "not_triggered_within_horizon"))
            continue
        assigned = False
        for day in range(trigger, scenario.horizon_days - engine.duration_days + 1):
            for team_id, _ in scenario.teams:
                for bay_id in scenario.bays:
                    if _assignment_feasible(
                        scenario, jobs, engine, day, team_id, bay_id
                    ):
                        jobs.append(
                            {
                                "engine_id": engine.engine_id,
                                "status": "scheduled",
                                "start_day": day,
                                "end_day": day + engine.duration_days,
                                "team_id": team_id,
                                "bay_id": bay_id,
                                "reason": "earliest_feasible",
                            }
                        )
                        assigned = True
                        break
                if assigned:
                    break
            if assigned:
                break
        if not assigned:
            jobs.append(_empty_job(engine, "resource_infeasible_within_horizon"))
    return {
        "policy": policy,
        "solver_status": "NOT_APPLICABLE",
        "lexicographic_optimality": "not_applicable",
        "solve_time_seconds": 0.0,
        "jobs": sorted(jobs, key=lambda job: job["engine_id"]),
    }


def build_baseline_schedule(scenario: Scenario, policy: str) -> dict[str, Any]:
    """Build one truth-free reactive, fixed-90, or predicted-RUL-30 schedule."""
    if policy == "reactive":
        return {
            "policy": policy,
            "solver_status": "NOT_APPLICABLE",
            "lexicographic_optimality": "not_applicable",
            "solve_time_seconds": 0.0,
            "jobs": [
                _empty_job(engine, "reactive_waits_for_failure")
                for engine in scenario.engines
            ],
        }
    if policy == "fixed_90":
        triggers = {}
        for engine in scenario.engines:
            target = math.ceil(engine.observed_cycle / 90) * 90
            triggers[engine.engine_id] = math.ceil(
                (target - engine.observed_cycle) / engine.cycles_per_day
            )
        return _schedule_greedily(scenario, policy, triggers)
    if policy == "predicted_rul_30":
        triggers = {
            engine.engine_id: max(
                0,
                math.ceil((engine.prediction - 30.0) / float(engine.cycles_per_day)),
            )
            for engine in scenario.engines
        }
        return _schedule_greedily(scenario, policy, triggers)
    raise OptimizationError(f"Unknown baseline policy: {policy}")


def _status_name(status: cp_model.CpSolverStatus) -> str:
    return {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }[status]


def _no_solution_result(
    *,
    status: str,
    stage1_status: str,
    solve_time_seconds: float,
) -> dict[str, Any]:
    return {
        "policy": "cp_sat",
        "solver_status": status,
        "stage1_status": stage1_status,
        "lexicographic_optimality": "unavailable_no_solution",
        "solve_time_seconds": solve_time_seconds,
        "jobs": [],
    }


def solve_cp_sat(
    scenario: Scenario,
    *,
    time_limit_seconds: float = 30.0,
    require_all_due: bool = False,
) -> dict[str, Any]:
    """Solve the two-stage integer maintenance schedule."""
    model = cp_model.CpModel()
    starts: dict[tuple[int, int, str, str], cp_model.IntVar] = {}
    deferred: dict[int, cp_model.IntVar] = {}
    engine_by_id = {engine.engine_id: engine for engine in scenario.engines}
    for engine in scenario.engines:
        deferred[engine.engine_id] = model.new_bool_var(f"deferred_{engine.engine_id}")
        choices = []
        for day in range(scenario.horizon_days - engine.duration_days + 1):
            for team_id, _ in scenario.teams:
                for bay_id in scenario.bays:
                    variable = model.new_bool_var(
                        f"start_{engine.engine_id}_{day}_{team_id}_{bay_id}"
                    )
                    starts[(engine.engine_id, day, team_id, bay_id)] = variable
                    choices.append(variable)
        model.add(sum(choices) + deferred[engine.engine_id] == 1)
        if require_all_due and engine.due_in_horizon:
            model.add(deferred[engine.engine_id] == 0)

    for day in range(scenario.horizon_days):
        for team_id, capacity in scenario.teams:
            active = [
                engine_by_id[engine_id].technicians * variable
                for (engine_id, start, team, _), variable in starts.items()
                if team == team_id
                and start <= day < start + engine_by_id[engine_id].duration_days
            ]
            model.add(sum(active) <= capacity)
        for bay_id in scenario.bays:
            active = [
                variable
                for (engine_id, start, _, bay), variable in starts.items()
                if bay == bay_id
                and start <= day < start + engine_by_id[engine_id].duration_days
            ]
            model.add(sum(active) <= 1)
        total_cycles = sum(engine.cycles_per_day for engine in scenario.engines)
        minimum_cycles = math.ceil(total_cycles * scenario.operating_demand_fraction)
        outage = [
            engine_by_id[engine_id].cycles_per_day * variable
            for (engine_id, start, _, _), variable in starts.items()
            if start <= day < start + engine_by_id[engine_id].duration_days
        ]
        model.add(sum(outage) <= total_cycles - minimum_cycles)
        for kit_type in scenario.part_types:
            consumed = [
                engine_by_id[engine_id].kit_quantity * variable
                for (engine_id, start, _, _), variable in starts.items()
                if start <= day and engine_by_id[engine_id].kit_type == kit_type
            ]
            available = scenario.initial_parts_per_type + sum(
                scenario.replenishment_units_per_type
                for replenish_day in scenario.replenishment_days
                if replenish_day <= day
            )
            model.add(sum(consumed) <= available)

    due_deferrals = sum(
        deferred[engine.engine_id]
        for engine in scenario.engines
        if engine.due_in_horizon
    )
    late_terms = []
    for (engine_id, day, _, _), variable in starts.items():
        engine = engine_by_id[engine_id]
        if engine.due_in_horizon and day > engine.safe_due_day:
            late_terms.append((day - engine.safe_due_day) * variable)
    late_terms.extend(
        (scenario.horizon_days - engine.safe_due_day) * deferred[engine.engine_id]
        for engine in scenario.engines
        if engine.due_in_horizon
    )
    late_days = sum(late_terms)
    safety_weight = len(scenario.engines) * scenario.horizon_days + 1
    safety_score = safety_weight * due_deferrals + late_days
    model.minimize(safety_score)
    for variable in starts.values():
        model.add_hint(variable, 0)
    for variable in deferred.values():
        model.add_hint(variable, 1)

    stage1 = cp_model.CpSolver()
    stage1.parameters.max_time_in_seconds = time_limit_seconds
    stage1.parameters.random_seed = scenario.seed
    stage1.parameters.num_search_workers = 1
    started = time.perf_counter()
    stage1_code = stage1.solve(model)
    stage1_elapsed = time.perf_counter() - started
    stage1_status = _status_name(stage1_code)
    if stage1_status not in SOLVER_STATUSES_WITH_SOLUTION:
        return _no_solution_result(
            status=stage1_status,
            stage1_status=stage1_status,
            solve_time_seconds=stage1_elapsed,
        )
    best_safety = round(stage1.objective_value)
    model.add(safety_score == best_safety)
    model.clear_hints()
    for variable in starts.values():
        model.add_hint(variable, stage1.value(variable))
    for variable in deferred.values():
        model.add_hint(variable, stage1.value(variable))
    operational_terms = []
    for (engine_id, day, _, _), variable in starts.items():
        engine = engine_by_id[engine_id]
        early_cycles = max(0, engine.point_rul_cycles - day * engine.cycles_per_day)
        operational_terms.append(
            (
                scenario.planned_maintenance_cost
                + scenario.unused_predicted_cycle_cost * early_cycles
            )
            * variable
        )
    operational_terms.extend(
        scenario.low_risk_deferral_cost * deferred[engine.engine_id]
        for engine in scenario.engines
        if not engine.due_in_horizon
    )
    model.minimize(sum(operational_terms))

    stage2 = cp_model.CpSolver()
    stage2.parameters.max_time_in_seconds = time_limit_seconds
    stage2.parameters.random_seed = scenario.seed
    stage2.parameters.num_search_workers = 1
    stage2_started = time.perf_counter()
    stage2_code = stage2.solve(model)
    elapsed = stage1_elapsed + time.perf_counter() - stage2_started
    stage2_status = _status_name(stage2_code)
    if stage2_status not in SOLVER_STATUSES_WITH_SOLUTION:
        return _no_solution_result(
            status=stage2_status,
            stage1_status=stage1_status,
            solve_time_seconds=elapsed,
        )
    jobs = []
    for engine in scenario.engines:
        if stage2.value(deferred[engine.engine_id]):
            jobs.append(_empty_job(engine, "solver_deferral"))
            continue
        selected = [
            (day, team, bay)
            for (engine_id, day, team, bay), variable in starts.items()
            if engine_id == engine.engine_id and stage2.value(variable)
        ]
        if len(selected) != 1:
            raise OptimizationError("Solver returned an invalid assignment count")
        day, team, bay = selected[0]
        jobs.append(
            {
                "engine_id": engine.engine_id,
                "status": "scheduled",
                "start_day": day,
                "end_day": day + engine.duration_days,
                "team_id": team,
                "bay_id": bay,
                "reason": "cp_sat_assignment",
            }
        )
    proven = stage1_status == "OPTIMAL" and stage2_status == "OPTIMAL"
    return {
        "policy": "cp_sat",
        "solver_status": "OPTIMAL" if proven else "FEASIBLE",
        "stage1_status": stage1_status,
        "stage2_status": stage2_status,
        "lexicographic_optimality": "proven" if proven else "unproven",
        "stage1_safety_score": best_safety,
        "stage2_operational_cost": round(stage2.objective_value),
        "solve_time_seconds": elapsed,
        "jobs": sorted(jobs, key=lambda job: job["engine_id"]),
    }


def evaluate_schedule(
    scenario: Scenario,
    schedule: dict[str, Any],
) -> dict[str, Any]:
    """Compute shared decision metrics and validate every resource invariant."""
    if schedule["solver_status"] in {
        "INFEASIBLE",
        "MODEL_INVALID",
        "UNKNOWN",
    }:
        if schedule["jobs"]:
            raise OptimizationError("No-solution status must not contain a schedule")
        return {
            "policy": schedule["policy"],
            "solver_status": schedule["solver_status"],
            "schedule_available": False,
            "solve_time_seconds": schedule["solve_time_seconds"],
        }
    jobs = schedule["jobs"]
    if {job["engine_id"] for job in jobs} != {
        engine.engine_id for engine in scenario.engines
    }:
        raise OptimizationError("Schedule must contain exactly one row per engine")
    engine_by_id = {engine.engine_id: engine for engine in scenario.engines}
    for job in jobs:
        if job["status"] == "scheduled":
            engine = engine_by_id[job["engine_id"]]
            other_jobs = [item for item in jobs if item is not job]
            if not _assignment_feasible(
                scenario,
                other_jobs,
                engine,
                int(job["start_day"]),
                str(job["team_id"]),
                str(job["bay_id"]),
            ):
                raise OptimizationError(
                    f"Schedule violates a resource invariant: engine {engine.engine_id}"
                )
    scheduled = [job for job in jobs if job["status"] == "scheduled"]
    deferred_jobs = [job for job in jobs if job["status"] == "deferred"]
    late_days = sum(
        max(
            0,
            (
                int(job["start_day"])
                if job["status"] == "scheduled"
                else scenario.horizon_days
            )
            - engine_by_id[job["engine_id"]].safe_due_day,
        )
        for job in jobs
        if engine_by_id[job["engine_id"]].due_in_horizon
    )
    due_deferrals = sum(
        engine_by_id[job["engine_id"]].due_in_horizon for job in deferred_jobs
    )
    early_cycles = sum(
        max(
            0,
            engine_by_id[job["engine_id"]].point_rul_cycles
            - int(job["start_day"]) * engine_by_id[job["engine_id"]].cycles_per_day,
        )
        for job in scheduled
    )
    low_risk_deferrals = sum(
        not engine_by_id[job["engine_id"]].due_in_horizon for job in deferred_jobs
    )
    planned_cost = len(scheduled) * scenario.planned_maintenance_cost
    decision_cost = (
        planned_cost
        + early_cycles * scenario.unused_predicted_cycle_cost
        + low_risk_deferrals * scenario.low_risk_deferral_cost
    )
    technician_days = sum(
        engine_by_id[job["engine_id"]].technicians
        * engine_by_id[job["engine_id"]].duration_days
        for job in scheduled
    )
    available_technician_days = (
        sum(capacity for _, capacity in scenario.teams) * scenario.horizon_days
    )
    bay_days = sum(engine_by_id[job["engine_id"]].duration_days for job in scheduled)
    return {
        "policy": schedule["policy"],
        "solver_status": schedule["solver_status"],
        "schedule_available": True,
        "scheduled_maintenance": len(scheduled),
        "deferred_maintenance": len(deferred_jobs),
        "due_deferrals": int(due_deferrals),
        "late_days": int(late_days),
        "early_cycles_lost": int(early_cycles),
        "planned_cost_units": int(planned_cost),
        "decision_cost_units": int(decision_cost),
        "team_utilization": technician_days / available_technician_days,
        "bay_utilization": bay_days / (len(scenario.bays) * scenario.horizon_days),
        "operating_capacity_shortfall_cycles": 0,
        "solve_time_seconds": schedule["solve_time_seconds"],
        "lexicographic_optimality": schedule["lexicographic_optimality"],
    }


def evaluate_retrospective(
    scenario: Scenario,
    schedule: dict[str, Any],
    true_rul_by_engine: dict[int, int],
) -> dict[str, int]:
    """Join truth only after a schedule is frozen and count simulated failures."""
    if not schedule["jobs"]:
        return {"unplanned_failures": 0, "emergency_cost_units": 0}
    engine_by_id = {engine.engine_id: engine for engine in scenario.engines}
    expected = set(engine_by_id)
    if set(true_rul_by_engine) != expected:
        raise OptimizationError("Retrospective truth must match the scenario fleet")
    failures = 0
    for job in schedule["jobs"]:
        engine = engine_by_id[job["engine_id"]]
        failure_day = math.floor(
            int(true_rul_by_engine[engine.engine_id]) / engine.cycles_per_day
        )
        if job["status"] != "scheduled" or int(job["start_day"]) > failure_day:
            failures += 1
    return {
        "unplanned_failures": failures,
        "emergency_cost_units": failures * scenario.emergency_maintenance_cost,
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    output = StringIO()
    pd.DataFrame(rows).to_csv(output, index=False, lineterminator="\n")
    return output.getvalue().encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _load_verified_official_predictions(
    project_root: Path, run_id: str
) -> tuple[pd.DataFrame, dict[int, int], dict[str, str]]:
    run_dir = project_root / "runs" / run_id
    official = run_dir / "official_test"
    manifest_path = official / "evaluation_manifest.json"
    lock_path = run_dir / "model_lock.json"
    predictions_path = official / "predictions.parquet"
    for path in (manifest_path, lock_path, predictions_path):
        if not path.is_file():
            raise OptimizationError(f"Required verified artifact is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(lock_path) != manifest["model_lock_sha256"]:
        raise OptimizationError("Official evaluation references a changed model lock")
    if sha256_file(predictions_path) != manifest["predictions_sha256"]:
        raise OptimizationError("Official prediction artifact hash mismatch")
    frame = pd.read_parquet(predictions_path)
    required = set(DECISION_COLUMNS) | {"rul_true"}
    if not required.issubset(frame.columns):
        raise OptimizationError("Official predictions do not satisfy the M3 boundary")
    decision_frame = frame.loc[:, DECISION_COLUMNS].copy()
    truth = {
        int(row.unit_id): int(row.rul_true)
        for row in frame.loc[:, ["unit_id", "rul_true"]].itertuples(index=False)
    }
    return (
        decision_frame,
        truth,
        {
            "model_lock_sha256": manifest["model_lock_sha256"],
            "predictions_sha256": manifest["predictions_sha256"],
            "evaluation_manifest_sha256": sha256_file(manifest_path),
        },
    )


def _data_dictionary() -> dict[str, Any]:
    return {
        "scope": "All operational, resource, and cost fields are synthetic.",
        "truth_boundary": (
            "rul_true is excluded from scenario, policies, and solver; it is joined "
            "only after schedules are frozen by the retrospective evaluator."
        ),
        "engine_fields": {
            "engine_id": "NASA simulated engine identifier",
            "observed_cycle": "last observed simulated cycle",
            "prediction": "locked model point RUL prediction",
            "interval_low": "nominal empirical interval lower bound",
            "interval_high": "nominal empirical interval upper bound",
            "risk_band": "band derived from interval_low",
            "cycles_per_day": "synthetic integer 1-4",
            "duration_days": "synthetic integer 2-5",
            "technicians": "synthetic integer 2-6",
            "kit_type": "synthetic kit_A, kit_B, or kit_C",
            "kit_quantity": "synthetic integer 1-2",
            "safe_due_day": "floor(floor(interval_low)/cycles_per_day)-1, clamped at 0",
        },
        "resource_fields": {
            "teams": "synthetic teams with six technicians each",
            "bays": "synthetic maintenance bays",
            "parts": "synthetic starting inventory and replenishment",
            "operating_demand_fraction": "synthetic minimum daily capacity fraction",
        },
        "cost_fields": "All values are synthetic cost_units, not real currency.",
    }


def optimize_run(project_root: Path, *, run_id: str) -> OptimizationResult:
    """Create the complete immutable M3 artifact set for one verified M2 run."""
    root = project_root.resolve()
    run_dir = root / "runs" / run_id
    output_dir = run_dir / "optimization"
    if output_dir.exists():
        raise OptimizationError(f"Refusing to overwrite existing output: {output_dir}")
    decision_frame, truth, source_hashes = _load_verified_official_predictions(
        root, run_id
    )
    config = _load_scenario_config(root)
    scenario = generate_scenario(decision_frame, config)
    selected_truth = {
        engine.engine_id: truth[engine.engine_id] for engine in scenario.engines
    }
    schedules = {
        policy: build_baseline_schedule(scenario, policy)
        for policy in ("reactive", "fixed_90", "predicted_rul_30")
    }
    schedules["cp_sat"] = solve_cp_sat(scenario)

    policy_rows = []
    for schedule in schedules.values():
        decision_metrics = evaluate_schedule(scenario, schedule)
        retrospective = evaluate_retrospective(scenario, schedule, selected_truth)
        row = {**decision_metrics, **retrospective}
        row["total_synthetic_cost_units"] = (
            int(row.get("decision_cost_units", 0))
            + retrospective["emergency_cost_units"]
        )
        policy_rows.append(row)

    capacity_rows = []
    capacity_schedules = {}
    for name in ("constrained", "base", "expanded"):
        variant = scenario_variant(scenario, name)
        schedule = schedules["cp_sat"] if name == "base" else solve_cp_sat(variant)
        capacity_schedules[name] = schedule
        decision_metrics = evaluate_schedule(variant, schedule)
        retrospective = evaluate_retrospective(variant, schedule, selected_truth)
        row = {
            "capacity_scenario": name,
            "bays": len(variant.bays),
            "operating_demand_fraction": variant.operating_demand_fraction,
            **decision_metrics,
            **retrospective,
        }
        row["total_synthetic_cost_units"] = (
            int(row.get("decision_cost_units", 0))
            + retrospective["emergency_cost_units"]
        )
        capacity_rows.append(row)

    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}-m3-", dir=str(run_dir)))
    try:
        artefacts: dict[str, bytes] = {
            "scenario.json": _json_bytes(scenario.to_dict()),
            "scenario_data_dictionary.json": _json_bytes(_data_dictionary()),
            "policy_comparison.json": _json_bytes(policy_rows),
            "policy_comparison.csv": _csv_bytes(policy_rows),
            "capacity_comparison.json": _json_bytes(capacity_rows),
            "capacity_comparison.csv": _csv_bytes(capacity_rows),
        }
        for policy, schedule in schedules.items():
            artefacts[f"schedules/{policy}.json"] = _json_bytes(schedule)
        for name, schedule in capacity_schedules.items():
            artefacts[f"capacity_schedules/{name}.json"] = _json_bytes(schedule)
        hashes = {}
        for relative_path, payload in sorted(artefacts.items()):
            _write_bytes(temporary / relative_path, payload)
            hashes[relative_path] = hashlib.sha256(payload).hexdigest()
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "optimization_complete",
            "seed": scenario.seed,
            "generator_version": scenario.generator_version,
            "scenario_config_sha256": sha256_file(root / "configs" / "scenario.yaml"),
            "source_artifacts": source_hashes,
            "truth_boundary": (
                "True RUL excluded from every planning artefact and used only for "
                "post-freeze retrospective failure metrics."
            ),
            "artifacts": hashes,
        }
        _write_bytes(temporary / "manifest.json", _json_bytes(manifest))
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return OptimizationResult(
        run_id=run_id,
        output_dir=output_dir,
        policy_comparison=tuple(policy_rows),
        capacity_comparison=tuple(capacity_rows),
    )
