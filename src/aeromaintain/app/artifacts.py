"""Verified, truth-free artefact loading for the Streamlit application."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from aeromaintain.data.pipeline import SENSOR_COLUMNS, sha256_file
from aeromaintain.optimization import Engine, Scenario, evaluate_schedule, solve_cp_sat

SAFE_PREDICTION_COLUMNS = (
    "unit_id",
    "cycle",
    "prediction",
    "interval_low",
    "interval_high",
    "risk_band",
)
TRUTH_FIELD_NAMES = frozenset({"rul_true", "true_rul", "actual_rul"})


class ArtifactValidationError(RuntimeError):
    """Raised when a run cannot be trusted by the decision application."""


@dataclass(frozen=True)
class AppArtifacts:
    """Validated data exposed to the presentation layer."""

    run_id: str
    run_dir: Path
    run_manifest: dict[str, Any]
    model_lock: dict[str, Any]
    metrics: dict[str, Any]
    explanation: dict[str, Any]
    scenario: dict[str, Any]
    risk_ranking: pd.DataFrame
    sensor_history: pd.DataFrame
    schedule: pd.DataFrame
    policy_comparison: pd.DataFrame
    capacity_comparison: pd.DataFrame
    downloads: dict[str, bytes]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactValidationError(f"Required artefact is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            f"Artefact is unreadable or invalid: {path}"
        ) from exc


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ArtifactValidationError(f"Required artefact is missing: {path}")
    if sha256_file(path) != expected:
        raise ArtifactValidationError(f"{label} SHA-256 mismatch: {path}")


def _resolve_within(base: Path, relative_path: str, label: str) -> Path:
    path = (base / relative_path).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as exc:
        raise ArtifactValidationError(
            f"{label} references an external path: {relative_path}"
        ) from exc
    return path


def _verify_artifact_map(base: Path, artifacts: dict[str, str], label: str) -> None:
    for relative_path, expected in artifacts.items():
        path = _resolve_within(base, relative_path, label)
        _require_hash(path, expected, label)


def _contains_truth_field(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower() in TRUTH_FIELD_NAMES or _contains_truth_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_truth_field(item) for item in value)
    return False


def _safe_run_dir(project_root: Path, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id:
        raise ArtifactValidationError("run_id must be one safe path component")
    runs_root = (project_root / "runs").resolve()
    run_dir = (runs_root / run_id).resolve()
    try:
        run_dir.relative_to(runs_root)
    except ValueError as exc:
        raise ArtifactValidationError("run_id resolves outside runs/") from exc
    if not run_dir.is_dir():
        raise ArtifactValidationError(f"Run directory does not exist: {run_dir}")
    return run_dir


def _load_verified_run(project_root: Path, run_id: str) -> AppArtifacts:
    """Load one explicitly named run after verifying the complete hash chain."""
    root = project_root.resolve()
    run_dir = _safe_run_dir(root, run_id)
    run_manifest = _read_json(run_dir / "manifest.json")
    if run_manifest.get("run_id") != run_id:
        raise ArtifactValidationError("Run manifest identity mismatch")
    if run_manifest.get("status") not in {"model_locked", "pipeline_complete"}:
        raise ArtifactValidationError("Run is not model-locked or pipeline-complete")
    if run_manifest.get("status") == "pipeline_complete":
        pipeline = run_manifest.get("pipeline", {})
        if pipeline.get("status") != "complete":
            raise ArtifactValidationError("Pipeline completion record is invalid")
        _verify_artifact_map(run_dir, pipeline.get("artifacts", {}), "Pipeline")

    lock_path = run_dir / "model_lock.json"
    _require_hash(lock_path, run_manifest["model_lock_sha256"], "Model lock")
    lock = _read_json(lock_path)
    if lock.get("run_id") != run_id or lock.get("schema_version") != 1:
        raise ArtifactValidationError("Model lock identity or schema mismatch")
    _verify_artifact_map(run_dir, lock.get("artifacts", {}), "Model artefact")
    _require_hash(
        _resolve_within(root, lock["config"]["path"], "Model config"),
        lock["config"]["sha256"],
        "Model config",
    )
    processed_dir = root / "data" / "processed" / "fd001"
    for name, key in (
        ("manifest.json", "manifest_sha256"),
        ("split_manifest.json", "split_manifest_sha256"),
        ("train.parquet", "train_sha256"),
        ("test.parquet", "test_sha256"),
        ("evaluation/test_rul.parquet", "official_test_rul_sha256"),
    ):
        _require_hash(processed_dir / name, lock["data"][key], "Locked data")

    official_dir = run_dir / "official_test"
    evaluation_manifest_path = official_dir / "evaluation_manifest.json"
    evaluation_manifest = _read_json(evaluation_manifest_path)
    if evaluation_manifest.get("run_id") != run_id:
        raise ArtifactValidationError("Evaluation manifest identity mismatch")
    _require_hash(
        lock_path,
        evaluation_manifest["model_lock_sha256"],
        "Evaluation model lock",
    )
    for name, key in (
        ("predictions.parquet", "predictions_sha256"),
        ("metrics.json", "metrics_sha256"),
        ("error_analysis.json", "error_analysis_sha256"),
    ):
        _require_hash(official_dir / name, evaluation_manifest[key], "Evaluation")

    optimization_dir = run_dir / "optimization"
    optimization_manifest = _read_json(optimization_dir / "manifest.json")
    if (
        optimization_manifest.get("run_id") != run_id
        or optimization_manifest.get("status") != "optimization_complete"
    ):
        raise ArtifactValidationError("Optimization manifest is not complete")
    _require_hash(
        root / "configs" / "scenario.yaml",
        optimization_manifest["scenario_config_sha256"],
        "Scenario config",
    )
    source = optimization_manifest.get("source_artifacts", {})
    _require_hash(lock_path, source["model_lock_sha256"], "Optimization model lock")
    _require_hash(
        official_dir / "predictions.parquet",
        source["predictions_sha256"],
        "Optimization predictions",
    )
    _require_hash(
        evaluation_manifest_path,
        source["evaluation_manifest_sha256"],
        "Optimization evaluation manifest",
    )
    _verify_artifact_map(
        optimization_dir,
        optimization_manifest.get("artifacts", {}),
        "Optimization artefact",
    )

    predictions = pd.read_parquet(official_dir / "predictions.parquet")
    missing_prediction_columns = set(SAFE_PREDICTION_COLUMNS) - set(predictions.columns)
    if missing_prediction_columns:
        raise ArtifactValidationError(
            "Predictions are missing decision columns: "
            + ", ".join(sorted(missing_prediction_columns))
        )
    risk_ranking = (
        predictions.loc[:, SAFE_PREDICTION_COLUMNS]
        .sort_values(["interval_low", "unit_id"])
        .reset_index(drop=True)
    )

    processed_test = processed_dir / "test.parquet"
    _require_hash(processed_test, lock["data"]["test_sha256"], "Processed test data")
    test = pd.read_parquet(processed_test)
    sensor_history = test.loc[:, ["unit_id", "cycle", *SENSOR_COLUMNS]].copy()

    scenario = _read_json(optimization_dir / "scenario.json")
    if _contains_truth_field(scenario):
        raise ArtifactValidationError("Planning scenario contains a true-RUL field")
    schedule_payload = _read_json(optimization_dir / "schedules" / "cp_sat.json")
    schedule = pd.DataFrame(schedule_payload.get("jobs", []))
    policy = pd.read_csv(optimization_dir / "policy_comparison.csv")
    capacity = pd.read_csv(optimization_dir / "capacity_comparison.csv")
    downloads = {
        "policy_comparison.csv": (
            optimization_dir / "policy_comparison.csv"
        ).read_bytes(),
        "capacity_comparison.csv": (
            optimization_dir / "capacity_comparison.csv"
        ).read_bytes(),
        "cp_sat_schedule.csv": schedule.to_csv(index=False).encode("utf-8"),
        "risk_ranking.csv": risk_ranking.to_csv(index=False).encode("utf-8"),
    }
    return AppArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        run_manifest=run_manifest,
        model_lock=lock,
        metrics=_read_json(official_dir / "metrics.json"),
        explanation=_read_json(run_dir / "explanation.json"),
        scenario=scenario,
        risk_ranking=risk_ranking,
        sensor_history=sensor_history,
        schedule=schedule,
        policy_comparison=policy,
        capacity_comparison=capacity,
        downloads=downloads,
    )


def load_verified_run(project_root: Path, run_id: str) -> AppArtifacts:
    """Return a validated run with structural failures translated for the UI."""
    try:
        return _load_verified_run(project_root, run_id)
    except ArtifactValidationError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"Verified run artefact structure is invalid: {exc}"
        ) from exc


def validate_what_if(bays: int, operating_demand_fraction: float) -> None:
    """Reject unsupported capacity inputs before constructing a solver model."""
    if bays not in {1, 2, 3}:
        raise ArtifactValidationError("Bays must be an integer from 1 to 3")
    if not 0.70 <= operating_demand_fraction <= 0.90:
        raise ArtifactValidationError("Operating demand must be between 70% and 90%")


def _scenario_from_payload(payload: dict[str, Any]) -> Scenario:
    if _contains_truth_field(payload):
        raise ArtifactValidationError("What-if scenario contains a true-RUL field")
    engines = tuple(
        Engine(**{field: row[field] for field in Engine.__dataclass_fields__})
        for row in payload["engines"]
    )
    teams = tuple((row["team_id"], int(row["technicians"])) for row in payload["teams"])
    return Scenario(
        schema_version=int(payload["schema_version"]),
        generator_version=str(payload["generator_version"]),
        seed=int(payload["seed"]),
        horizon_days=int(payload["horizon_days"]),
        operating_demand_fraction=float(payload["operating_demand_fraction"]),
        engines=engines,
        teams=teams,
        bays=tuple(payload["bays"]),
        part_types=tuple(payload["part_types"]),
        initial_parts_per_type=int(payload["initial_parts_per_type"]),
        replenishment_days=tuple(payload["replenishment_days"]),
        replenishment_units_per_type=int(payload["replenishment_units_per_type"]),
        planned_maintenance_cost=int(payload["planned_maintenance_cost"]),
        emergency_maintenance_cost=int(payload["emergency_maintenance_cost"]),
        unused_predicted_cycle_cost=int(payload["unused_predicted_cycle_cost"]),
        low_risk_deferral_cost=int(payload["low_risk_deferral_cost"]),
    )


def run_capacity_what_if(
    artifacts: AppArtifacts,
    *,
    bays: int,
    operating_demand_fraction: float,
    time_limit_seconds: float = 5.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Solve one validated truth-free capacity variation."""
    validate_what_if(bays, operating_demand_fraction)
    base = _scenario_from_payload(artifacts.scenario)
    varied = replace(
        base,
        bays=tuple(f"what_if_bay_{index}" for index in range(1, bays + 1)),
        operating_demand_fraction=operating_demand_fraction,
    )
    schedule = solve_cp_sat(varied, time_limit_seconds=time_limit_seconds)
    return schedule, evaluate_schedule(varied, schedule)
