from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from aeromaintain.app import (
    ArtifactValidationError,
    load_verified_run,
    run_capacity_what_if,
    validate_what_if,
)
from aeromaintain.data import PrepareResult
from aeromaintain.data.pipeline import SENSOR_COLUMNS, sha256_file
from aeromaintain.delivery import DeliveryError, run_pipeline
from aeromaintain.models.rul import EvaluationResult, TrainResult
from aeromaintain.optimization import OptimizationResult


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_verified_run(root: Path, run_id: str = "fixture") -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "project.yaml").write_text("seed: 42\n", encoding="utf-8")
    (config_dir / "scenario.yaml").write_text("seed: 42\n", encoding="utf-8")
    processed = root / "data" / "processed" / "fd001"
    (processed / "evaluation").mkdir(parents=True)
    _write_json(processed / "manifest.json", {"dataset": "FD001"})
    _write_json(processed / "split_manifest.json", {"seed": 42})
    pd.DataFrame({"unit_id": [1], "cycle": [1]}).to_parquet(
        processed / "train.parquet", index=False
    )
    test_row = {"unit_id": 1, "cycle": 1}
    test_row.update({name: float(index) for index, name in enumerate(SENSOR_COLUMNS)})
    pd.DataFrame([test_row]).to_parquet(processed / "test.parquet", index=False)
    pd.DataFrame({"unit_id": [1], "rul_true": [10]}).to_parquet(
        processed / "evaluation" / "test_rul.parquet", index=False
    )

    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    explanation = {
        "method": "standardized Ridge coefficients",
        "scope": "model behavior; not physical causality",
        "global_importance": [{"feature": "engine_age", "coefficient": -1.0}],
        "local_explanation": {"unit_id": 1, "cycle": 1, "features": []},
    }
    _write_json(run_dir / "explanation.json", explanation)
    lock = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": "FD001",
        "seed": 42,
        "config": {
            "path": "configs/project.yaml",
            "sha256": sha256_file(config_dir / "project.yaml"),
        },
        "data": {
            "manifest_sha256": sha256_file(processed / "manifest.json"),
            "split_manifest_sha256": sha256_file(processed / "split_manifest.json"),
            "train_sha256": sha256_file(processed / "train.parquet"),
            "test_sha256": sha256_file(processed / "test.parquet"),
            "official_test_rul_sha256": sha256_file(
                processed / "evaluation" / "test_rul.parquet"
            ),
        },
        "champion": {"kind": "ridge", "path": "model.joblib"},
        "calibration": {
            "label": "nominal empirical interval; not a safety guarantee",
            "coverage": 0.9,
            "q": 5.0,
        },
        "artifacts": {
            "explanation.json": sha256_file(run_dir / "explanation.json"),
        },
    }
    _write_json(run_dir / "model_lock.json", lock)
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "status": "model_locked",
            "model_lock_sha256": sha256_file(run_dir / "model_lock.json"),
        },
    )

    official = run_dir / "official_test"
    official.mkdir()
    predictions = pd.DataFrame(
        {
            "unit_id": [1],
            "cycle": [1],
            "rul_true": [10],
            "prediction": [9.5],
            "interval_low": [4.5],
            "interval_high": [14.5],
            "risk_band": ["critical"],
        }
    )
    predictions.to_parquet(official / "predictions.parquet", index=False)
    metrics = {
        "engines": 1,
        "mae": 0.5,
        "rmse": 0.5,
        "nasa_score_motor_normalized": 0.1,
        "critical_rul": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "nominal_empirical_interval": {
            "nominal_coverage": 0.9,
            "observed_official_test_coverage": 1.0,
            "mean_width": 10.0,
            "label": "empirical prediction interval; not a safety guarantee",
        },
    }
    _write_json(official / "metrics.json", metrics)
    _write_json(official / "error_analysis.json", {"largest_absolute_errors": []})
    evaluation_manifest = {
        "run_id": run_id,
        "model_lock_sha256": sha256_file(run_dir / "model_lock.json"),
        "predictions_sha256": sha256_file(official / "predictions.parquet"),
        "metrics_sha256": sha256_file(official / "metrics.json"),
        "error_analysis_sha256": sha256_file(official / "error_analysis.json"),
    }
    _write_json(official / "evaluation_manifest.json", evaluation_manifest)

    optimization = run_dir / "optimization"
    (optimization / "schedules").mkdir(parents=True)
    scenario = {
        "schema_version": 1,
        "generator_version": "fixture-v1",
        "seed": 42,
        "horizon_days": 5,
        "operating_demand_fraction": 0.8,
        "engines": [
            {
                "engine_id": 1,
                "observed_cycle": 1,
                "prediction": 9.5,
                "interval_low": 4.5,
                "interval_high": 14.5,
                "risk_band": "critical",
                "cycles_per_day": 2,
                "duration_days": 2,
                "technicians": 2,
                "kit_type": "kit_A",
                "kit_quantity": 1,
                "point_rul_cycles": 9,
                "lower_rul_cycles": 4,
                "safe_due_day": 1,
                "due_in_horizon": True,
            }
        ],
        "teams": [{"team_id": "team_A", "technicians": 6}],
        "bays": ["bay_1"],
        "part_types": ["kit_A"],
        "initial_parts_per_type": 4,
        "replenishment_days": [3],
        "replenishment_units_per_type": 1,
        "planned_maintenance_cost": 100,
        "emergency_maintenance_cost": 500,
        "unused_predicted_cycle_cost": 1,
        "low_risk_deferral_cost": 150,
        "metadata": {"synthetic": True, "truth_fields_present": False},
    }
    _write_json(optimization / "scenario.json", scenario)
    schedule = {
        "policy": "cp_sat",
        "solver_status": "OPTIMAL",
        "lexicographic_optimality": "proven",
        "jobs": [
            {
                "engine_id": 1,
                "status": "scheduled",
                "start_day": 0,
                "end_day": 2,
                "team_id": "team_A",
                "bay_id": "bay_1",
                "reason": "fixture",
            }
        ],
    }
    _write_json(optimization / "schedules" / "cp_sat.json", schedule)
    policy = pd.DataFrame(
        [
            {
                "policy": "cp_sat",
                "solver_status": "OPTIMAL",
                "lexicographic_optimality": "proven",
                "scheduled_maintenance": 1,
                "due_deferrals": 0,
                "total_synthetic_cost_units": 100,
            }
        ]
    )
    capacity = pd.DataFrame(
        [
            {
                "capacity_scenario": "base",
                "solver_status": "OPTIMAL",
                "scheduled_maintenance": 1,
            }
        ]
    )
    policy.to_json(optimization / "policy_comparison.json", orient="records")
    policy.to_csv(optimization / "policy_comparison.csv", index=False)
    capacity.to_json(optimization / "capacity_comparison.json", orient="records")
    capacity.to_csv(optimization / "capacity_comparison.csv", index=False)
    artifact_names = (
        "scenario.json",
        "schedules/cp_sat.json",
        "policy_comparison.json",
        "policy_comparison.csv",
        "capacity_comparison.json",
        "capacity_comparison.csv",
    )
    _write_json(
        optimization / "manifest.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "status": "optimization_complete",
            "scenario_config_sha256": sha256_file(config_dir / "scenario.yaml"),
            "source_artifacts": {
                "model_lock_sha256": sha256_file(run_dir / "model_lock.json"),
                "predictions_sha256": sha256_file(official / "predictions.parquet"),
                "evaluation_manifest_sha256": sha256_file(
                    official / "evaluation_manifest.json"
                ),
            },
            "artifacts": {
                name: sha256_file(optimization / name) for name in artifact_names
            },
        },
    )
    return run_dir


def test_verified_app_loader_excludes_truth_and_exposes_downloads(
    tmp_path: Path,
) -> None:
    _build_verified_run(tmp_path)

    artifacts = load_verified_run(tmp_path, "fixture")

    assert artifacts.run_id == "fixture"
    assert "rul_true" not in artifacts.risk_ranking.columns
    assert artifacts.risk_ranking.loc[0, "prediction"] == 9.5
    assert set(artifacts.downloads) == {
        "policy_comparison.csv",
        "capacity_comparison.csv",
        "cp_sat_schedule.csv",
        "risk_ranking.csv",
    }
    assert list(artifacts.sensor_history["unit_id"]) == [1]


def test_verified_app_loader_rejects_changed_or_truth_leaking_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = _build_verified_run(tmp_path)
    metrics = run_dir / "official_test" / "metrics.json"
    metrics.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="SHA-256 mismatch"):
        load_verified_run(tmp_path, "fixture")

    second_root = tmp_path / "second"
    second_run = _build_verified_run(second_root)
    scenario = second_run / "optimization" / "scenario.json"
    payload = json.loads(scenario.read_text(encoding="utf-8"))
    payload["engines"][0]["rul_true"] = 10
    _write_json(scenario, payload)
    manifest_path = second_run / "optimization" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["scenario.json"] = sha256_file(scenario)
    _write_json(manifest_path, manifest)

    with pytest.raises(ArtifactValidationError, match="true-RUL"):
        load_verified_run(second_root, "fixture")

    third_root = tmp_path / "third"
    third_run = _build_verified_run(third_root)
    lock_path = third_run / "model_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["config"]["path"] = "../outside.yaml"
    _write_json(lock_path, lock)
    run_manifest_path = third_run / "manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["model_lock_sha256"] = sha256_file(lock_path)
    _write_json(run_manifest_path, run_manifest)

    with pytest.raises(ArtifactValidationError, match="external path"):
        load_verified_run(third_root, "fixture")


def test_what_if_validation_precedes_solver(monkeypatch, tmp_path: Path) -> None:
    _build_verified_run(tmp_path)
    artifacts = load_verified_run(tmp_path, "fixture")
    called = False

    def solver(*args, **kwargs):
        nonlocal called
        called = True
        return {
            "policy": "cp_sat",
            "solver_status": "UNKNOWN",
            "solve_time_seconds": 0.01,
            "jobs": [],
        }

    monkeypatch.setattr("aeromaintain.app.artifacts.solve_cp_sat", solver)
    with pytest.raises(ArtifactValidationError, match="Bays"):
        run_capacity_what_if(
            artifacts,
            bays=4,
            operating_demand_fraction=0.8,
        )
    assert not called

    schedule, metrics = run_capacity_what_if(
        artifacts,
        bays=2,
        operating_demand_fraction=0.75,
    )
    assert called
    assert schedule["solver_status"] == "UNKNOWN"
    assert not metrics["schedule_available"]
    with pytest.raises(ArtifactValidationError, match="between 70% and 90%"):
        validate_what_if(2, 0.95)


def test_streamlit_small_fixture_renders_all_pages(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _build_verified_run(tmp_path)
    monkeypatch.setenv("AEROMAINTAIN_RUN_ID", "fixture")
    monkeypatch.setenv("AEROMAINTAIN_PROJECT_ROOT", str(tmp_path))
    script = (
        Path(__file__).resolve().parents[1] / "src" / "aeromaintain" / "app" / "main.py"
    )

    tested = AppTest.from_file(str(script), default_timeout=30).run()
    assert not tested.exception
    assert tested.header[0].value == "Overview"

    for page in ("Engine risk", "Maintenance plan", "Policy analysis"):
        tested.sidebar.radio[0].set_value(page)
        tested.run()
        assert not tested.exception
        assert tested.header[0].value == page

    tested.button[0].click()
    tested.run()
    assert not tested.exception


def test_small_fixture_pipeline_completes_all_stages_and_report(
    tmp_path: Path,
) -> None:
    calls = []

    def prepare(root, *, archive_path=None):
        calls.append(("prepare", archive_path))
        return PrepareResult(root, 1, 1, 1, 0, {})

    def train(root, *, run_id):
        calls.append(("train", run_id))
        run_dir = _build_verified_run(root, run_id)
        return TrainResult(
            run_id,
            run_dir,
            "ridge",
            5.0,
            run_dir / "model_lock.json",
        )

    def evaluate(root, *, run_id):
        calls.append(("evaluate", run_id))
        return EvaluationResult(
            run_id,
            root / "runs" / run_id / "official_test",
            {"mae": 0.5},
        )

    def optimize(root, *, run_id):
        calls.append(("optimize", run_id))
        return OptimizationResult(
            run_id,
            root / "runs" / run_id / "optimization",
            (),
            (),
        )

    result = run_pipeline(
        tmp_path,
        run_id="fixture",
        prepare_step=prepare,
        train_step=train,
        evaluation_step=evaluate,
        optimization_step=optimize,
    )

    assert [name for name, _ in calls] == [
        "prepare",
        "train",
        "evaluate",
        "optimize",
    ]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "pipeline_complete"
    assert manifest["pipeline"]["status"] == "complete"
    assert result.report_path.is_file()
    assert (result.run_dir / "report.json").is_file()
    load_verified_run(tmp_path, "fixture")


def test_pipeline_refuses_overwrite_and_does_not_mark_partial_complete(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "runs" / "existing"
    existing.mkdir(parents=True)
    with pytest.raises(DeliveryError, match="already exists"):
        run_pipeline(tmp_path, run_id="existing")

    def prepare(root, *, archive_path=None):
        return PrepareResult(root, 1, 1, 1, 0, {})

    def train(root, *, run_id):
        run_dir = root / "runs" / run_id
        run_dir.mkdir(parents=True)
        _write_json(
            run_dir / "manifest.json",
            {"run_id": run_id, "status": "model_locked"},
        )
        return TrainResult(run_id, run_dir, "ridge", 1.0, run_dir / "lock.json")

    def fail_evaluation(root, *, run_id):
        raise RuntimeError("fixture evaluation failure")

    with pytest.raises(DeliveryError, match="stopped before completion"):
        run_pipeline(
            tmp_path,
            run_id="partial",
            prepare_step=prepare,
            train_step=train,
            evaluation_step=fail_evaluation,
        )
    partial = json.loads(
        (tmp_path / "runs" / "partial" / "manifest.json").read_text(encoding="utf-8")
    )
    assert partial["status"] == "model_locked"
    assert not (tmp_path / "runs" / "partial" / "report.json").exists()
