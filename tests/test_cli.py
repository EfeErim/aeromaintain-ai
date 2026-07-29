from __future__ import annotations

import shutil
from pathlib import Path

from typer.testing import CliRunner

from aeromaintain.cli import app, collect_doctor_checks
from aeromaintain.config import REQUIRED_LOCAL_DATA_DIRS
from aeromaintain.data import DataPipelineError, PrepareResult
from aeromaintain.models import ModelingError
from aeromaintain.models.rul import EvaluationResult, TrainResult
from aeromaintain.optimization import OptimizationError, OptimizationResult

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_doctor_reports_a_healthy_repository(tmp_path: Path) -> None:
    shutil.copytree(REPOSITORY_ROOT / "configs", tmp_path / "configs")
    for directory in REQUIRED_LOCAL_DATA_DIRS:
        (tmp_path / directory).mkdir(parents=True)

    result = runner.invoke(
        app,
        ["doctor", "--project-root", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert "[PASS] python:" in result.output
    assert "[PASS] package:" in result.output
    assert "Summary: 5/5 checks passed" in result.output


def test_doctor_rejects_an_unsupported_python_version() -> None:
    checks = collect_doctor_checks(
        REPOSITORY_ROOT,
        version_info=(3, 12, 0),
    )

    assert checks[0].name == "python"
    assert not checks[0].passed
    assert "required >=3.11,<3.12" in checks[0].detail


def test_doctor_reports_missing_project_contract(tmp_path: Path) -> None:
    checks = collect_doctor_checks(tmp_path, version_info=(3, 11, 9))
    checks_by_name = {check.name: check for check in checks}

    assert not checks_by_name["configs"].passed
    assert "project.yaml" in checks_by_name["configs"].detail
    assert not checks_by_name["local directories"].passed
    assert "data/raw" in checks_by_name["local directories"].detail


def test_prepare_reports_success(monkeypatch, tmp_path: Path) -> None:
    result = PrepareResult(
        output_dir=tmp_path / "data" / "processed" / "fd001",
        train_rows=20_631,
        test_rows=13_096,
        development_engines=80,
        calibration_engines=20,
        artifact_hashes={"train.parquet": "abc"},
    )
    monkeypatch.setattr(
        "aeromaintain.cli.prepare_fd001", lambda *args, **kwargs: result
    )

    command = runner.invoke(app, ["prepare", "--project-root", str(tmp_path)])

    assert command.exit_code == 0, command.output
    assert "FD001 preparation complete" in command.output
    assert "train=20631, test=13096" in command.output


def test_prepare_reports_pipeline_failure(monkeypatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise DataPipelineError("fixture failure")

    monkeypatch.setattr("aeromaintain.cli.prepare_fd001", fail)
    command = runner.invoke(app, ["prepare", "--project-root", str(tmp_path)])

    assert command.exit_code == 1
    assert "Preparation failed: fixture failure" in command.output


def test_train_and_evaluate_report_success(monkeypatch, tmp_path: Path) -> None:
    train_result = TrainResult(
        run_id="fixture",
        run_dir=tmp_path / "runs" / "fixture",
        champion="ridge",
        calibration_q=12.5,
        model_lock=tmp_path / "runs" / "fixture" / "model_lock.json",
    )
    evaluation_result = EvaluationResult(
        run_id="fixture",
        output_dir=tmp_path / "runs" / "fixture" / "official_test",
        metrics={
            "mae": 10.0,
            "rmse": 12.0,
            "nasa_score_motor_normalized": 25.0,
        },
    )
    monkeypatch.setattr(
        "aeromaintain.cli.train_and_lock",
        lambda *args, **kwargs: train_result,
    )
    monkeypatch.setattr(
        "aeromaintain.cli.evaluate_locked",
        lambda *args, **kwargs: evaluation_result,
    )

    trained = runner.invoke(
        app,
        ["train", "--run-id", "fixture", "--project-root", str(tmp_path)],
    )
    evaluated = runner.invoke(
        app,
        ["evaluate", "--run-id", "fixture", "--project-root", str(tmp_path)],
    )

    assert trained.exit_code == 0, trained.output
    assert "Champion: ridge" in trained.output
    assert evaluated.exit_code == 0, evaluated.output
    assert "MAE=10.000000; RMSE=12.000000; NASA=25.000000" in evaluated.output


def test_train_and_evaluate_report_modeling_failure(
    monkeypatch, tmp_path: Path
) -> None:
    def fail(*args, **kwargs):
        raise ModelingError("fixture lock failure")

    monkeypatch.setattr("aeromaintain.cli.train_and_lock", fail)
    monkeypatch.setattr("aeromaintain.cli.evaluate_locked", fail)

    trained = runner.invoke(
        app,
        ["train", "--run-id", "fixture", "--project-root", str(tmp_path)],
    )
    evaluated = runner.invoke(
        app,
        ["evaluate", "--run-id", "fixture", "--project-root", str(tmp_path)],
    )

    assert trained.exit_code == 1
    assert "Training failed: fixture lock failure" in trained.output
    assert evaluated.exit_code == 1
    assert "Evaluation failed: fixture lock failure" in evaluated.output


def test_optimize_reports_success(monkeypatch, tmp_path: Path) -> None:
    result = OptimizationResult(
        run_id="fixture",
        output_dir=tmp_path / "runs" / "fixture" / "optimization",
        policy_comparison=(
            {
                "policy": "cp_sat",
                "solver_status": "FEASIBLE",
                "due_deferrals": 1,
                "late_days": 2,
            },
        ),
        capacity_comparison=(),
    )
    monkeypatch.setattr(
        "aeromaintain.cli.optimize_run",
        lambda *args, **kwargs: result,
    )

    command = runner.invoke(
        app,
        ["optimize", "--run-id", "fixture", "--project-root", str(tmp_path)],
    )

    assert command.exit_code == 0, command.output
    assert "CP-SAT status=FEASIBLE; due deferrals=1; late days=2" in command.output


def test_optimize_reports_failure(monkeypatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise OptimizationError("fixture optimization failure")

    monkeypatch.setattr("aeromaintain.cli.optimize_run", fail)

    command = runner.invoke(
        app,
        ["optimize", "--run-id", "fixture", "--project-root", str(tmp_path)],
    )

    assert command.exit_code == 1
    assert "Optimization failed: fixture optimization failure" in command.output
