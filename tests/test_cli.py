from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from aeromaintain.cli import app
from aeromaintain.data import DataPipelineError, PrepareResult
from aeromaintain.models import ModelingError
from aeromaintain.models.rul import EvaluationResult, TrainResult

runner = CliRunner()


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
        "aeromaintain.cli.prepare_dataset", lambda *args, **kwargs: result
    )

    command = runner.invoke(app, ["prepare", "--project-root", str(tmp_path)])

    assert command.exit_code == 0, command.output
    assert "FD001 preparation complete" in command.output
    assert "train=20631, test=13096" in command.output


def test_prepare_reports_pipeline_failure(monkeypatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise DataPipelineError("fixture failure")

    monkeypatch.setattr("aeromaintain.cli.prepare_dataset", fail)
    command = runner.invoke(app, ["prepare", "--project-root", str(tmp_path)])

    assert command.exit_code == 1
    assert "Preparation failed: fixture failure" in command.output


def test_train_and_evaluate_report_success(monkeypatch, tmp_path: Path) -> None:
    train_result = TrainResult(
        run_id="fixture",
        run_dir=tmp_path / "runs" / "fixture",
        champion="ridge",
        interval_half_width=12.5,
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
