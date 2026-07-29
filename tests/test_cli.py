from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from aeromaintain.cli import app, collect_doctor_checks
from aeromaintain.data import DataPipelineError, PrepareResult

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_doctor_reports_a_healthy_repository() -> None:
    result = runner.invoke(
        app,
        ["doctor", "--project-root", str(REPOSITORY_ROOT)],
    )
    assert result.exit_code == 0, result.output
    assert "[PASS] python:" in result.output
    assert "[PASS] package:" in result.output
    assert "Summary: 5/5 checks passed" in result.output


def test_doctor_rejects_an_unsupported_python_version() -> None:
    checks = collect_doctor_checks(REPOSITORY_ROOT, version_info=(3, 12, 0))
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
