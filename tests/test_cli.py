from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from aeromaintain.cli import app, collect_doctor_checks

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
