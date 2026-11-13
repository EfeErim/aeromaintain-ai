"""Exit codes and output of the CLI commands that read verified runs."""

from __future__ import annotations

import subprocess
from pathlib import Path

from run_fixture import build_verified_run
from typer.testing import CliRunner

from aeromaintain.artifacts import load_public_bundle
from aeromaintain.cli import app

runner = CliRunner()


def test_export_demo_writes_a_loadable_bundle(tmp_path: Path) -> None:
    build_verified_run(tmp_path)
    out = tmp_path / "bundle"
    result = runner.invoke(
        app,
        [
            "export-demo",
            "--run-id",
            "fixture",
            "--output-dir",
            str(out),
            "--project-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert load_public_bundle(out).run_id == "fixture"


def test_export_demo_fails_for_unverified_run(tmp_path: Path) -> None:
    run_dir = build_verified_run(tmp_path)
    (run_dir / "explanation.json").write_text("{}\n", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "export-demo",
            "--run-id",
            "fixture",
            "--output-dir",
            str(tmp_path / "bundle"),
            "--project-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "Export failed" in result.output
    assert not (tmp_path / "bundle" / "manifest.json").exists()


def test_app_command_verifies_before_launching(monkeypatch, tmp_path: Path) -> None:
    launched = []

    def fake_run(args, check, env):
        launched.append(env["AEROMAINTAIN_RUN_ID"])
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("aeromaintain.cli.subprocess.run", fake_run)
    missing = runner.invoke(
        app, ["app", "--run-id", "fixture", "--project-root", str(tmp_path)]
    )
    assert missing.exit_code == 1
    assert "Application failed" in missing.output
    assert launched == []

    build_verified_run(tmp_path)
    ok = runner.invoke(
        app, ["app", "--run-id", "fixture", "--project-root", str(tmp_path)]
    )
    assert ok.exit_code == 0, ok.output
    assert launched == ["fixture"]


def test_pipeline_command_exit_codes(monkeypatch, tmp_path: Path) -> None:
    from aeromaintain.delivery import DeliveryError, PipelineResult

    def fail(root, **kwargs):
        raise DeliveryError("fixture stop")

    monkeypatch.setattr("aeromaintain.cli.run_pipeline", fail)
    failed = runner.invoke(
        app, ["pipeline", "--run-id", "x", "--project-root", str(tmp_path)]
    )
    assert failed.exit_code == 1
    assert "Pipeline failed: fixture stop" in failed.output

    def succeed(root, *, run_id, dataset, archive_path):
        run_dir = root / "runs" / run_id
        return PipelineResult(
            run_id, run_dir, run_dir / "report.html", run_dir / "manifest.json"
        )

    monkeypatch.setattr("aeromaintain.cli.run_pipeline", succeed)
    ok = runner.invoke(
        app,
        [
            "pipeline",
            "--run-id",
            "x",
            "--dataset",
            "FD003",
            "--project-root",
            str(tmp_path),
        ],
    )
    assert ok.exit_code == 0, ok.output
    assert "pipeline complete" in ok.output
