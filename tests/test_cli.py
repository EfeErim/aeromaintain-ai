from __future__ import annotations

import subprocess
from pathlib import Path

from run_fixture import build_verified_run
from typer.testing import CliRunner

from aeromaintain.artifacts import load_public_bundle
from aeromaintain.cli import app
from aeromaintain.data import DataPipelineError
from aeromaintain.models import ModelingError

runner = CliRunner()


def test_stage_failures_exit_with_a_message(monkeypatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        raise ModelingError("fixture lock failure")

    def fail_prepare(*args, **kwargs):
        raise DataPipelineError("fixture failure")

    monkeypatch.setattr("aeromaintain.cli.prepare_dataset", fail_prepare)
    monkeypatch.setattr("aeromaintain.cli.train_and_lock", fail)
    monkeypatch.setattr("aeromaintain.cli.evaluate_locked", fail)
    prepared = runner.invoke(app, ["prepare", "--project-root", str(tmp_path)])
    assert prepared.exit_code == 1
    assert "Preparation failed: fixture failure" in prepared.output
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


def test_export_demo_writes_a_bundle_only_for_a_verified_run(tmp_path: Path) -> None:
    run_dir = build_verified_run(tmp_path)

    def export(out: Path):
        return runner.invoke(
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

    exported = export(tmp_path / "bundle")
    assert exported.exit_code == 0, exported.output
    assert load_public_bundle(tmp_path / "bundle").run_id == "fixture"

    (run_dir / "explanation.json").write_text("{}\n", encoding="utf-8")
    refused = export(tmp_path / "refused")
    assert refused.exit_code == 1
    assert "Export failed" in refused.output


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


def test_study_commands_write_each_report_once(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "aeromaintain.cli.compare_runs",
        lambda root, *, first_run_id, second_run_id: {
            "identical_files": ["a"],
            "different_files": [],
        },
    )
    arguments = [
        "compare-runs",
        "--first",
        "one",
        "--second",
        "two",
        "--project-root",
        str(tmp_path),
    ]
    first = runner.invoke(app, arguments)
    assert first.exit_code == 0, first.output
    assert (tmp_path / "runs/one/robustness/compare_two.json").is_file()
    # Same report again is accepted; a different one is refused.
    assert runner.invoke(app, arguments).exit_code == 0
    monkeypatch.setattr(
        "aeromaintain.cli.compare_runs",
        lambda root, **kwargs: {"identical_files": [], "different_files": ["a"]},
    )
    changed = runner.invoke(app, arguments)
    assert changed.exit_code == 1
    assert "Refusing to overwrite" in changed.output

    unknown = runner.invoke(
        app, ["escalation-study", "--base-run-id", "one", "--method", "gru"]
    )
    assert unknown.exit_code == 2
