from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from aeromaintain import synthetic
from aeromaintain.cli import app
from aeromaintain.data import prepare_dataset

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def test_archive_is_reproducible(synthetic_archive: bytes) -> None:
    assert synthetic.synthetic_archive_bytes() == synthetic_archive


@pytest.mark.usefixtures("reuse_synthetic_archive")
def test_project_passes_the_real_data_contract(tmp_path: Path) -> None:
    target = tmp_path / "project"
    archive = synthetic.build_synthetic_project(ROOT, target)

    config = yaml.safe_load((target / "configs/project.yaml").read_text("utf-8"))
    assert config["project"]["synthetic_data"] is True
    assert (
        config["data"]["archive_sha256"]
        == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert (target / synthetic.MARKER).is_file()

    result = prepare_dataset(target, dataset="FD001", archive_path=archive)
    assert (result.train_rows, result.test_rows) == (20_631, 13_096)
    assert (result.development_engines, result.calibration_engines) == (80, 20)
    # The real configuration is never touched.
    real = yaml.safe_load((ROOT / "configs/project.yaml").read_text("utf-8"))
    assert real["data"]["source_url"].startswith("https://")


def test_project_must_be_outside_the_repository(tmp_path: Path) -> None:
    with pytest.raises(synthetic.SyntheticProjectError, match="outside"):
        synthetic.build_synthetic_project(ROOT, ROOT / "synthetic-project")
    with pytest.raises(synthetic.SyntheticProjectError, match="outside"):
        synthetic.build_synthetic_project(ROOT, ROOT)


def test_project_refuses_a_non_synthetic_directory(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("user file", encoding="utf-8")
    with pytest.raises(synthetic.SyntheticProjectError, match="not empty"):
        synthetic.build_synthetic_project(ROOT, tmp_path)


def test_cli_synthetic_needs_a_project_root(monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    command = runner.invoke(app, ["pipeline", "--synthetic", "--run-id", "s"])
    assert command.exit_code == 2
    assert "--project-root" in command.output


@pytest.mark.usefixtures("reuse_synthetic_archive")
def test_cli_synthetic_runs_the_normal_pipeline(monkeypatch, tmp_path: Path) -> None:
    calls = {}

    def fake_pipeline(root, *, run_id, dataset, archive_path):
        calls.update(root=root, archive=archive_path, dataset=dataset)
        assert archive_path.read_bytes()[:2] == b"PK"
        return type(
            "Result", (), {"run_id": run_id, "manifest_path": "m", "report_path": "r"}
        )()

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr("aeromaintain.cli.run_pipeline", fake_pipeline)
    target = tmp_path / "synthetic"
    command = runner.invoke(
        app,
        ["pipeline", "--synthetic", "--run-id", "s", "--project-root", str(target)],
    )
    assert command.exit_code == 0, command.output
    assert calls["root"] == target.resolve()
    assert calls["dataset"] == "FD001"
    assert "SYNTHETIC" in command.output
