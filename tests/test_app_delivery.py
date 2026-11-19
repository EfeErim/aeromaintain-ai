from __future__ import annotations

import json
from pathlib import Path

import pytest
from run_fixture import build_verified_run, write_json
from streamlit.testing.v1 import AppTest

from aeromaintain.artifacts import (
    export_public_bundle,
    load_public_bundle,
    load_verified_run,
)
from aeromaintain.data import PrepareResult
from aeromaintain.delivery import DeliveryError, run_pipeline
from aeromaintain.models.rul import EvaluationResult, TrainResult

ROOT = Path(__file__).resolve().parents[1]
APP_SCRIPT = str(ROOT / "src" / "aeromaintain" / "app" / "main.py")


def test_verified_app_loader_excludes_truth_and_only_exposes_risk_download(
    tmp_path: Path,
) -> None:
    build_verified_run(tmp_path)

    artifacts = load_verified_run(tmp_path, "fixture")

    assert artifacts.run_id == "fixture"
    assert "rul_true" not in artifacts.risk_ranking.columns
    assert artifacts.risk_ranking.loc[0, "prediction"] == 9.5
    assert list(artifacts.sensor_history["unit_id"]) == [1]


def test_streamlit_small_fixture_renders_active_pages(
    monkeypatch,
    tmp_path: Path,
) -> None:
    build_verified_run(tmp_path)
    monkeypatch.setenv("AEROMAINTAIN_RUN_ID", "fixture")
    monkeypatch.setenv("AEROMAINTAIN_PROJECT_ROOT", str(tmp_path))
    tested = AppTest.from_file(APP_SCRIPT, default_timeout=30).run()
    assert not tested.exception
    assert tested.header[0].value == "Overview"

    tested.sidebar.radio[0].set_value("Engine risk")
    tested.run()
    assert not tested.exception
    assert tested.header[0].value == "Engine risk"


def test_public_bundle_has_no_rows_or_truth_and_renders(
    monkeypatch,
    tmp_path: Path,
) -> None:
    build_verified_run(tmp_path)
    bundle = tmp_path / "bundle"
    export_public_bundle(tmp_path, "fixture", bundle)

    exported = (bundle / "risk_ranking.csv").read_text(encoding="utf-8")
    run_payload = (bundle / "run.json").read_text(encoding="utf-8")
    assert "rul_true" not in exported
    assert "sensor_" not in exported
    assert "sensor_history" not in run_payload
    assert load_public_bundle(bundle).sensor_history.empty

    monkeypatch.delenv("AEROMAINTAIN_RUN_ID", raising=False)
    monkeypatch.setenv("AEROMAINTAIN_PUBLIC_BUNDLE", str(bundle))
    tested = AppTest.from_file(APP_SCRIPT, default_timeout=30).run()
    assert not tested.exception
    tested.sidebar.radio[0].set_value("Engine risk")
    tested.run()
    assert not tested.exception
    assert any("local-only" in item.value for item in tested.caption)


def test_tracked_demo_entrypoint_renders_without_nasa_rows(monkeypatch) -> None:
    monkeypatch.delenv("AEROMAINTAIN_RUN_ID", raising=False)
    monkeypatch.delenv("AEROMAINTAIN_PUBLIC_BUNDLE", raising=False)
    tracked = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((ROOT / "demo").iterdir())
    )
    assert "rul_true" not in tracked
    assert "sensor_history" not in tracked
    assert "setting_1," not in tracked

    tested = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
    assert not tested.exception
    assert tested.header[0].value == "Overview"
    tested.sidebar.radio[0].set_value("Engine risk")
    tested.run()
    assert not tested.exception


def test_small_fixture_pipeline_completes_evaluation_and_report(
    tmp_path: Path,
) -> None:
    calls = []

    def prepare(root, *, dataset, archive_path=None):
        calls.append(("prepare", archive_path))
        return PrepareResult(root, 1, 1, 1, 0, {})

    def train(root, *, run_id, dataset):
        calls.append(("train", run_id))
        run_dir = build_verified_run(root, run_id)
        return TrainResult(run_id, run_dir, "ridge", 5.0, run_dir / "model_lock.json")

    def evaluate(root, *, run_id):
        calls.append(("evaluate", run_id))
        return EvaluationResult(
            run_id,
            root / "runs" / run_id / "official_test",
            {"mae": 0.5},
        )

    result = run_pipeline(
        tmp_path,
        run_id="fixture",
        prepare_step=prepare,
        train_step=train,
        evaluation_step=evaluate,
    )

    assert [name for name, _ in calls] == ["prepare", "train", "evaluate"]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "pipeline_complete"
    assert manifest["pipeline"]["stages"] == [
        "prepare",
        "train_and_lock",
        "evaluate_locked",
        "report",
    ]
    assert result.report_path.is_file()
    load_verified_run(tmp_path, "fixture")


def test_pipeline_refuses_overwrite_and_does_not_mark_partial_complete(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "runs" / "existing"
    existing.mkdir(parents=True)
    with pytest.raises(DeliveryError, match="already exists"):
        run_pipeline(tmp_path, run_id="existing")

    def prepare(root, *, dataset, archive_path=None):
        return PrepareResult(root, 1, 1, 1, 0, {})

    def train(root, *, run_id, dataset):
        run_dir = root / "runs" / run_id
        run_dir.mkdir(parents=True)
        write_json(
            run_dir / "manifest.json", {"run_id": run_id, "status": "model_locked"}
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
