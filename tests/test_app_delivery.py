from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from aeromaintain.app import ArtifactValidationError, load_verified_run
from aeromaintain.data import PrepareResult
from aeromaintain.data.pipeline import SENSOR_COLUMNS, sha256_file
from aeromaintain.delivery import DeliveryError, run_pipeline
from aeromaintain.models.rul import EvaluationResult, TrainResult


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_verified_run(root: Path, run_id: str = "fixture") -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "project.yaml").write_text("seed: 42\n", encoding="utf-8")

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
    pd.DataFrame(
        {
            "unit_id": [1],
            "cycle": [1],
            "rul_true": [10],
            "prediction": [9.5],
            "interval_low": [4.5],
            "interval_high": [14.5],
            "risk_band": ["critical"],
        }
    ).to_parquet(official / "predictions.parquet", index=False)
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
    _write_json(
        official / "evaluation_manifest.json",
        {
            "run_id": run_id,
            "model_lock_sha256": sha256_file(run_dir / "model_lock.json"),
            "predictions_sha256": sha256_file(official / "predictions.parquet"),
            "metrics_sha256": sha256_file(official / "metrics.json"),
            "error_analysis_sha256": sha256_file(official / "error_analysis.json"),
        },
    )
    return run_dir


def test_verified_app_loader_excludes_truth_and_only_exposes_risk_download(
    tmp_path: Path,
) -> None:
    _build_verified_run(tmp_path)

    artifacts = load_verified_run(tmp_path, "fixture")

    assert artifacts.run_id == "fixture"
    assert "rul_true" not in artifacts.risk_ranking.columns
    assert artifacts.risk_ranking.loc[0, "prediction"] == 9.5
    assert set(artifacts.downloads) == {"risk_ranking.csv"}
    assert list(artifacts.sensor_history["unit_id"]) == [1]


def test_verified_app_loader_rejects_changed_or_external_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = _build_verified_run(tmp_path)
    (run_dir / "official_test" / "metrics.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="SHA-256 mismatch"):
        load_verified_run(tmp_path, "fixture")

    second_root = tmp_path / "second"
    second_run = _build_verified_run(second_root)
    lock_path = second_run / "model_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["config"]["path"] = "../outside.yaml"
    _write_json(lock_path, lock)
    manifest_path = second_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_lock_sha256"] = sha256_file(lock_path)
    _write_json(manifest_path, manifest)

    with pytest.raises(ArtifactValidationError, match="external path"):
        load_verified_run(second_root, "fixture")


def test_streamlit_small_fixture_renders_active_pages(
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

    tested.sidebar.radio[0].set_value("Engine risk")
    tested.run()
    assert not tested.exception
    assert tested.header[0].value == "Engine risk"


def test_small_fixture_pipeline_completes_evaluation_and_report(
    tmp_path: Path,
) -> None:
    calls = []

    def prepare(root, *, archive_path=None):
        calls.append(("prepare", archive_path))
        return PrepareResult(root, 1, 1, 1, 0, {})

    def train(root, *, run_id):
        calls.append(("train", run_id))
        run_dir = _build_verified_run(root, run_id)
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
    report = json.loads((result.run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["retired_scope"].startswith("Synthetic maintenance resources")
    assert result.report_path.is_file()
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
