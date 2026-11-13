"""Small synthetic run directories shared by the delivery and artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aeromaintain.data.pipeline import SENSOR_COLUMNS, sha256_file
from aeromaintain.evaluation import alert_rule_metrics


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_verified_run(root: Path, run_id: str = "fixture") -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "project.yaml").write_text("seed: 42\n", encoding="utf-8")

    processed = root / "data" / "processed" / "fd001"
    (processed / "evaluation").mkdir(parents=True)
    write_json(processed / "manifest.json", {"dataset": "FD001"})
    write_json(processed / "split_manifest.json", {"seed": 42})
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
    write_json(run_dir / "explanation.json", explanation)
    # Stands in for the Ridge bundle; the loader checks its hash, never unpickles it.
    (run_dir / "model.joblib").write_bytes(b"fixture model bytes")
    lock = {
        "schema_version": 2,
        "protocol": {
            "version": 2,
            "document": "docs/protocol_v2.md",
            "sha256": None,
            "test_status": "previously_seen",
            "label": "protocol v2, test previously seen",
        },
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
            "bands": [
                {
                    "upper_predicted_rul": None,
                    "engines": 1,
                    "rows": 1,
                    "q_low": -5.0,
                    "q_high": 5.0,
                }
            ],
            "risk_threshold": 30.0,
            "elevated_threshold": 60.0,
        },
        "artifacts": {
            "explanation.json": sha256_file(run_dir / "explanation.json"),
            "model.joblib": sha256_file(run_dir / "model.joblib"),
        },
    }
    write_json(run_dir / "model_lock.json", lock)
    write_json(
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
    fixture_rule = alert_rule_metrics(np.array([10.0]), np.array([9.5]))
    metrics = {
        "engines": 1,
        "mae": 0.5,
        "rmse": 0.5,
        "nasa_score_motor_normalized": 0.1,
        "critical_rul": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
        "alert_rules": {
            "point_prediction": fixture_rule,
            "interval_low": fixture_rule,
        },
        "nominal_empirical_interval": {
            "nominal_coverage": 0.9,
            "observed_official_test_coverage": 1.0,
            "mean_width": 10.0,
            "by_predicted_rul": {},
            "by_true_rul": {},
            "label": "empirical prediction interval; not a safety guarantee",
        },
        "bootstrap": {"intervals_95": {"rmse": [0.5, 0.5]}},
    }
    write_json(official / "metrics.json", metrics)
    write_json(official / "error_analysis.json", {"largest_absolute_errors": []})
    write_json(
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
