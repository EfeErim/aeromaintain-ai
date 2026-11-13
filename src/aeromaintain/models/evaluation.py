"""Official evaluation for an already verified and immutable model lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aeromaintain.data.pipeline import sha256_file
from aeromaintain.evaluation import alert_rule_metrics, prediction_metrics
from aeromaintain.evaluation.ledger import claim_holdout
from aeromaintain.evaluation.protocol import bootstrap_official_metrics, interval_report
from aeromaintain.features import CONDITION_FEATURE, FEATURE_NAMES
from aeromaintain.models.inference import (
    load_locked_model,
    locked_features,
    locked_interval,
    predict_model,
    resolve_locked_artifact,
)
from aeromaintain.models.rul import (
    SUPPORTED_LOCK_SCHEMAS,
    TEST_CLEAN_SINGLE_OPEN,
    EvaluationResult,
    ModelingError,
)
from aeromaintain.models.run_files import (
    json_bytes,
    parquet_bytes,
    processed_dir,
    write_json,
    write_new_bytes,
)

HOLDOUT_LEDGER = "holdout_ledger.json"


def _required_hash(path: Path) -> str:
    if not path.is_file():
        raise ModelingError(f"Locked input is missing: {path}")
    return sha256_file(path)


def _read_locked_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelingError(f"{label} is unreadable or invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ModelingError(f"{label} must contain a JSON object: {path}")
    return payload


def validate_lock(
    project_root: Path,
    run_id: str,
) -> tuple[Path, dict[str, Any]]:
    run_dir = (project_root / "runs" / run_id).resolve()
    lock_path = run_dir / "model_lock.json"
    if not lock_path.is_file():
        raise ModelingError("Official evaluation requires model_lock.json")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ModelingError("Official evaluation requires the run manifest")
    manifest = _read_locked_json(manifest_path, "Run manifest")
    if manifest.get("model_lock_sha256") != sha256_file(lock_path):
        raise ModelingError("Model lock SHA-256 mismatch against run manifest")
    lock = _read_locked_json(lock_path, "Model lock")
    if (
        lock.get("run_id") != run_id
        or lock.get("schema_version") not in SUPPORTED_LOCK_SCHEMAS
    ):
        raise ModelingError("Model lock identity or schema is invalid")

    try:
        processed = processed_dir(project_root, lock["dataset"])
        current_hashes = {
            "manifest_sha256": _required_hash(processed / "manifest.json"),
            "split_manifest_sha256": _required_hash(processed / "split_manifest.json"),
            "train_sha256": _required_hash(processed / "train.parquet"),
            "test_sha256": _required_hash(processed / "test.parquet"),
            "official_test_rul_sha256": _required_hash(
                processed / "evaluation" / "test_rul.parquet"
            ),
        }
        for name, observed in current_hashes.items():
            if lock["data"].get(name) != observed:
                raise ModelingError(f"Model lock data hash mismatch: {name}")
        if lock["config"]["sha256"] != _required_hash(
            project_root / lock["config"].get("path", "configs/project.yaml")
        ):
            raise ModelingError("Model lock config hash mismatch")
        if not set(lock["features"]["order"]) <= {*FEATURE_NAMES, CONDITION_FEATURE}:
            raise ModelingError("Model lock feature order mismatch")
        for relative_path, expected_hash in lock["artifacts"].items():
            artifact = resolve_locked_artifact(run_dir, relative_path)
            if sha256_file(artifact) != expected_hash:
                raise ModelingError(f"Locked artifact hash mismatch: {relative_path}")
    except (AttributeError, KeyError, TypeError) as exc:
        raise ModelingError(f"Model lock structure is invalid: {exc}") from exc
    return run_dir, lock


# Older private name, still imported by robustness and escalation studies.
_validate_lock = validate_lock


def _claim_single_open(project_root: Path, run_id: str, lock: dict[str, Any]) -> None:
    """Allow one model lock to open a clean holdout, and refuse any other."""
    if lock["protocol"]["test_status"] != TEST_CLEAN_SINGLE_OPEN:
        return
    ledger_path = project_root / "runs" / HOLDOUT_LEDGER
    dataset = lock["dataset"]
    lock_hash = sha256_file(project_root / "runs" / run_id / "model_lock.json")
    entry = {"run_id": run_id, "model_lock_sha256": lock_hash}
    try:
        claim_holdout(ledger_path, dataset, entry)
    except (ValueError, OSError) as error:
        raise ModelingError(str(error)) from error


def evaluate_locked(
    project_root: Path,
    *,
    run_id: str,
) -> EvaluationResult:
    """Evaluate a verified lock against official labels without changing it."""
    root = project_root.resolve()
    run_dir, lock = validate_lock(root, run_id)
    model, preprocessor, champion = load_locked_model(run_dir, lock)

    processed = processed_dir(root, lock["dataset"])
    test = pd.read_parquet(processed / "test.parquet")
    test_features = locked_features(run_dir, lock, test)
    final_mask = (
        test.groupby("unit_id", sort=False)["cycle"].transform("max").eq(test["cycle"])
    )
    final_rows = test.loc[final_mask].reset_index(drop=True)
    final_features = test_features.loc[final_mask].reset_index(drop=True)
    prediction = predict_model(model, preprocessor, final_features, champion)

    calibration = lock["calibration"]
    interval_low, interval_high = locked_interval(
        run_dir, calibration, prediction, final_features
    )
    risk_threshold = float(calibration["risk_threshold"])
    elevated_threshold = float(calibration["elevated_threshold"])

    # Official labels are semantically opened only after lock and model validation.
    _claim_single_open(root, run_id, lock)
    labels = pd.read_parquet(processed / "evaluation" / "test_rul.parquet")
    evaluation = final_rows.loc[:, ["unit_id", "cycle"]].merge(
        labels,
        on="unit_id",
        validate="one_to_one",
    )
    evaluation["prediction"] = prediction
    evaluation["interval_low"] = interval_low
    evaluation["interval_high"] = interval_high
    evaluation["risk_band"] = np.select(
        [
            evaluation["interval_low"] <= risk_threshold,
            evaluation["interval_low"] <= elevated_threshold,
        ],
        ["critical", "elevated"],
        default="routine",
    )
    evaluation["interval_contains_true_rul"] = evaluation["rul_true"].between(
        evaluation["interval_low"], evaluation["interval_high"]
    )
    truth = evaluation["rul_true"].to_numpy(dtype=np.float64)
    critical_threshold = float(lock["config"]["modeling"]["critical_rul_threshold"])
    metrics = prediction_metrics(
        truth,
        prediction,
        evaluation["unit_id"].to_numpy(),
        critical_threshold=critical_threshold,
    )
    metrics["nasa_score_mean_per_engine"] = metrics[
        "nasa_score_motor_normalized"
    ] / float(metrics["engines"])
    metrics["protocol"] = {
        "version": lock["protocol"]["version"],
        "test_status": lock["protocol"]["test_status"],
        "label": lock["protocol"]["label"],
    }
    metrics["alert_rules"] = {
        "scope": "one row per engine; intervals are two-sided Wilson 95%",
        "point_prediction": alert_rule_metrics(
            truth,
            prediction,
            critical_threshold=critical_threshold,
        ),
        "interval_low": alert_rule_metrics(
            truth,
            interval_low,
            critical_threshold=critical_threshold,
            alert_threshold=risk_threshold,
        ),
    }
    band_edges = tuple(lock["config"]["modeling"]["interval_band_edges"])
    report = interval_report(
        truth,
        prediction,
        interval_low,
        interval_high,
        edges=band_edges,
    )
    metrics["nominal_empirical_interval"] = {
        "nominal_coverage": calibration["coverage"],
        "observed_official_test_coverage": report["coverage"],
        "mean_width": report["mean_width"],
        "by_predicted_rul": report["by_predicted_rul"],
        "by_true_rul": report["by_true_rul"],
        "label": "empirical prediction interval; not a safety guarantee",
    }
    metrics["bootstrap"] = bootstrap_official_metrics(
        truth,
        prediction,
        interval_low,
        interval_high,
        critical_threshold=critical_threshold,
        alert_threshold=risk_threshold,
        resamples=int(lock["config"]["modeling"]["bootstrap_resamples"]),
        seed=int(lock["seed"]),
    )
    errors = evaluation.assign(
        absolute_error=lambda frame: (frame["prediction"] - frame["rul_true"]).abs(),
        signed_error=lambda frame: frame["prediction"] - frame["rul_true"],
    ).sort_values(["absolute_error", "unit_id"], ascending=[False, True])
    error_analysis = {
        "largest_absolute_errors": errors.head(20).to_dict(orient="records"),
        "overpredicted_engines": int(errors["signed_error"].gt(0).sum()),
        "underpredicted_engines": int(errors["signed_error"].lt(0).sum()),
        "note": "Official results did not alter model, features, thresholds, or q",
    }

    output_dir = run_dir / "official_test"
    prediction_payload = parquet_bytes(evaluation)
    metrics_payload = json_bytes(metrics)
    errors_payload = json_bytes(error_analysis)
    write_new_bytes(output_dir / "predictions.parquet", prediction_payload)
    write_new_bytes(output_dir / "metrics.json", metrics_payload)
    write_new_bytes(output_dir / "error_analysis.json", errors_payload)
    evaluation_manifest = {
        "run_id": run_id,
        "model_lock_sha256": sha256_file(run_dir / "model_lock.json"),
        "predictions_sha256": hashlib.sha256(prediction_payload).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_payload).hexdigest(),
        "error_analysis_sha256": hashlib.sha256(errors_payload).hexdigest(),
        "champion_unchanged": lock["champion"]["kind"],
        "official_labels_usage": "locked evaluation only",
    }
    write_json(output_dir / "evaluation_manifest.json", evaluation_manifest)
    return EvaluationResult(
        run_id=run_id,
        output_dir=output_dir,
        metrics=metrics,
    )
