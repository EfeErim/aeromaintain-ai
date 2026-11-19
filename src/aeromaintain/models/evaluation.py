"""Official evaluation for an already verified and immutable model lock."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aeromaintain.artifacts import ArtifactValidationError, verify_lock
from aeromaintain.evaluation import alert_rule_metrics, prediction_metrics
from aeromaintain.evaluation.ledger import claim_holdout
from aeromaintain.evaluation.protocol import bootstrap_official_metrics, interval_report
from aeromaintain.features import CONDITION_FEATURE, FEATURE_NAMES
from aeromaintain.files import (
    json_bytes,
    parquet_bytes,
    processed_dir,
    sha256_file,
    write_once,
)
from aeromaintain.models.inference import (
    load_locked_model,
    locked_features,
    locked_interval,
    predict_model,
)
from aeromaintain.models.rul import (
    TEST_CLEAN_SINGLE_OPEN,
    EvaluationResult,
    ModelingError,
)

HOLDOUT_LEDGER = "holdout_ledger.json"


def validate_lock(
    project_root: Path,
    run_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Verify a lock with ``artifacts.verify_lock`` and check its feature names.

    Returns the run directory and the lock as stored, for code that refits or
    evaluates the locked model.
    """
    try:
        verified = verify_lock(project_root, run_id)
    except ArtifactValidationError as exc:
        raise ModelingError(str(exc)) from exc
    lock = verified.raw_lock
    features = lock.get("features")
    order = features.get("order") if isinstance(features, dict) else None
    if not isinstance(order, list) or not set(order) <= {
        *FEATURE_NAMES,
        CONDITION_FEATURE,
    }:
        raise ModelingError("Model lock feature order mismatch")
    return verified.run_dir, lock


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


def _official_metrics(
    evaluation: pd.DataFrame,
    lock: dict[str, Any],
    prediction: np.ndarray,
    interval_low: np.ndarray,
    interval_high: np.ndarray,
    risk_threshold: float,
) -> dict[str, Any]:
    """Point, alert-rule, interval and bootstrap metrics for one row per engine."""
    truth = evaluation["rul_true"].to_numpy(dtype=np.float64)
    calibration = lock["calibration"]
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
    return metrics


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
    metrics = _official_metrics(
        evaluation, lock, prediction, interval_low, interval_high, risk_threshold
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
    write_once(output_dir / "predictions.parquet", prediction_payload, ModelingError)
    write_once(output_dir / "metrics.json", metrics_payload, ModelingError)
    write_once(output_dir / "error_analysis.json", errors_payload, ModelingError)
    evaluation_manifest = {
        "run_id": run_id,
        "model_lock_sha256": sha256_file(run_dir / "model_lock.json"),
        "predictions_sha256": hashlib.sha256(prediction_payload).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_payload).hexdigest(),
        "error_analysis_sha256": hashlib.sha256(errors_payload).hexdigest(),
        "champion_unchanged": lock["champion"]["kind"],
        "official_labels_usage": "locked evaluation only",
    }
    write_once(
        output_dir / "evaluation_manifest.json",
        json_bytes(evaluation_manifest),
        ModelingError,
    )
    return EvaluationResult(
        run_id=run_id,
        output_dir=output_dir,
        metrics=metrics,
    )
