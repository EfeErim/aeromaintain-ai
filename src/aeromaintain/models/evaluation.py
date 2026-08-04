"""Official evaluation for an already verified and immutable model lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from aeromaintain.data.pipeline import sha256_file
from aeromaintain.evaluation import prediction_metrics
from aeromaintain.features import FEATURE_NAMES, FoldPreprocessor, build_causal_features
from aeromaintain.models.rul import (
    EvaluationResult,
    ModelingError,
    RidgeBundle,
    _json_bytes,
    _parquet_bytes,
    _predict_model,
    _write_json,
    _write_new_bytes,
)


def _resolve_locked_artifact(run_dir: Path, relative_path: str) -> Path:
    path = (run_dir / relative_path).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ModelingError("Model lock references an external artifact") from exc
    if not path.is_file():
        raise ModelingError(f"Locked artifact is missing: {relative_path}")
    return path


def _validate_lock(
    project_root: Path,
    run_id: str,
) -> tuple[Path, dict[str, Any]]:
    run_dir = (project_root / "runs" / run_id).resolve()
    lock_path = run_dir / "model_lock.json"
    if not lock_path.is_file():
        raise ModelingError("Official evaluation requires model_lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("run_id") != run_id or lock.get("schema_version") != 1:
        raise ModelingError("Model lock identity or schema is invalid")

    processed = project_root / "data" / "processed" / "fd001"
    current_hashes = {
        "manifest_sha256": sha256_file(processed / "manifest.json"),
        "split_manifest_sha256": sha256_file(processed / "split_manifest.json"),
        "train_sha256": sha256_file(processed / "train.parquet"),
        "test_sha256": sha256_file(processed / "test.parquet"),
        "official_test_rul_sha256": sha256_file(
            processed / "evaluation" / "test_rul.parquet"
        ),
    }
    for name, observed in current_hashes.items():
        if lock["data"].get(name) != observed:
            raise ModelingError(f"Model lock data hash mismatch: {name}")
    if lock["config"]["sha256"] != sha256_file(
        project_root / "configs" / "project.yaml"
    ):
        raise ModelingError("Model lock config hash mismatch")
    if tuple(lock["features"]["order"]) != FEATURE_NAMES:
        raise ModelingError("Model lock feature order mismatch")
    for relative_path, expected_hash in lock["artifacts"].items():
        artifact = _resolve_locked_artifact(run_dir, relative_path)
        if sha256_file(artifact) != expected_hash:
            raise ModelingError(f"Locked artifact hash mismatch: {relative_path}")
    return run_dir, lock


def _load_locked_model(
    run_dir: Path,
    lock: dict[str, Any],
) -> tuple[Any, FoldPreprocessor, str]:
    champion = lock["champion"]
    kind = champion["kind"]
    if kind == "ridge":
        bundle = joblib.load(_resolve_locked_artifact(run_dir, champion["path"]))
        if not isinstance(bundle, RidgeBundle):
            raise ModelingError("Trusted local Ridge artifact has wrong type")
        return bundle, bundle.preprocessor, kind
    if kind == "xgboost":
        preprocessor = joblib.load(
            _resolve_locked_artifact(run_dir, champion["preprocessor_path"])
        )
        if not isinstance(preprocessor, FoldPreprocessor):
            raise ModelingError("Trusted local preprocessor has wrong type")
        model = XGBRegressor()
        model.load_model(_resolve_locked_artifact(run_dir, champion["path"]))
        return model, preprocessor, kind
    raise ModelingError(f"Unsupported locked champion kind: {kind}")


def evaluate_locked(
    project_root: Path,
    *,
    run_id: str,
) -> EvaluationResult:
    """Evaluate a verified lock against official labels without changing it."""
    root = project_root.resolve()
    run_dir, lock = _validate_lock(root, run_id)
    model, preprocessor, champion = _load_locked_model(run_dir, lock)

    processed = root / "data" / "processed" / "fd001"
    test = pd.read_parquet(processed / "test.parquet")
    test_features = build_causal_features(test)
    final_mask = (
        test.groupby("unit_id", sort=False)["cycle"].transform("max").eq(test["cycle"])
    )
    final_rows = test.loc[final_mask].reset_index(drop=True)
    final_features = test_features.loc[final_mask].reset_index(drop=True)
    prediction = _predict_model(model, preprocessor, final_features, champion)

    # Official labels are semantically opened only after lock and model validation.
    labels = pd.read_parquet(processed / "evaluation" / "test_rul.parquet")
    evaluation = final_rows.loc[:, ["unit_id", "cycle"]].merge(
        labels,
        on="unit_id",
        validate="one_to_one",
    )
    evaluation["prediction"] = prediction
    q = float(lock["calibration"]["q"])
    evaluation["interval_low"] = np.maximum(0.0, prediction - q)
    evaluation["interval_high"] = prediction + q
    evaluation["risk_band"] = np.select(
        [
            evaluation["interval_low"] <= 30,
            evaluation["interval_low"] <= 60,
        ],
        ["critical", "elevated"],
        default="routine",
    )
    evaluation["interval_contains_true_rul"] = evaluation["rul_true"].between(
        evaluation["interval_low"], evaluation["interval_high"]
    )
    metrics = prediction_metrics(
        evaluation["rul_true"].to_numpy(),
        evaluation["prediction"].to_numpy(),
        evaluation["unit_id"].to_numpy(),
        critical_threshold=float(lock["config"]["modeling"]["critical_rul_threshold"]),
    )
    metrics["nominal_empirical_interval"] = {
        "nominal_coverage": lock["calibration"]["coverage"],
        "observed_official_test_coverage": float(
            evaluation["interval_contains_true_rul"].mean()
        ),
        "mean_width": float(
            (evaluation["interval_high"] - evaluation["interval_low"]).mean()
        ),
        "label": "empirical prediction interval; not a safety guarantee",
    }
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
    prediction_payload = _parquet_bytes(evaluation)
    metrics_payload = _json_bytes(metrics)
    errors_payload = _json_bytes(error_analysis)
    _write_new_bytes(output_dir / "predictions.parquet", prediction_payload)
    _write_new_bytes(output_dir / "metrics.json", metrics_payload)
    _write_new_bytes(output_dir / "error_analysis.json", errors_payload)
    evaluation_manifest = {
        "run_id": run_id,
        "model_lock_sha256": sha256_file(run_dir / "model_lock.json"),
        "predictions_sha256": hashlib.sha256(prediction_payload).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_payload).hexdigest(),
        "error_analysis_sha256": hashlib.sha256(errors_payload).hexdigest(),
        "champion_unchanged": lock["champion"]["kind"],
        "official_labels_usage": "locked evaluation only",
    }
    _write_json(output_dir / "evaluation_manifest.json", evaluation_manifest)
    return EvaluationResult(
        run_id=run_id,
        output_dir=output_dir,
        metrics=metrics,
    )
