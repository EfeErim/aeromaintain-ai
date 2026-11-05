"""Seed sensitivity and byte-level repeatability of locked runs."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aeromaintain.data.pipeline import sha256_file
from aeromaintain.evaluation import alert_rule_metrics, prediction_metrics
from aeromaintain.evaluation.protocol import (
    apply_mondrian_cqr,
    apply_mondrian_interval,
)
from aeromaintain.features import build_causal_features, build_condition_features
from aeromaintain.models.evaluation import _validate_lock
from aeromaintain.models.rul import (
    TEST_PREVIOUSLY_SEEN,
    ModelingConfig,
    ModelingError,
    _feature_view,
    _json_bytes,
    _predict_model,
    _processed_dir,
    calibrate_interval,
    cross_fit_predictions,
    fit_final_model,
    fit_quantile_fixed,
    partition_features,
    predict_quantiles,
)

SEED_METRICS = (
    "mae",
    "rmse",
    "nasa_score_motor_normalized",
    "interval_coverage",
    "interval_mean_width",
    "point_rule_recall",
    "point_rule_precision",
    "interval_low_rule_recall",
    "interval_low_rule_precision",
    "interval_low_rule_false_alarms",
    "risk_threshold",
)
SKIPPED_LOCK_FIELDS = {"run_id"}


def _config_from_lock(lock: dict[str, Any]) -> ModelingConfig:
    known = {field.name for field in fields(ModelingConfig)}
    values = {
        name: tuple(value) if isinstance(value, list) else value
        for name, value in lock["config"]["modeling"].items()
        if name in known
    }
    return ModelingConfig(**values)


def _seed_result(
    spec: dict[str, Any],
    locked_calibration: dict[str, Any],
    features: pd.DataFrame,
    frame: pd.DataFrame,
    test: pd.DataFrame,
    final_mask: pd.Series,
    evaluation: pd.DataFrame,
    config: ModelingConfig,
    seed: int,
) -> dict[str, Any]:
    kind = locked_calibration.get("quantile_target")
    cross_fit = cross_fit_predictions(
        spec,
        features,
        frame,
        config,
        seed=seed,
        quantile_kinds=(kind,) if kind else (),
    )
    calibration, _ = calibrate_interval(
        cross_fit,
        frame,
        config,
        seed=seed,
        forced_method=f"cqr_{kind}" if kind else "mondrian",
    )
    rows = np.arange(len(frame))
    final_features, normalizer = partition_features(spec, frame, features, rows, config)
    model, preprocessor = fit_final_model(
        spec, final_features, frame, config, seed=seed
    )
    test_features = (
        build_causal_features(test)
        if normalizer is None
        else build_condition_features(test, normalizer)
    )
    last_features = test_features.loc[final_mask].reset_index(drop=True)
    prediction = _predict_model(model, preprocessor, last_features, spec["family"])
    if kind:
        quantile_model, quantile_preprocessor = fit_quantile_fixed(
            spec,
            kind,
            calibration["quantile_tree_count"],
            _feature_view(final_features, spec),
            frame,
            config,
            seed=seed,
        )
        low, high = apply_mondrian_cqr(
            prediction,
            predict_quantiles(quantile_model, quantile_preprocessor, last_features),
            calibration,
        )
    else:
        low, high = apply_mondrian_interval(prediction, calibration)
    truth = evaluation["rul_true"].to_numpy(dtype=np.float64)
    metrics = prediction_metrics(
        truth,
        prediction,
        evaluation["unit_id"].to_numpy(),
        critical_threshold=config.critical_rul_threshold,
    )
    point = alert_rule_metrics(
        truth, prediction, critical_threshold=config.critical_rul_threshold
    )
    risk = alert_rule_metrics(
        truth,
        low,
        critical_threshold=config.critical_rul_threshold,
        alert_threshold=calibration["risk_threshold"],
    )
    return {
        "seed": seed,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "nasa_score_motor_normalized": metrics["nasa_score_motor_normalized"],
        "interval_coverage": float(np.mean((truth >= low) & (truth <= high))),
        "interval_mean_width": float(np.mean(high - low)),
        "point_rule_recall": point["recall"],
        "point_rule_precision": point["precision"],
        "interval_low_rule_recall": risk["recall"],
        "interval_low_rule_precision": risk["precision"],
        "interval_low_rule_false_alarms": risk["false_positive"],
        "risk_threshold": calibration["risk_threshold"],
    }


def run_seed_sweep(
    project_root: Path,
    *,
    base_run_id: str,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Refit a locked specification across seeds and score its official test set."""
    root = project_root.resolve()
    if not seeds:
        raise ModelingError("A seed sweep requires at least one seed")
    _, lock = _validate_lock(root, base_run_id)
    if lock["protocol"]["test_status"] != TEST_PREVIOUSLY_SEEN:
        raise ModelingError(
            "Seed sweeps score test labels, so they run only on a subset whose "
            "labels were already seen"
        )
    config = _config_from_lock(lock)
    decision = lock["champion_rule_evidence"]
    spec = decision[decision["champion"]]["spec"]

    processed = _processed_dir(root, lock["dataset"])
    frame = pd.read_parquet(processed / "train.parquet").reset_index(drop=True)
    features = build_causal_features(frame)
    test = pd.read_parquet(processed / "test.parquet")
    final_mask = (
        test.groupby("unit_id", sort=False)["cycle"].transform("max").eq(test["cycle"])
    )
    labels = pd.read_parquet(processed / "evaluation" / "test_rul.parquet")
    evaluation = (
        test.loc[final_mask, ["unit_id"]]
        .reset_index(drop=True)
        .merge(labels, on="unit_id", validate="one_to_one")
    )

    rows = [
        _seed_result(
            spec,
            lock["calibration"],
            features,
            frame,
            test,
            final_mask,
            evaluation,
            config,
            seed,
        )
        for seed in seeds
    ]
    table = pd.DataFrame(rows)
    return {
        "schema_version": 1,
        "dataset": lock["dataset"],
        "base_run_id": base_run_id,
        "protocol_label": lock["protocol"]["label"],
        "locked_spec": spec,
        "scope": (
            "Seeds vary cross-fit folds, model randomness, draws, and the final "
            "fit. Selection is not rerun, so spread excludes selection variance."
        ),
        "seeds": list(seeds),
        "per_seed": rows,
        "summary": {
            name: {
                "mean": float(table[name].mean()),
                "std": float(table[name].std(ddof=1)) if len(table) > 1 else 0.0,
                "min": float(table[name].min()),
                "max": float(table[name].max()),
            }
            for name in SEED_METRICS
        },
    }


def _hashes_under(run_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(run_dir).as_posix(): sha256_file(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    }


def _changed_lock_fields(
    left: dict[str, Any],
    right: dict[str, Any],
) -> list[str]:
    return sorted(
        name
        for name in set(left) | set(right)
        if name not in SKIPPED_LOCK_FIELDS and left.get(name) != right.get(name)
    )


def compare_runs(
    project_root: Path,
    *,
    first_run_id: str,
    second_run_id: str,
) -> dict[str, Any]:
    """Report which files of two runs repeat byte for byte."""
    root = project_root.resolve()
    first_dir = root / "runs" / first_run_id
    second_dir = root / "runs" / second_run_id
    for run_dir in (first_dir, second_dir):
        if not (run_dir / "model_lock.json").is_file():
            raise ModelingError(f"Run has no model lock: {run_dir}")
    first = _hashes_under(first_dir)
    second = _hashes_under(second_dir)
    first_lock = json.loads((first_dir / "model_lock.json").read_text(encoding="utf-8"))
    second_lock = json.loads(
        (second_dir / "model_lock.json").read_text(encoding="utf-8")
    )
    shared = sorted(set(first) & set(second))
    return {
        "schema_version": 1,
        "first_run_id": first_run_id,
        "second_run_id": second_run_id,
        "identical_files": [name for name in shared if first[name] == second[name]],
        "different_files": [name for name in shared if first[name] != second[name]],
        "only_in_first": sorted(set(first) - set(second)),
        "only_in_second": sorted(set(second) - set(first)),
        "changed_lock_fields_ignoring_run_id": _changed_lock_fields(
            first_lock, second_lock
        ),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write one new JSON report and refuse to replace a different one."""
    payload = _json_bytes(report)
    if path.exists() and path.read_bytes() != payload:
        raise ModelingError(f"Refusing to overwrite different report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
