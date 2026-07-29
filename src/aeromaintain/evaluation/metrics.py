"""Deterministic RUL prediction metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

RUL_BANDS = (
    ("0-30", 0, 30),
    ("31-60", 31, 60),
    ("61-125", 61, 125),
    (">125", 126, None),
)


def nasa_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    engine_ids: np.ndarray,
) -> float:
    """Return the motor-normalized NASA asymmetric score."""
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    engines = np.asarray(engine_ids)
    error = prediction - truth
    penalty = np.where(
        error < 0.0,
        np.expm1(-error / 13.0),
        np.expm1(error / 10.0),
    )
    frame = pd.DataFrame({"engine": engines, "penalty": penalty})
    return float(frame.groupby("engine", sort=True)["penalty"].mean().sum())


def prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    engine_ids: np.ndarray,
    *,
    critical_threshold: float = 30.0,
) -> dict[str, Any]:
    """Compute the shared development and official-test metric contract."""
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.maximum(0.0, np.asarray(y_pred, dtype=np.float64))
    engines = np.asarray(engine_ids)
    if not (len(truth) == len(prediction) == len(engines)):
        raise ValueError("Metric arrays must have equal lengths")
    if not len(truth):
        raise ValueError("Metric arrays must not be empty")

    actual_critical = truth <= critical_threshold
    predicted_critical = prediction <= critical_threshold
    true_positive = int(np.sum(actual_critical & predicted_critical))
    false_positive = int(np.sum(~actual_critical & predicted_critical))
    false_negative = int(np.sum(actual_critical & ~predicted_critical))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    band_metrics: dict[str, Any] = {}
    for label, lower, upper in RUL_BANDS:
        mask = truth >= lower
        if upper is not None:
            mask &= truth <= upper
        band_metrics[label] = {
            "rows": int(mask.sum()),
            "mae": (
                float(mean_absolute_error(truth[mask], prediction[mask]))
                if mask.any()
                else None
            ),
        }

    return {
        "rows": len(truth),
        "engines": int(pd.Series(engines).nunique()),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "nasa_score_motor_normalized": nasa_score(truth, prediction, engines),
        "signed_bias": float(np.mean(prediction - truth)),
        "overprediction_rate": float(np.mean(prediction > truth)),
        "critical_rul": {
            "threshold": critical_threshold,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "rul_bands": band_metrics,
    }
