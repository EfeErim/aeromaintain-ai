"""Deterministic RUL prediction metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

WILSON_Z_95 = 1.959963984540054
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


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = WILSON_Z_95,
) -> dict[str, float] | None:
    """Return the two-sided Wilson score interval for a proportion."""
    if trials <= 0:
        return None
    if not 0 <= successes <= trials:
        raise ValueError("Successes must be between zero and the trial count")
    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return {
        "low": 0.0 if successes == 0 else max(0.0, center - margin),
        "high": 1.0 if successes == trials else min(1.0, center + margin),
    }


def alert_rule_metrics(
    y_true: np.ndarray,
    alert_score: np.ndarray,
    *,
    critical_threshold: float = 30.0,
    alert_threshold: float | None = None,
) -> dict[str, Any]:
    """Score an ``alert_score <= threshold`` rule against truly critical engines.

    Each row must represent one engine, because the Wilson intervals treat rows as
    independent trials.
    """
    truth = np.asarray(y_true, dtype=np.float64)
    score = np.asarray(alert_score, dtype=np.float64)
    if len(truth) != len(score):
        raise ValueError("Alert rule arrays must have equal lengths")
    threshold = critical_threshold if alert_threshold is None else alert_threshold
    actual = truth <= critical_threshold
    alerted = score <= threshold
    true_positive = int(np.sum(actual & alerted))
    false_positive = int(np.sum(~actual & alerted))
    false_negative = int(np.sum(actual & ~alerted))
    true_negative = int(np.sum(~actual & ~alerted))
    alerts = true_positive + false_positive
    critical = true_positive + false_negative
    healthy = false_positive + true_negative
    precision = true_positive / alerts if alerts else 0.0
    recall = true_positive / critical if critical else 0.0
    return {
        "critical_threshold": float(critical_threshold),
        "alert_threshold": float(threshold),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "false_alarm_rate": false_positive / healthy if healthy else 0.0,
        "wilson_95": {
            "precision": wilson_interval(true_positive, alerts),
            "recall": wilson_interval(true_positive, critical),
            "false_alarm_rate": wilson_interval(false_positive, healthy),
        },
    }


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
