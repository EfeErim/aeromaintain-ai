"""Protocol v2 measurement: per-engine truncation, intervals, and uncertainty."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from aeromaintain.evaluation.metrics import (
    RUL_BANDS,
    alert_rule_metrics,
    nasa_score,
    prediction_metrics,
)

DRAW_METRICS = ("rmse", "mae", "nasa_score_motor_normalized", "signed_bias")


def eligible_rows(rul_true: np.ndarray, window: float | None) -> np.ndarray:
    """Return the rows a test-like truncation may land on."""
    truth = np.asarray(rul_true, dtype=np.float64)
    return np.ones(len(truth), dtype=bool) if window is None else truth <= window


def draw_engine_rows(
    engine_ids: np.ndarray,
    eligible: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Pick one uniformly random eligible row per engine for every draw."""
    engines = np.asarray(engine_ids)
    mask = np.asarray(eligible, dtype=bool)
    if draws < 1:
        raise ValueError("Truncation requires at least one draw")
    positions = pd.Series(np.flatnonzero(mask)).groupby(engines[mask], sort=True)
    groups = [group.to_numpy(dtype=np.int64) for _, group in positions]
    if not groups:
        raise ValueError("Truncation requires at least one eligible row")
    generator = np.random.default_rng(seed)
    return np.column_stack(
        [group[generator.integers(0, len(group), size=draws)] for group in groups]
    )


def summarize_draws(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    engine_ids: np.ndarray,
    plan: np.ndarray,
    *,
    critical_threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Score every draw with the official metric function and summarize it."""
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    engines = np.asarray(engine_ids)
    records: dict[str, list[float]] = {
        name: [] for name in (*DRAW_METRICS, "critical_recall", "critical_precision")
    }
    critical_engines: list[int] = []
    for rows in plan:
        metrics = prediction_metrics(
            truth[rows],
            prediction[rows],
            engines[rows],
            critical_threshold=critical_threshold,
        )
        for name in DRAW_METRICS:
            records[name].append(float(metrics[name]))
        critical = metrics["critical_rul"]
        records["critical_recall"].append(float(critical["recall"]))
        records["critical_precision"].append(float(critical["precision"]))
        critical_engines.append(critical["true_positive"] + critical["false_negative"])
    summary: dict[str, Any] = {
        "draws": len(plan),
        "engines": int(plan.shape[1]),
        "median_critical_engines": float(np.median(critical_engines)),
    }
    for name, values in records.items():
        observed = np.asarray(values, dtype=np.float64)
        summary[name] = {
            "median": float(np.median(observed)),
            "p05": float(np.percentile(observed, 5)),
            "p95": float(np.percentile(observed, 95)),
        }
    return summary, np.asarray(records["rmse"], dtype=np.float64)


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    level: float,
) -> float:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    cumulative = np.cumsum(weights[order]) / np.sum(weights)
    position = int(np.searchsorted(cumulative, level, side="left"))
    return float(ordered[min(position, len(ordered) - 1)])


def _merged_bands(
    prediction: np.ndarray,
    engines: np.ndarray,
    edges: tuple[float, ...],
    min_engines: int,
) -> list[tuple[float | None, np.ndarray, int]]:
    """Return ``(upper, row mask, engine count)`` with small bands merged upward."""
    bands: list[tuple[float | None, np.ndarray, int]] = []
    lower = -math.inf
    pending = np.zeros(len(prediction), dtype=bool)
    for upper in (*edges, None):
        limit = math.inf if upper is None else float(upper)
        pending = pending | ((prediction > lower) & (prediction <= limit))
        lower = limit
        count = int(pd.Series(engines[pending]).nunique())
        if count >= min_engines or (upper is None and not bands):
            bands.append((upper, pending, count))
            pending = np.zeros(len(prediction), dtype=bool)
    if pending.any():
        previous_mask = bands[-1][1] | pending
        bands[-1] = (
            None,
            previous_mask,
            int(pd.Series(engines[previous_mask]).nunique()),
        )
    return bands


def fit_mondrian_interval(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    engine_ids: np.ndarray,
    *,
    coverage: float,
    edges: tuple[float, ...],
    min_engines: int,
) -> dict[str, Any]:
    """Fit engine-weighted signed-residual quantiles per predicted-RUL band."""
    prediction = np.asarray(y_pred, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.float64)
    engines = np.asarray(engine_ids)
    if not len(prediction):
        raise ValueError("Interval calibration requires at least one row")
    tail = (1.0 - coverage) / 2.0
    residual = truth - prediction
    bands: list[dict[str, Any]] = []
    for upper, mask, engine_count in _merged_bands(
        prediction, engines, edges, min_engines
    ):
        if not mask.any():
            continue
        counts = pd.Series(engines[mask]).map(pd.Series(engines[mask]).value_counts())
        weights = 1.0 / counts.to_numpy(dtype=np.float64)
        high_level = min(1.0, (1.0 - tail) * (engine_count + 1) / engine_count)
        q_high = _weighted_quantile(residual[mask], weights, high_level)
        q_low = _weighted_quantile(residual[mask], weights, 1.0 - high_level)
        bands.append(
            {
                "upper_predicted_rul": upper,
                "engines": engine_count,
                "rows": int(mask.sum()),
                # The point prediction always stays inside its interval.
                "q_low": min(0.0, q_low),
                "q_high": max(0.0, q_high),
            }
        )
    return {"coverage": coverage, "bands": bands}


def apply_mondrian_interval(
    y_pred: np.ndarray,
    interval: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return non-negative lower and upper bounds for predictions."""
    prediction = np.asarray(y_pred, dtype=np.float64)
    low = np.empty(len(prediction), dtype=np.float64)
    high = np.empty(len(prediction), dtype=np.float64)
    lower = -math.inf
    for band in interval["bands"]:
        upper = band["upper_predicted_rul"]
        limit = math.inf if upper is None else float(upper)
        mask = (prediction > lower) & (prediction <= limit)
        low[mask] = prediction[mask] + band["q_low"]
        high[mask] = prediction[mask] + band["q_high"]
        lower = limit
    # A last band that was merged away still has to cover larger predictions.
    remaining = prediction > lower
    if remaining.any():
        last = interval["bands"][-1]
        low[remaining] = prediction[remaining] + last["q_low"]
        high[remaining] = prediction[remaining] + last["q_high"]
    return np.maximum(0.0, low), high


def leave_engines_out_interval(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    engine_ids: np.ndarray,
    eligible: np.ndarray,
    fold_ids: np.ndarray,
    *,
    coverage: float,
    edges: tuple[float, ...],
    min_engines: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound each engine using an interval calibrated without that engine."""
    prediction = np.asarray(y_pred, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.float64)
    engines = np.asarray(engine_ids)
    mask = np.asarray(eligible, dtype=bool)
    folds = np.asarray(fold_ids)
    low = np.empty(len(prediction), dtype=np.float64)
    high = np.empty(len(prediction), dtype=np.float64)
    for fold in np.unique(folds):
        held_out = folds == fold
        calibrating = ~held_out & mask
        interval = fit_mondrian_interval(
            prediction[calibrating],
            truth[calibrating],
            engines[calibrating],
            coverage=coverage,
            edges=edges,
            min_engines=min_engines,
        )
        low[held_out], high[held_out] = apply_mondrian_interval(
            prediction[held_out], interval
        )
    return low, high


def fit_mondrian_cqr(
    y_pred: np.ndarray,
    quantiles: np.ndarray,
    y_true: np.ndarray,
    engine_ids: np.ndarray,
    *,
    coverage: float,
    edges: tuple[float, ...],
    min_engines: int,
) -> dict[str, Any]:
    """Fit one conformal margin per predicted-RUL band for quantile predictions."""
    prediction = np.asarray(y_pred, dtype=np.float64)
    lower = np.min(quantiles, axis=1)
    upper = np.max(quantiles, axis=1)
    truth = np.asarray(y_true, dtype=np.float64)
    engines = np.asarray(engine_ids)
    if not len(prediction):
        raise ValueError("Interval calibration requires at least one row")
    score = np.maximum(lower - truth, truth - upper)
    bands: list[dict[str, Any]] = []
    for upper_rul, mask, engine_count in _merged_bands(
        prediction, engines, edges, min_engines
    ):
        if not mask.any():
            continue
        counts = pd.Series(engines[mask]).map(pd.Series(engines[mask]).value_counts())
        level = min(1.0, coverage * (engine_count + 1) / engine_count)
        bands.append(
            {
                "upper_predicted_rul": upper_rul,
                "engines": engine_count,
                "rows": int(mask.sum()),
                "margin": _weighted_quantile(
                    score[mask],
                    1.0 / counts.to_numpy(dtype=np.float64),
                    level,
                ),
            }
        )
    return {"coverage": coverage, "bands": bands}


def apply_mondrian_cqr(
    y_pred: np.ndarray,
    quantiles: np.ndarray,
    interval: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return non-negative bounds from quantiles and per-band margins."""
    prediction = np.asarray(y_pred, dtype=np.float64)
    lower = np.min(quantiles, axis=1)
    upper = np.max(quantiles, axis=1)
    margin = np.full(len(prediction), interval["bands"][-1]["margin"], dtype=np.float64)
    previous = -math.inf
    for band in interval["bands"]:
        limit = band["upper_predicted_rul"]
        limit = math.inf if limit is None else float(limit)
        margin[(prediction > previous) & (prediction <= limit)] = band["margin"]
        previous = limit
    low = np.maximum(0.0, lower - margin)
    return low, np.maximum(low, upper + margin)


def leave_engines_out_cqr(
    y_pred: np.ndarray,
    quantiles: np.ndarray,
    y_true: np.ndarray,
    engine_ids: np.ndarray,
    eligible: np.ndarray,
    fold_ids: np.ndarray,
    *,
    coverage: float,
    edges: tuple[float, ...],
    min_engines: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound each engine using margins calibrated without that engine."""
    prediction = np.asarray(y_pred, dtype=np.float64)
    truth = np.asarray(y_true, dtype=np.float64)
    engines = np.asarray(engine_ids)
    mask = np.asarray(eligible, dtype=bool)
    folds = np.asarray(fold_ids)
    low = np.empty(len(prediction), dtype=np.float64)
    high = np.empty(len(prediction), dtype=np.float64)
    for fold in np.unique(folds):
        held_out = folds == fold
        calibrating = ~held_out & mask
        interval = fit_mondrian_cqr(
            prediction[calibrating],
            quantiles[calibrating],
            truth[calibrating],
            engines[calibrating],
            coverage=coverage,
            edges=edges,
            min_engines=min_engines,
        )
        low[held_out], high[held_out] = apply_mondrian_cqr(
            prediction[held_out], quantiles[held_out], interval
        )
    return low, high


def _band_masks(values: np.ndarray, edges: tuple[float, ...]) -> list[tuple[str, Any]]:
    masks: list[tuple[str, Any]] = []
    lower = -math.inf
    previous_label = 0.0
    for upper in (*edges, None):
        limit = math.inf if upper is None else float(upper)
        if upper is None:
            label = f">{previous_label:.0f}"
        elif lower == -math.inf:
            label = f"<={limit:.0f}"
        else:
            label = f"{previous_label + 1:.0f}-{limit:.0f}"
        masks.append((label, (values > lower) & (values <= limit)))
        lower = limit
        previous_label = limit
    return masks


def _interval_stats(contained: np.ndarray, width: np.ndarray) -> dict[str, Any]:
    if not len(contained):
        return {"rows": 0, "coverage": None, "mean_width": None}
    return {
        "rows": len(contained),
        "coverage": float(np.mean(contained)),
        "mean_width": float(np.mean(width)),
    }


def interval_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    edges: tuple[float, ...],
) -> dict[str, Any]:
    """Report one row-per-engine interval overall, by predicted and by true RUL."""
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    lower = np.asarray(low, dtype=np.float64)
    upper = np.asarray(high, dtype=np.float64)
    contained = (truth >= lower) & (truth <= upper)
    width = upper - lower
    return {
        **_interval_stats(contained, width),
        "by_predicted_rul": {
            label: _interval_stats(contained[mask], width[mask])
            for label, mask in _band_masks(prediction, edges)
        },
        "by_true_rul": {
            label: _interval_stats(
                contained[(truth >= bottom) & (truth <= (top or math.inf))],
                width[(truth >= bottom) & (truth <= (top or math.inf))],
            )
            for label, bottom, top in RUL_BANDS
        },
    }


def _median_interval_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    def median(values: list[float | None]) -> float | None:
        observed = [value for value in values if value is not None]
        return float(np.median(observed)) if observed else None

    def combine(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "median_rows": median([item["rows"] for item in items]),
            "median_coverage": median([item["coverage"] for item in items]),
            "median_mean_width": median([item["mean_width"] for item in items]),
        }

    return {
        **combine(reports),
        "by_predicted_rul": {
            label: combine([report["by_predicted_rul"][label] for report in reports])
            for label in reports[0]["by_predicted_rul"]
        },
        "by_true_rul": {
            label: combine([report["by_true_rul"][label] for report in reports])
            for label in reports[0]["by_true_rul"]
        },
    }


def summarize_interval_draws(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    plan: np.ndarray,
    *,
    edges: tuple[float, ...],
) -> dict[str, Any]:
    """Return median leave-engines-out coverage and width across draws."""
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    lower = np.asarray(low, dtype=np.float64)
    upper = np.asarray(high, dtype=np.float64)
    return _median_interval_reports(
        [
            interval_report(
                truth[rows], prediction[rows], lower[rows], upper[rows], edges=edges
            )
            for rows in plan
        ]
    )


def risk_threshold_table(
    y_true: np.ndarray,
    interval_low: np.ndarray,
    plan: np.ndarray,
    *,
    critical_threshold: float,
    thresholds: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Show the miss and false-alarm cost of each interval-low threshold."""
    truth = np.asarray(y_true, dtype=np.float64)
    low = np.asarray(interval_low, dtype=np.float64)
    table: list[dict[str, Any]] = []
    for threshold in thresholds:
        rules = [
            alert_rule_metrics(
                truth[rows],
                low[rows],
                critical_threshold=critical_threshold,
                alert_threshold=threshold,
            )
            for rows in plan
        ]
        table.append(
            {
                "threshold": float(threshold),
                "median_recall": float(np.median([rule["recall"] for rule in rules])),
                "median_precision": float(
                    np.median([rule["precision"] for rule in rules])
                ),
                "median_false_alarm_rate": float(
                    np.median([rule["false_alarm_rate"] for rule in rules])
                ),
                "median_missed": float(
                    np.median([rule["false_negative"] for rule in rules])
                ),
                "median_false_alarms": float(
                    np.median([rule["false_positive"] for rule in rules])
                ),
            }
        )
    return table


def select_risk_threshold(
    table: list[dict[str, Any]],
    *,
    recall_target: float,
) -> dict[str, Any]:
    """Pick the smallest threshold reaching the recall target."""
    ordered = sorted(table, key=lambda row: row["threshold"])
    for row in ordered:
        if row["median_recall"] >= recall_target:
            return {**row, "recall_target": recall_target, "target_met": True}
    best = max(row["median_recall"] for row in ordered)
    fallback = next(row for row in ordered if row["median_recall"] == best)
    return {**fallback, "recall_target": recall_target, "target_met": False}


def bootstrap_official_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    interval_low: np.ndarray,
    interval_high: np.ndarray,
    *,
    critical_threshold: float,
    alert_threshold: float,
    resamples: int = 2_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Return engine-level percentile bootstrap 95% intervals."""
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    low = np.asarray(interval_low, dtype=np.float64)
    high = np.asarray(interval_high, dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(truth), size=(resamples, len(truth)))

    def rates(alerted: np.ndarray) -> dict[str, np.ndarray]:
        actual = truth <= critical_threshold
        true_positive = np.sum(actual[samples] & alerted[samples], axis=1)
        critical = np.sum(actual[samples], axis=1)
        alerts = np.sum(alerted[samples], axis=1)
        false_positive = alerts - true_positive
        healthy = len(truth) - critical
        with np.errstate(divide="ignore", invalid="ignore"):
            return {
                "recall": true_positive / critical,
                "precision": true_positive / alerts,
                "false_alarm_rate": false_positive / healthy,
            }

    error = prediction - truth
    penalty = np.where(error < 0.0, np.expm1(-error / 13.0), np.expm1(error / 10.0))
    series: dict[str, np.ndarray] = {
        "mae": np.mean(np.abs(error[samples]), axis=1),
        "rmse": np.sqrt(np.mean(error[samples] ** 2, axis=1)),
        "nasa_score_motor_normalized": np.sum(penalty[samples], axis=1),
        "signed_bias": np.mean(error[samples], axis=1),
        "interval_coverage": np.mean(
            ((truth >= low) & (truth <= high))[samples], axis=1
        ),
        "interval_mean_width": np.mean((high - low)[samples], axis=1),
    }
    for name, values in rates(prediction <= critical_threshold).items():
        series[f"point_rule_{name}"] = values
    for name, values in rates(low <= alert_threshold).items():
        series[f"interval_low_rule_{name}"] = values

    def bounds(values: np.ndarray) -> list[float] | None:
        observed = values[np.isfinite(values)]
        if not len(observed):
            return None
        return [
            float(np.percentile(observed, 2.5)),
            float(np.percentile(observed, 97.5)),
        ]

    return {
        "method": "engine-level percentile bootstrap",
        "resamples": resamples,
        "seed": seed,
        "intervals_95": {name: bounds(values) for name, values in series.items()},
    }


__all__ = [
    "apply_mondrian_cqr",
    "apply_mondrian_interval",
    "bootstrap_official_metrics",
    "draw_engine_rows",
    "eligible_rows",
    "fit_mondrian_cqr",
    "fit_mondrian_interval",
    "interval_report",
    "leave_engines_out_cqr",
    "leave_engines_out_interval",
    "nasa_score",
    "risk_threshold_table",
    "select_risk_threshold",
    "summarize_draws",
    "summarize_interval_draws",
]
