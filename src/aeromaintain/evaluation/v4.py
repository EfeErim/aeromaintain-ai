"""Full-range, engine-paired v4 measurement and persistent alarm replay."""

from __future__ import annotations

import numpy as np
import pandas as pd


def point_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    """Score a draws-by-engines matrix without treating draws as engines."""
    error = prediction - truth
    critical = truth <= 30
    long = truth > 150
    misses = critical & (prediction > 30)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "nasa_per_engine": float(
            np.mean(np.expm1(np.where(error < 0, -error / 13, error / 10)))
        ),
        "bias": float(np.mean(error)),
        "long_rmse": float(np.sqrt(np.mean(error[long] ** 2))) if long.any() else None,
        "long_bias": float(np.mean(error[long])) if long.any() else None,
        "critical_miss_rate": float(misses.sum() / critical.sum())
        if critical.any()
        else None,
        "mean_critical_misses_per_draw": float(misses.sum(axis=1).mean()),
        "engines": truth.shape[1],
        "draws": truth.shape[0],
        "bands": {
            label: {
                "sampled_cuts": int(mask.sum()),
                "engines": int(mask.any(axis=0).sum()),
            }
            for label, mask in [
                ("<=30", critical),
                ("31-150", (truth > 30) & ~long),
                (">150", long),
            ]
        },
    }


def paired_bootstrap(truth, baseline, candidate, *, resamples=2000, seed=44):
    """Resample engine columns, keeping paired predictions and every engine cut."""

    # Sufficient statistics reduce 2000 resamples without materializing huge tensors.
    def stats(prediction):
        error = prediction - truth
        long = truth > 150
        critical = truth <= 30
        return np.array(
            [
                np.mean(error**2, axis=0),
                np.mean(np.abs(error), axis=0),
                np.mean(np.expm1(np.where(error < 0, -error / 13, error / 10)), axis=0),
                np.sum(np.where(long, error**2, 0), axis=0),
                long.sum(axis=0),
                ((prediction > 30) & critical).sum(axis=0),
                critical.sum(axis=0),
            ]
        )

    rng = np.random.default_rng(seed)
    columns = rng.integers(0, truth.shape[1], size=(resamples, truth.shape[1]))

    def aggregate(values):
        selected = values[:, columns].mean(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.array(
                [
                    np.sqrt(selected[0]),
                    selected[1],
                    selected[2],
                    np.sqrt(selected[3] / selected[4]),
                    selected[5] / selected[6],
                ]
            )

    differences = aggregate(stats(candidate)) - aggregate(stats(baseline))
    return {
        "method": "paired engine-column bootstrap; candidate minus reference",
        "resamples": resamples,
        "seed": seed,
        "intervals_95": {
            name: np.nanpercentile(values, [2.5, 97.5]).tolist()
            for name, values in zip(
                ["rmse", "mae", "nasa_per_engine", "long_rmse", "critical_miss_rate"],
                differences,
                strict=True,
            )
        },
    }


def acceptance(reference: dict, candidate: dict, tolerance: float = 0.02) -> dict:
    """Predeclared point-estimate gates; missing bands cannot pass."""
    checks = {
        "long_rmse_improves": candidate["long_rmse"] is not None
        and reference["long_rmse"] is not None
        and candidate["long_rmse"] < reference["long_rmse"],
        "overall_rmse_not_worse": candidate["rmse"] <= reference["rmse"],
        "nasa_not_worse": candidate["nasa_per_engine"] <= reference["nasa_per_engine"],
        "critical_misses_acceptable": candidate["critical_miss_rate"] is not None
        and reference["critical_miss_rate"] is not None
        and candidate["critical_miss_rate"]
        <= reference["critical_miss_rate"] + tolerance,
    }
    return {"accepted": all(checks.values()), "checks": checks}


def interval_metrics(truth, low, high, engines, *, alpha=0.1):
    """Interval score and tails for sampled cuts, with distinct-engine support."""
    if not len(truth):
        return {"cuts": 0, "engines": 0, "small_group": True}
    below, above = truth < low, truth > high
    width = high - low
    score = width + 2 / alpha * (
        np.maximum(low - truth, 0) + np.maximum(truth - high, 0)
    )
    count = len(np.unique(engines))
    return {
        "cuts": len(truth),
        "engines": count,
        "small_group": count < 20,
        "coverage": float(np.mean(~(below | above))),
        "mean_width": float(width.mean()),
        "interval_score": float(score.mean()),
        "below_lower": int(below.sum()),
        "above_upper": int(above.sum()),
    }


def replay_alarms(
    frame: pd.DataFrame,
    values: np.ndarray,
    *,
    persistence=3,
    threshold=30,
    min_rul=0,
):
    """Confirm on the persistence-th cycle; merge contiguous alert runs only.

    An engine counts as detected only while a confirmed alarm is active at a cycle
    whose true RUL lies in ``[min_rul, threshold]``; v4 used ``min_rul=0``.
    """
    if persistence < 1 or len(frame) != len(values):
        raise ValueError("Invalid replay persistence or prediction length")
    data = frame[["unit_id", "cycle", "rul_true"]].copy()
    data["value"] = values
    records = []
    for unit, history in data.groupby("unit_id", sort=True):
        history = history.sort_values("cycle")
        streak, false_events, first_lead = 0, 0, None
        detected = False
        previous = None
        for row in history.itertuples():
            if previous is not None and row.cycle != previous + 1:
                streak = 0
            streak = streak + 1 if row.value <= threshold else 0
            previous = row.cycle
            if streak == persistence:
                if first_lead is None:
                    first_lead = float(row.rul_true)
                false_events += int(row.rul_true > threshold)
            if streak >= persistence and min_rul <= row.rul_true <= threshold:
                detected = True
        critical_observed = bool((history.rul_true <= threshold).any())
        records.append(
            {
                "unit_id": int(unit),
                "first_confirmed_lead": first_lead,
                "false_events": false_events,
                "critical_observed": critical_observed,
                "missed": critical_observed and not detected,
            }
        )
    leads = [
        r["first_confirmed_lead"]
        for r in records
        if r["first_confirmed_lead"] is not None
    ]
    return {
        "engines": len(records),
        "missed_engines": sum(r["missed"] for r in records),
        "false_events_per_engine": sum(r["false_events"] for r in records)
        / len(records),
        "median_first_confirmed_lead": float(np.median(leads)) if leads else None,
        "engines_with_alarm": len(leads),
        "records": records,
    }
