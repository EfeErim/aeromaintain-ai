"""One-sided and per-condition interval coverage on the v3 development engines.

Development engines only: reads each run's stored leave-engines-out cross-fit
predictions over its training engines. No test label is read, no model is fitted, and
no lock or published row changes.

For each v3 run it rebuilds the leave-engines-out Mondrian interval with the frozen
protocol functions and the settings its lock recorded, checks that the rebuilt median
coverage and width equal the stored ``calibration_analysis.json`` exactly, then reports:

- one-sided coverage: the share of engines above the lower bound and below the upper
  bound, and the share predicted late (``prediction > rul_true``), which the NASA score
  penalizes more than early predictions; the same on truly critical engines;
- on the six-condition subsets, coverage per operating condition, and a variant that
  calibrates the Mondrian bands separately within each condition, at the same nominal
  coverage and on the same engine draws.

Every figure is a median over the seeded draws of one eligible row per engine that the
locked calibration used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aeromaintain.artifacts import verify_lock
from aeromaintain.evaluation.protocol import (
    draw_engine_rows,
    eligible_rows,
    leave_engines_out_interval,
    summarize_interval_draws,
)
from aeromaintain.features import ConditionNormalizer
from aeromaintain.models.robustness import config_from_lock

RUNS = {
    "FD001": "fd001-v3",
    "FD002": "fd002-v3-clean",
    "FD003": "fd003-v3",
    "FD004": "fd004-v3-clean",
}


def _spread(values: list[float]) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def one_sided(
    truth: np.ndarray,
    prediction: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    plan: np.ndarray,
    critical: float,
) -> dict[str, Any]:
    """Per-draw one-sided coverage and late-prediction rates, summarized over draws."""
    keys = (
        "two_sided_coverage",
        "lower_bound_coverage",
        "upper_bound_coverage",
        "late_rate",
        "critical_lower_bound_coverage",
        "critical_late_rate",
    )
    draws: dict[str, list[float]] = {key: [] for key in keys}
    critical_engines = []
    for rows in plan:
        t, p, lo, hi = truth[rows], prediction[rows], low[rows], high[rows]
        draws["two_sided_coverage"].append(float(np.mean((t >= lo) & (t <= hi))))
        draws["lower_bound_coverage"].append(float(np.mean(t >= lo)))
        draws["upper_bound_coverage"].append(float(np.mean(t <= hi)))
        draws["late_rate"].append(float(np.mean(p > t)))
        near = t <= critical
        critical_engines.append(int(near.sum()))
        t, p, lo = t[near], p[near], lo[near]
        draws["critical_lower_bound_coverage"].append(float(np.mean(t >= lo)))
        draws["critical_late_rate"].append(float(np.mean(p > t)))
    return {
        **{key: _spread(values) for key, values in draws.items()},
        "critical_engines_per_draw": _spread(critical_engines),
    }


def by_condition(
    truth: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    plan: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Median coverage and width per operating condition over the draws."""
    result: dict[str, Any] = {}
    for condition in np.unique(labels):
        coverage, width, rows_per_draw = [], [], []
        for rows in plan:
            picked = rows[labels[rows] == condition]
            t, lo, hi = truth[picked], low[picked], high[picked]
            coverage.append(float(np.mean((t >= lo) & (t <= hi))))
            width.append(float(np.mean(hi - lo)))
            rows_per_draw.append(len(picked))
        result[str(int(condition))] = {
            "median_coverage": float(np.median(coverage)),
            "median_mean_width": float(np.median(width)),
            "median_rows": float(np.median(rows_per_draw)),
        }
    medians = [entry["median_coverage"] for entry in result.values()]
    return {
        "conditions": result,
        "coverage_range": [float(min(medians)), float(max(medians))],
    }


def evaluate_run(root: Path, run_id: str) -> dict[str, Any]:
    verified = verify_lock(root, run_id)
    lock = verified.raw_lock
    config = config_from_lock(lock)
    stored = json.loads(
        (verified.run_dir / "calibration_analysis.json").read_text(encoding="utf-8")
    )
    if stored["adopted"] != "mondrian":
        raise ValueError(f"{run_id} did not adopt the Mondrian interval")

    cross_fit = pd.read_parquet(verified.run_dir / "cross_fit_predictions.parquet")
    truth = cross_fit["rul_true"].to_numpy(dtype=np.float64)
    prediction = cross_fit["prediction"].to_numpy(dtype=np.float64)
    engines = cross_fit["unit_id"].to_numpy()
    folds = cross_fit["fold"].to_numpy()
    eligible = eligible_rows(truth, config.scoring_rul_window)
    arguments = {
        "coverage": config.nominal_interval_coverage,
        "edges": config.interval_band_edges,
        "min_engines": config.interval_min_band_engines,
    }
    # calibrate_interval draws with seed + 1, where seed is the lock's seed.
    plan = draw_engine_rows(
        engines, eligible, draws=config.scoring_draws, seed=int(lock["seed"]) + 1
    )

    low, high = leave_engines_out_interval(
        prediction, truth, engines, eligible, folds, **arguments
    )
    rebuilt = summarize_interval_draws(
        truth, prediction, low, high, plan, edges=config.interval_band_edges
    )
    for key in ("median_coverage", "median_mean_width"):
        if rebuilt[key] != stored["mondrian"][key]:
            raise ValueError(
                f"{run_id}: rebuilt {key} {rebuilt[key]} != stored "
                f"{stored['mondrian'][key]}"
            )

    critical = float(config.critical_rul_threshold)
    entry: dict[str, Any] = {
        "run_id": run_id,
        "engines": int(pd.Series(engines).nunique()),
        "draws": len(plan),
        "nominal_coverage": config.nominal_interval_coverage,
        "rul_window": config.scoring_rul_window,
        "critical_rul_threshold": critical,
        "stored_check": {
            key: rebuilt[key] for key in ("median_coverage", "median_mean_width")
        },
        "mondrian": one_sided(truth, prediction, low, high, plan, critical),
    }
    if config.operating_conditions == 1:
        return entry

    normalizer = joblib.load(verified.run_dir / "condition_normalizer.joblib")
    if not isinstance(normalizer, ConditionNormalizer):
        raise ValueError(f"{run_id}: condition normalizer has the wrong type")
    train = pd.read_parquet(
        root / "data" / "processed" / str(lock["dataset"]).lower() / "train.parquet"
    ).reset_index(drop=True)
    if not (
        np.array_equal(train["unit_id"].to_numpy(), engines)
        and np.array_equal(train["cycle"].to_numpy(), cross_fit["cycle"].to_numpy())
    ):
        raise ValueError(f"{run_id}: cross-fit rows do not follow train.parquet")
    labels = np.asarray(normalizer.labels(train))

    conditional_low = np.empty_like(low)
    conditional_high = np.empty_like(high)
    for condition in np.unique(labels):
        rows = labels == condition
        conditional_low[rows], conditional_high[rows] = leave_engines_out_interval(
            prediction[rows],
            truth[rows],
            engines[rows],
            eligible[rows],
            folds[rows],
            **arguments,
        )
    conditional = summarize_interval_draws(
        truth,
        prediction,
        conditional_low,
        conditional_high,
        plan,
        edges=config.interval_band_edges,
    )
    entry["by_condition"] = {
        "mondrian": by_condition(truth, low, high, plan, labels),
        "condition_conditional": {
            "median_coverage": conditional["median_coverage"],
            "median_mean_width": conditional["median_mean_width"],
            **by_condition(truth, conditional_low, conditional_high, plan, labels),
            "one_sided": one_sided(
                truth, prediction, conditional_low, conditional_high, plan, critical
            ),
        },
    }
    return entry


def evaluate(root: Path) -> dict[str, Any]:
    return {
        "scope": (
            "Development engines only: leave-engines-out cross-fit predictions over "
            "training engines. No test label is read."
        ),
        "datasets": {
            dataset: evaluate_run(root, run_id) for dataset, run_id in RUNS.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path())
    parser.add_argument(
        "--output", type=Path, default=Path("docs/development_coverage.json")
    )
    args = parser.parse_args()
    evidence = evaluate(args.root)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for dataset, entry in evidence["datasets"].items():
        mondrian = entry["mondrian"]
        print(
            dataset,
            "two-sided",
            f"{mondrian['two_sided_coverage']['median']:.3f}",
            "lower",
            f"{mondrian['lower_bound_coverage']['median']:.3f}",
            "upper",
            f"{mondrian['upper_bound_coverage']['median']:.3f}",
            "late",
            f"{mondrian['late_rate']['median']:.3f}",
        )


if __name__ == "__main__":
    main()
