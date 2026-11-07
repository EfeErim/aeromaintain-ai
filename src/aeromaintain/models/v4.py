"""Reproducible v4 development experiment; never reads official test labels.

Run with ``python -m aeromaintain.models.v4 --project-root .``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from aeromaintain.data.pipeline import sha256_file
from aeromaintain.evaluation.protocol import (
    apply_mondrian_interval,
    draw_engine_rows,
    fit_mondrian_interval,
)
from aeromaintain.evaluation.v4 import (
    acceptance,
    interval_metrics,
    paired_bootstrap,
    point_metrics,
    replay_alarms,
)
from aeromaintain.features.causal import (
    CONDITION_FEATURE,
    FEATURE_SETS,
    ConditionNormalizer,
    FoldPreprocessor,
    build_causal_features,
    build_condition_features,
)
from aeromaintain.models.rul import (
    ModelingConfig,
    _new_xgboost,
    engine_equal_weights,
    group_fold_plan,
    inner_engine_split,
)

CANDIDATES = {
    "reference": {"family": "xgboost", "cap": 125, "condition": True},
    "xgboost_normalized": {"family": "xgboost", "cap": None, "condition": True},
    "xgboost_raw": {"family": "xgboost", "cap": None, "condition": False},
    "ridge_normalized": {"family": "ridge", "cap": None, "condition": True},
    "ridge_raw": {"family": "ridge", "cap": None, "condition": False},
}


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def split_engines(frame, seed=42):
    engines = np.random.default_rng(seed).permutation(np.sort(frame.unit_id.unique()))
    first, second = int(0.6 * len(engines)), int(0.8 * len(engines))
    return {
        name: sorted(ids.tolist())
        for name, ids in zip(
            ["selection", "calibration", "evaluation"],
            [engines[:first], engines[first:second], engines[second:]],
            strict=True,
        )
    }


def feature_partition(frame, fit_rows, spec):
    normalizer = None
    if spec["condition"]:
        normalizer = ConditionNormalizer(conditions=6, seed=42).fit(
            frame.iloc[fit_rows]
        )
        features = build_condition_features(frame, normalizer)
    else:
        features = build_causal_features(frame)
    columns = FEATURE_SETS["compact"]
    if spec["condition"]:
        columns = (*columns, CONDITION_FEATURE)
    features = features.loc[:, list(columns)]
    prep = FoldPreprocessor(scale=spec["family"] == "ridge", input_features=columns)
    prep.fit(features.iloc[fit_rows])
    return prep.transform(features), prep, normalizer


def fit_partition(frame, train_rows, spec, parameters, *, seed=42, tree_count=None):
    target = frame.rul_true.to_numpy(dtype=float)
    if spec["cap"] is not None:
        target = np.minimum(target, spec["cap"])
    config = ModelingConfig(xgboost_n_jobs=6)
    if spec["family"] == "xgboost" and tree_count is None:
        inner_fit, inner_stop = inner_engine_split(
            frame.iloc[train_rows].unit_id.to_numpy(), fraction=0.2, seed=seed
        )
        fit_rows, stop_rows = train_rows[inner_fit], train_rows[inner_stop]
        matrix, _, _ = feature_partition(frame, fit_rows, spec)
        model = _new_xgboost(
            parameters, config, seed=seed, n_estimators=1500, early_stopping=True
        )
        model.fit(
            matrix[fit_rows],
            target[fit_rows],
            sample_weight=engine_equal_weights(frame.iloc[fit_rows].unit_id.to_numpy()),
            eval_set=[(matrix[stop_rows], target[stop_rows])],
            sample_weight_eval_set=[
                engine_equal_weights(frame.iloc[stop_rows].unit_id.to_numpy())
            ],
            verbose=False,
        )
        tree_count = int(model.best_iteration) + 1
    matrix, prep, normalizer = feature_partition(frame, train_rows, spec)
    model = (
        Ridge(alpha=1)
        if spec["family"] == "ridge"
        else _new_xgboost(
            parameters, config, seed=seed, n_estimators=tree_count, early_stopping=False
        )
    )
    model.fit(
        matrix[train_rows],
        target[train_rows],
        sample_weight=engine_equal_weights(frame.iloc[train_rows].unit_id.to_numpy()),
    )
    prediction = np.maximum(0, model.predict(matrix))
    return (
        prediction,
        {"model": model, "preprocessor": prep, "normalizer": normalizer, "spec": spec},
        tree_count,
    )


def selection_run(root, output, dataset, config):
    frame = pd.read_parquet(root / "data/processed" / dataset.lower() / "train.parquet")
    partitions = split_engines(frame, config["seed"])
    selected = frame[frame.unit_id.isin(partitions["selection"])].reset_index(drop=True)
    plan = draw_engine_rows(
        selected.unit_id.to_numpy(),
        np.ones(len(selected), dtype=bool),
        draws=config["draws"],
        seed=43,
    )
    folds = group_fold_plan(
        selected.unit_id.to_numpy(), folds=config["folds"], seed=config["seed"]
    )
    destination = output / dataset
    destination.mkdir()
    write_json(destination / "partitions.json", partitions)
    records, predictions = {}, {}
    for name, spec in CANDIDATES.items():
        prediction = np.zeros(len(selected))
        counts = []
        for fold, (train_rows, valid_rows) in enumerate(folds):
            print(f"{dataset} {name} fold {fold + 1}/{len(folds)}", flush=True)
            estimated, _, count = fit_partition(
                selected, train_rows, spec, config["xgboost"], seed=42 + fold
            )
            prediction[valid_rows] = estimated[valid_rows]
            if count is not None:
                counts.append(count)
        predictions[name] = prediction
        records[name] = {
            "spec": spec,
            "tree_counts": counts,
            "metrics": point_metrics(
                selected.rul_true.to_numpy()[plan], prediction[plan]
            ),
        }
        if name != "reference":
            records[name]["paired_comparison"] = paired_bootstrap(
                selected.rul_true.to_numpy()[plan],
                predictions["reference"][plan],
                prediction[plan],
                resamples=config["bootstrap_resamples"],
            )
            records[name]["gate"] = acceptance(
                records["reference"]["metrics"],
                records[name]["metrics"],
                config["critical_miss_tolerance"],
            )
        write_json(destination / "selection.json", records)
    evidence = selected[["unit_id", "cycle", "rul_true"]].copy()
    for name, values in predictions.items():
        evidence[name] = values
    evidence.to_parquet(destination / "oof.parquet", index=False)
    np.save(destination / "cuts.npy", plan)
    return records


def evaluate_intervals(root, output, dataset, chosen, record, config):
    destination = output / dataset
    partitions = json.loads((destination / "partitions.json").read_text())
    frame = pd.read_parquet(root / "data/processed" / dataset.lower() / "train.parquet")
    rows = np.flatnonzero(frame.unit_id.isin(partitions["selection"]))
    counts = record["tree_counts"]
    count = int(np.median(counts)) if counts else None
    prediction, bundle, _ = fit_partition(
        frame, rows, CANDIDATES[chosen], config["xgboost"], tree_count=count
    )
    joblib.dump(bundle, destination / "model.joblib")
    cal = frame.unit_id.isin(partitions["calibration"]).to_numpy()
    ev = frame.unit_id.isin(partitions["evaluation"]).to_numpy()
    truth = frame.rul_true.to_numpy()
    engines = frame.unit_id.to_numpy()
    mondrian = fit_mondrian_interval(
        prediction[cal],
        truth[cal],
        engines[cal],
        coverage=0.9,
        edges=(30, 60, 100, 150),
        min_engines=20,
    )
    cal_rows = draw_engine_rows(engines, cal, draws=1, seed=45)[0]
    residual = np.abs(truth[cal_rows] - prediction[cal_rows])
    rank = math.ceil((len(residual) + 1) * 0.9)
    if rank > len(residual):
        raise ValueError("Too few calibration engines for finite 90% interval")
    margin = float(np.sort(residual)[rank - 1])
    write_json(
        destination / "model_lock.json",
        {
            "dataset": dataset,
            "candidate": chosen,
            "tree_count": count,
            "model_sha256": sha256_file(destination / "model.joblib"),
            "protocol_sha256": sha256_file(root / "docs/protocol_v4.md"),
            "config_sha256": sha256_file(root / "configs/protocol_v4.json"),
            "data_sha256": sha256_file(
                root / "data/processed" / dataset.lower() / "train.parquet"
            ),
            "partitions": partitions,
            "mondrian": mondrian,
            "split_margin": margin,
            "scope": "v4 held-out development engines; previously used under v3",
        },
    )
    cuts = draw_engine_rows(engines, ev, draws=config["draws"], seed=46).ravel()
    # A separate train-only normalizer supplies operating-condition group labels.
    condition_model = ConditionNormalizer(conditions=6, seed=42).fit(frame.iloc[rows])
    conditions = condition_model.labels(frame)
    report = {
        "point": point_metrics(
            truth[cuts].reshape(config["draws"], -1),
            prediction[cuts].reshape(config["draws"], -1),
        ),
        "intervals": {},
    }
    intervals = {
        "mondrian": apply_mondrian_interval(prediction, mondrian),
        "split_conformal": (np.maximum(0, prediction - margin), prediction + margin),
    }
    replay_frame = frame.loc[ev].reset_index(drop=True)
    point_replay = replay_alarms(replay_frame, prediction[ev])
    for name, (low, high) in intervals.items():
        sampled_truth = truth[cuts]
        groups = {
            "overall": np.ones(len(cuts), dtype=bool),
            "rul_<=30": sampled_truth <= 30,
            "rul_31-150": (sampled_truth > 30) & (sampled_truth <= 150),
            "rul_>150": sampled_truth > 150,
        }
        groups.update(
            {
                f"condition_{condition}": conditions[cuts] == condition
                for condition in np.unique(conditions[cuts])
            }
        )
        report["intervals"][name] = {
            "groups": {
                label: interval_metrics(
                    truth[cuts][mask],
                    low[cuts][mask],
                    high[cuts][mask],
                    engines[cuts][mask],
                )
                for label, mask in groups.items()
            },
            "point_alarm": point_replay,
            "lower_bound_alarm": replay_alarms(replay_frame, low[ev]),
        }
    evidence = frame.loc[ev, ["unit_id", "cycle", "rul_true"]].copy()
    evidence["prediction"] = prediction[ev]
    for name, (low, high) in intervals.items():
        evidence[f"{name}_low"] = low[ev]
        evidence[f"{name}_high"] = high[ev]
    evidence.to_parquet(destination / "evaluation.parquet", index=False)
    write_json(destination / "evaluation.json", report)
    return report


def run(root: Path, output: Path):
    config = json.loads((root / "configs/protocol_v4.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "protocol_sha256": sha256_file(root / "docs/protocol_v4.md"),
        "config_sha256": sha256_file(root / "configs/protocol_v4.json"),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(output / "provenance.json", provenance)
    selections = {
        dataset: selection_run(root, output, dataset, config)
        for dataset in config["datasets"]
    }
    eligible = [
        name
        for name in CANDIDATES
        if name != "reference"
        and all(records[name]["gate"]["accepted"] for records in selections.values())
    ]
    chosen = (
        min(
            eligible,
            key=lambda name: np.mean(
                [
                    records[name]["metrics"]["rmse"]
                    / records["reference"]["metrics"]["rmse"]
                    for records in selections.values()
                ]
            ),
        )
        if eligible
        else "reference"
    )
    write_json(
        output / "decision.json",
        {
            "chosen": chosen,
            "eligible": eligible,
            "reason": "All subset gates passed"
            if eligible
            else "No candidate passed both subsets; retain v3 specification",
        },
    )
    evaluations = {
        dataset: evaluate_intervals(
            root, output, dataset, chosen, selections[dataset][chosen], config
        )
        for dataset in config["datasets"]
    }
    write_json(
        output / "evidence.json",
        {
            "provenance": provenance,
            "selection": selections,
            "chosen": chosen,
            "evaluation": evaluations,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("runs/v4"))
    args = parser.parse_args()
    run(args.project_root.resolve(), args.output.resolve())
