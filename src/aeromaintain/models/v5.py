"""Protocol v5: hybrid development check and the final N-CMAPSS clean test.

Run ``python -m aeromaintain.models.v5 {hybrid,prepare,evaluate} --output DIR``.
``prepare`` never opens a final test file. ``evaluate`` refuses to run unless the lock
written by ``prepare`` is committed unchanged in ``HEAD`` and the tree is clean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeromaintain.data.ncmapss import cycle_features, read_multiple_targets
from aeromaintain.data.pipeline import sha256_file
from aeromaintain.evaluation.ledger import claim_holdout
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
from aeromaintain.models.rul import (
    ModelingConfig,
    _new_xgboost,
    engine_equal_weights,
    group_fold_plan,
    inner_engine_split,
)
from aeromaintain.models.v4 import write_json

PROTOCOL = "docs/protocol_v5.md"
CONFIG = "configs/protocol_v5.json"
RESERVATION = "docs/ncmapss_locks/final_holdout_reservation.json"
PUBLISHED_LOCK = "docs/ncmapss_locks/final_v5.json"
DEVELOPMENT = "docs/protocol_v5_development.json"
LEDGER = "runs/ncmapss_holdout_ledger.json"
LEDGER_KEY = "FINAL-v5"
SOURCES = (
    "src/aeromaintain/models/v5.py",
    "src/aeromaintain/models/v4.py",
    "src/aeromaintain/models/rul.py",
    "src/aeromaintain/data/ncmapss.py",
    "src/aeromaintain/evaluation/v4.py",
    "src/aeromaintain/evaluation/protocol.py",
    "src/aeromaintain/evaluation/ledger.py",
)
CANDIDATES = ("ridge", "xgboost")
TRAINING_ENGINES = 64
TEST_ENGINES = 35
TEST_NAMESPACE = 10


def source_hashes(root: Path) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in SOURCES}


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, check=True
    ).stdout


def require_clean_tree(root: Path) -> str:
    """Return HEAD, refusing to run with any uncommitted or untracked change."""
    if _git(root, "status", "--porcelain").strip():
        raise RuntimeError("v5 runs only from a clean, committed working tree")
    return _git(root, "rev-parse", "HEAD").decode().strip()


# Part A ------------------------------------------------------------------------------


def hybrid_prediction(reference, uncapped, switch_below):
    """Capped prediction below the switch, uncapped prediction from it upward."""
    reference = np.asarray(reference, dtype=float)
    return np.where(reference < switch_below, reference, np.asarray(uncapped))


def run_hybrid(root: Path, output: Path, config: dict) -> dict:
    settings = config["hybrid"]
    output.mkdir(parents=True, exist_ok=False)
    datasets = {}
    for dataset in settings["datasets"]:
        folder = root / "runs/v4" / dataset
        frame = pd.read_parquet(folder / "oof.parquet")
        plan = np.load(folder / "cuts.npy")
        truth = frame.rul_true.to_numpy(dtype=float)[plan]
        reference = frame.reference.to_numpy(dtype=float)
        uncapped = frame.xgboost_normalized.to_numpy(dtype=float)
        hybrid = hybrid_prediction(reference, uncapped, settings["switch_below"])
        metrics = {
            name: point_metrics(truth, values[plan])
            for name, values in [
                ("reference", reference),
                ("uncapped", uncapped),
                ("hybrid", hybrid),
            ]
        }
        datasets[dataset] = {
            "inputs_sha256": {
                "oof.parquet": sha256_file(folder / "oof.parquet"),
                "cuts.npy": sha256_file(folder / "cuts.npy"),
            },
            "switched_cut_fraction": float(
                np.mean(reference[plan] >= settings["switch_below"])
            ),
            "metrics": metrics,
            "paired_comparison": paired_bootstrap(
                truth,
                reference[plan],
                hybrid[plan],
                resamples=config["bootstrap_resamples"],
            ),
            "gate": acceptance(
                metrics["reference"],
                metrics["hybrid"],
                settings["critical_miss_tolerance"],
            ),
        }
    report = {
        "status": "development evidence only; rule designed after v4 results",
        "rule": f"reference if reference < {settings['switch_below']}, "
        "else xgboost_normalized",
        "protocol_sha256": sha256_file(root / PROTOCOL),
        "datasets": datasets,
        "accepted_on_all": all(d["gate"]["accepted"] for d in datasets.values()),
    }
    write_json(output / "hybrid.json", report)
    return report


# Part B ------------------------------------------------------------------------------


def load_members(paths: list[Path], first_index: int) -> pd.DataFrame:
    frames = []
    for offset, path in enumerate(paths):
        print(f"Summarizing measured cycles: {path.name}", flush=True)
        member = cycle_features(path)
        member["source_file"] = path.name
        member["archive_unit_id"] = member.unit_id
        member["unit_id"] = 1000 * (first_index + offset) + member.unit_id
        frames.append(member)
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values(["unit_id", "cycle"]).reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = [c for c in frame.columns if c.startswith(("sensor_", "setting_"))]
    if len(columns) != 72:
        raise ValueError("v5 requires exactly 72 measured cycle summaries")
    return columns


def fit_candidate(name, x, y, engines, rows, params, *, seed, tree_count=None):
    """Fit one frozen candidate on ``rows``; returns the model and its tree count."""
    weights = engine_equal_weights(engines[rows])
    if name == "ridge":
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1)
        )
        model.fit(x[rows], y[rows], ridge__sample_weight=weights)
        return model, None
    if name != "xgboost":
        raise ValueError(f"Unknown v5 candidate {name}")
    config = ModelingConfig(xgboost_n_jobs=6)
    if tree_count is None:
        inner_fit, inner_stop = inner_engine_split(
            engines[rows], fraction=0.2, seed=seed
        )
        fit_rows, stop_rows = rows[inner_fit], rows[inner_stop]
        probe = _new_xgboost(
            params, config, seed=seed, n_estimators=1500, early_stopping=True
        )
        probe.fit(
            x[fit_rows],
            y[fit_rows],
            sample_weight=engine_equal_weights(engines[fit_rows]),
            eval_set=[(x[stop_rows], y[stop_rows])],
            sample_weight_eval_set=[engine_equal_weights(engines[stop_rows])],
            verbose=False,
        )
        tree_count = int(probe.best_iteration) + 1
    model = _new_xgboost(
        params, config, seed=seed, n_estimators=tree_count, early_stopping=False
    )
    model.fit(x[rows], y[rows], sample_weight=weights)
    return model, tree_count


def predict(model, x) -> np.ndarray:
    return np.maximum(0.0, model.predict(x))


def split_margin(truth, prediction, coverage):
    rank = math.ceil((len(truth) + 1) * coverage)
    if rank > len(truth):
        raise ValueError("Too few calibration engines for a finite split interval")
    return float(np.sort(np.abs(truth - prediction))[rank - 1]), rank


def reserved_test_engines(reservation: dict, test_files: list[str]) -> dict:
    """Namespaced reserved engines per test file, split by archive suffix."""
    result = {}
    for offset, name in enumerate(test_files):
        entry = reservation["files"][name]
        base = 1000 * (TEST_NAMESPACE + offset)
        result[name] = {
            "dev": [base + unit for unit in entry["archive_dev_units"]],
            "test": [base + unit for unit in entry["archive_test_units"]],
        }
    return result


def summarize_replay(replay: dict) -> dict:
    critical = sum(r["critical_observed"] for r in replay["records"])
    summary = {k: v for k, v in replay.items() if k != "records"}
    summary["critical_engines"] = critical
    summary["missed_fraction"] = (
        replay["missed_engines"] / critical if critical else None
    )
    return summary


def prepare(root: Path, output: Path, config: dict) -> dict:
    commit = require_clean_tree(root)
    output.mkdir(parents=True, exist_ok=False)
    raw = root / config["raw_dir"]
    reservation = json.loads((root / RESERVATION).read_text(encoding="utf-8"))
    paths = [raw / name for name in config["training_files"]]
    if set(config["training_files"]) & set(config["test_files"]):
        raise ValueError("Training and final test files overlap")
    frame = load_members(paths, 0)
    if frame.unit_id.nunique() != TRAINING_ENGINES:
        raise ValueError("v5 training pool must contain 64 engines")
    columns = feature_columns(frame)
    y = read_multiple_targets(paths, frame)
    x = frame[columns].to_numpy(dtype=float)
    engines = frame.unit_id.to_numpy()
    every = np.ones(len(frame), dtype=bool)
    folds = group_fold_plan(engines, folds=config["folds"], seed=config["seed"])
    plan = draw_engine_rows(engines, every, draws=config["draws"], seed=43)
    oof, counts = {}, {}
    for name in CANDIDATES:
        prediction = np.full(len(frame), np.nan)
        counts[name] = []
        for fold, (train_rows, valid_rows) in enumerate(folds):
            print(f"v5 {name} fold {fold + 1}/{len(folds)}", flush=True)
            model, count = fit_candidate(
                name, x, y, engines, train_rows, config["xgboost"], seed=42 + fold
            )
            prediction[valid_rows] = predict(model, x[valid_rows])
            if count is not None:
                counts[name].append(count)
        if np.isnan(prediction).any():
            raise RuntimeError("Every training row needs an out-of-fold prediction")
        oof[name] = prediction
    baseline = float(np.average(y, weights=engine_equal_weights(engines)))
    selection = {name: point_metrics(y[plan], oof[name][plan]) for name in CANDIDATES}
    selection["baseline"] = point_metrics(y[plan], np.full(plan.shape, baseline))
    chosen = (
        "ridge"
        if selection["ridge"]["rmse"] <= selection["xgboost"]["rmse"]
        else "xgboost"
    )
    tree_count = int(np.median(counts["xgboost"])) if chosen == "xgboost" else None
    mondrian = fit_mondrian_interval(
        oof[chosen],
        y,
        engines,
        coverage=config["coverage"],
        edges=tuple(config["mondrian_edges"]),
        min_engines=config["mondrian_min_engines"],
    )
    calibration_rows = draw_engine_rows(engines, every, draws=1, seed=45)[0]
    margin, rank = split_margin(
        y[calibration_rows], oof[chosen][calibration_rows], config["coverage"]
    )
    final, _ = fit_candidate(
        chosen,
        x,
        y,
        engines,
        np.arange(len(frame)),
        config["xgboost"],
        seed=config["seed"],
        tree_count=tree_count,
    )
    joblib.dump(final, output / "model.joblib")
    replay_frame = frame[["unit_id", "cycle"]].assign(rul_true=y)
    development = {
        "status": "development evidence; out-of-fold on previously opened engines",
        "selection": selection,
        "tree_counts": counts,
        "chosen": chosen,
        "oof_point_alarm": summarize_replay(
            replay_alarms(
                replay_frame,
                oof[chosen],
                persistence=config["persistence_cycles"],
                threshold=config["alarm_threshold"],
                min_rul=config["timely_min_rul"],
            )
        ),
    }
    lock = {
        "scope": "protocol v5 final N-CMAPSS clean test",
        "git_commit": commit,
        "protocol_sha256": sha256_file(root / PROTOCOL),
        "config_sha256": sha256_file(root / CONFIG),
        "reservation_sha256": sha256_file(root / RESERVATION),
        "source_sha256": source_hashes(root),
        "training_data_sha256": {path.name: sha256_file(path) for path in paths},
        "test_data_sha256": {
            name: reservation["files"][name]["sha256"] for name in config["test_files"]
        },
        "training_engines": sorted(int(e) for e in np.unique(engines)),
        "test_engines": reserved_test_engines(reservation, config["test_files"]),
        "features": columns,
        "chosen": chosen,
        "tree_count": tree_count,
        "model_sha256": sha256_file(output / "model.joblib"),
        "baseline": baseline,
        "mondrian": mondrian,
        "split_margin": margin,
        "split_rank": rank,
        "gates": config["gates"],
        "development_sha256": None,
    }
    write_json(output / "development.json", development)
    lock["development_sha256"] = sha256_file(output / "development.json")
    write_json(output / "model_lock.json", lock)
    for source, target in [
        (output / "model_lock.json", root / PUBLISHED_LOCK),
        (output / "development.json", root / DEVELOPMENT),
    ]:
        with target.open("xb") as handle:
            handle.write(source.read_bytes())
    return lock


def engine_bootstrap(per_engine: dict[str, np.ndarray], resamples: int, seed: int):
    """Percentile intervals over resampled engine columns."""
    count = len(next(iter(per_engine.values())))
    columns = np.random.default_rng(seed).integers(0, count, size=(resamples, count))
    return {
        name: np.percentile(transform(values[columns].mean(axis=1)), [2.5, 97.5])
        .astype(float)
        .tolist()
        for name, (values, transform) in per_engine.items()
    }


def final_gates(point, baseline, mondrian_coverage, replay, gates) -> dict:
    checks = {
        "G1_rmse_not_worse_than_baseline": point["rmse"] <= baseline["rmse"],
        "G2_mondrian_coverage": mondrian_coverage >= gates["min_mondrian_coverage"],
        "G3_timely_point_alarm": replay["missed_fraction"] is not None
        and replay["missed_fraction"] <= gates["max_untimely_fraction"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def evaluate(root: Path, output: Path, config: dict) -> dict:
    commit = require_clean_tree(root)
    if (output / "evidence.json").exists():
        raise FileExistsError("v5 final evidence already exists")
    lock_bytes = (output / "model_lock.json").read_bytes()
    if _git(root, "show", f"HEAD:{PUBLISHED_LOCK}") != lock_bytes:
        raise RuntimeError("Run lock differs from the lock committed in HEAD")
    lock = json.loads(lock_bytes)
    checks = {
        "protocol": (sha256_file(root / PROTOCOL), lock["protocol_sha256"]),
        "config": (sha256_file(root / CONFIG), lock["config_sha256"]),
        "reservation": (sha256_file(root / RESERVATION), lock["reservation_sha256"]),
        "model": (sha256_file(output / "model.joblib"), lock["model_sha256"]),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise RuntimeError(f"{label} hash differs from the v5 lock")
    if source_hashes(root) != lock["source_sha256"]:
        raise RuntimeError("Source code differs from the v5 lock")
    raw = root / config["raw_dir"]
    paths = [raw / name for name in config["test_files"]]
    for path in paths:
        if sha256_file(path) != lock["test_data_sha256"][path.name]:
            raise RuntimeError(f"{path.name} differs from the reserved file")
    claim_holdout(
        root / LEDGER,
        LEDGER_KEY,
        {
            "model_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "protocol_sha256": lock["protocol_sha256"],
            "data_sha256": lock["test_data_sha256"],
            "git_commit": commit,
        },
    )
    frame = load_members(paths, TEST_NAMESPACE)
    for name, units in lock["test_engines"].items():
        member = frame[frame.source_file == name]
        for split in ("dev", "test"):
            found = sorted(member.loc[member.source_split == split].unit_id.unique())
            if found != units[split]:
                raise RuntimeError(f"{name} {split} engines differ from reservation")
    if frame.unit_id.nunique() != TEST_ENGINES:
        raise RuntimeError("Final test must contain the 35 reserved engines")
    if feature_columns(frame) != lock["features"]:
        raise RuntimeError("Feature columns differ from the lock")
    y = read_multiple_targets(paths, frame)
    model = joblib.load(output / "model.joblib")
    prediction = predict(model, frame[lock["features"]].to_numpy(dtype=float))
    engines = frame.unit_id.to_numpy()
    plan = draw_engine_rows(
        engines, np.ones(len(frame), dtype=bool), draws=config["draws"], seed=46
    )
    truth, sampled = y[plan], prediction[plan]
    baseline_value = np.full(plan.shape, lock["baseline"])
    point = point_metrics(truth, sampled)
    baseline = point_metrics(truth, baseline_value)
    files = frame.source_file.to_numpy()[plan[0]]
    intervals = {
        "mondrian": apply_mondrian_interval(prediction, lock["mondrian"]),
        "split_conformal": (
            np.maximum(0.0, prediction - lock["split_margin"]),
            prediction + lock["split_margin"],
        ),
    }
    flat = plan.ravel()
    groups = {
        "overall": np.ones(len(flat), dtype=bool),
        "rul_<=30": y[flat] <= 30,
        "rul_31-60": (y[flat] > 30) & (y[flat] <= 60),
        "rul_>60": y[flat] > 60,
    }
    for name in config["test_files"]:
        groups[f"file_{name}"] = frame.source_file.to_numpy()[flat] == name
    interval_report = {
        method: {
            label: interval_metrics(
                y[flat][mask], low[flat][mask], high[flat][mask], engines[flat][mask]
            )
            for label, mask in groups.items()
        }
        for method, (low, high) in intervals.items()
    }
    replay_frame = frame[["unit_id", "cycle"]].assign(rul_true=y)
    alarm = {
        "persistence": config["persistence_cycles"],
        "threshold": config["alarm_threshold"],
    }
    replays = {
        "point_timely": replay_alarms(
            replay_frame, prediction, min_rul=config["timely_min_rul"], **alarm
        ),
        "point_any": replay_alarms(replay_frame, prediction, **alarm),
        "mondrian_lower_timely": replay_alarms(
            replay_frame,
            intervals["mondrian"][0],
            min_rul=config["timely_min_rul"],
            **alarm,
        ),
        "split_lower_timely": replay_alarms(
            replay_frame,
            intervals["split_conformal"][0],
            min_rul=config["timely_min_rul"],
            **alarm,
        ),
    }
    low, high = intervals["mondrian"]
    covered = (truth >= low[plan]) & (truth <= high[plan])
    squared = (sampled - truth) ** 2
    squared_baseline = (baseline_value - truth) ** 2
    report = {
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "git_commit": commit,
        "chosen": lock["chosen"],
        "point": point,
        "baseline": baseline,
        "per_file": {
            name: {
                "point": point_metrics(
                    truth[:, files == name], sampled[:, files == name]
                ),
                "baseline": point_metrics(
                    truth[:, files == name], baseline_value[:, files == name]
                ),
            }
            for name in config["test_files"]
        },
        "intervals": interval_report,
        "replay": {name: summarize_replay(r) for name, r in replays.items()},
        "bootstrap_95": engine_bootstrap(
            {
                "rmse": (squared.mean(axis=0), np.sqrt),
                "baseline_rmse": (squared_baseline.mean(axis=0), np.sqrt),
                "mondrian_coverage": (covered.mean(axis=0), lambda v: v),
            },
            config["bootstrap_resamples"],
            seed=44,
        ),
    }
    report["gates"] = final_gates(
        point,
        baseline,
        interval_report["mondrian"]["overall"]["coverage"],
        report["replay"]["point_timely"],
        lock["gates"],
    )
    records = pd.DataFrame(replays["point_timely"]["records"])
    records.to_csv(output / "point_alarm_records.csv", index=False)
    evidence = frame[["source_file", "archive_unit_id", "unit_id", "cycle"]].copy()
    evidence["rul_true"], evidence["prediction"] = y, prediction
    for method, (lower, upper) in intervals.items():
        evidence[f"{method}_low"], evidence[f"{method}_high"] = lower, upper
    evidence.to_parquet(output / "evaluation.parquet", index=False)
    write_json(output / "evidence.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["hybrid", "prepare", "evaluate"])
    parser.add_argument("--project-root", type=Path, default=Path())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads((root / CONFIG).read_text(encoding="utf-8"))
    stage = {"hybrid": run_hybrid, "prepare": prepare, "evaluate": evaluate}
    stage[args.stage](root, args.output.resolve(), config)


if __name__ == "__main__":
    main()
