"""A tiny prepared C-MAPSS project for the training and evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from aeromaintain.data.pipeline import SCHEMA
from aeromaintain.files import sha256_file
from aeromaintain.models.rul import ModelingConfig


def sensor_frame(
    engine_ids: list[int],
    *,
    cycles: int,
    targets: bool,
    roles: dict[int, str] | None = None,
) -> pd.DataFrame:
    rows: list[list[float]] = []
    for unit_id in engine_ids:
        for cycle in range(1, cycles + 1):
            settings = [unit_id / 10, cycle / 20, (unit_id + cycle) % 3]
            sensors = [
                0.3 * unit_id
                + 0.05 * cycle * (sensor_index + 1)
                + ((unit_id * sensor_index) % 5) / 10
                for sensor_index in range(21)
            ]
            rows.append([unit_id, cycle, *settings, *sensors])
    frame = pd.DataFrame(rows, columns=SCHEMA)
    frame[["unit_id", "cycle"]] = frame[["unit_id", "cycle"]].astype("int64")
    frame[list(SCHEMA[2:])] = frame[list(SCHEMA[2:])].astype("float64")
    if targets:
        frame["rul_true"] = (
            frame.groupby("unit_id")["cycle"].transform("max") - frame["cycle"]
        )
        frame["rul_target"] = frame["rul_true"].clip(upper=6)
        frame["role"] = frame["unit_id"].map(roles or {})
    return frame


def write_prepared_fixture(root: Path) -> None:
    constraints = root / "constraints"
    constraints.mkdir(parents=True)
    (constraints / "python311-tested.txt").write_text(
        "numpy==2.4.6\nscikit-learn==1.9.0\n",
        encoding="utf-8",
    )
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    config = {
        "project": {"seed": 42},
        "modeling": {
            "group_folds": 2,
            "critical_rul_threshold": 30,
            "nominal_interval_coverage": 0.9,
        },
    }
    config_path = config_dir / "project.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    development_ids = [1, 2, 3, 4, 5, 6]
    calibration_ids = [7, 8]
    roles = {
        **dict.fromkeys(development_ids, "development"),
        **dict.fromkeys(calibration_ids, "calibration"),
    }
    train = sensor_frame(
        development_ids + calibration_ids,
        cycles=8,
        targets=True,
        roles=roles,
    )
    test = sensor_frame(list(range(1, 9)), cycles=5, targets=False)
    test_rul = pd.DataFrame(
        {"unit_id": list(range(1, 9)), "rul_true": list(range(4, 12))}
    )
    processed = root / "data" / "processed" / "fd001"
    evaluation = processed / "evaluation"
    evaluation.mkdir(parents=True)
    train.to_parquet(processed / "train.parquet", index=False)
    test.to_parquet(processed / "test.parquet", index=False)
    test_rul.to_parquet(evaluation / "test_rul.parquet", index=False)
    split = {
        "seed": 42,
        "development_engine_ids": development_ids,
        "calibration_engine_ids": calibration_ids,
    }
    (processed / "split_manifest.json").write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        "train.parquet": sha256_file(processed / "train.parquet"),
        "test.parquet": sha256_file(processed / "test.parquet"),
        "split_manifest.json": sha256_file(processed / "split_manifest.json"),
        "evaluation/test_rul.parquet": sha256_file(evaluation / "test_rul.parquet"),
    }
    manifest = {
        "dataset": "FD001",
        "config_sha256": sha256_file(config_path),
        "artifacts": artifacts,
    }
    (processed / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def small_config() -> ModelingConfig:
    return ModelingConfig(
        seed=42,
        group_folds=2,
        critical_rul_threshold=30,
        nominal_interval_coverage=0.9,
        xgboost_candidates=1,
        xgboost_max_trees=20,
        xgboost_early_stopping_rounds=5,
        xgboost_n_jobs=1,
        scoring_draws=5,
        sensitivity_windows=(None,),
        interval_min_band_engines=2,
        bootstrap_resamples=50,
        ensemble_members=2,
    )
