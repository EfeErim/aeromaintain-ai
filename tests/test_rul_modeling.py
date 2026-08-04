from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from aeromaintain.data.pipeline import SCHEMA, sha256_file
from aeromaintain.evaluation import nasa_score, prediction_metrics
from aeromaintain.features import (
    FEATURE_NAMES,
    FoldPreprocessor,
    build_causal_features,
)
from aeromaintain.features.causal import FeatureContractError
from aeromaintain.models.rul import (
    ModelingConfig,
    ModelingError,
    engine_equal_weights,
    evaluate_locked,
    finite_sample_quantile,
    train_and_lock,
)


def _sensor_frame(
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


def _write_phase1_fixture(root: Path) -> None:
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
    train = _sensor_frame(
        development_ids + calibration_ids,
        cycles=8,
        targets=True,
        roles=roles,
    )
    test = _sensor_frame(list(range(1, 9)), cycles=5, targets=False)
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


def test_causal_features_do_not_change_past_and_keep_order() -> None:
    source = _sensor_frame([1, 2], cycles=25, targets=False)
    original = build_causal_features(source)
    changed = source.copy()
    future_index = changed.index[changed["unit_id"].eq(1) & changed["cycle"].eq(22)][0]
    changed.loc[future_index, "sensor_1"] += 10_000
    modified = build_causal_features(changed)

    past = source["unit_id"].eq(1) & source["cycle"].lt(22)
    pd.testing.assert_frame_equal(original.loc[past], modified.loc[past])
    assert tuple(original.columns) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 403
    assert (
        original.loc[future_index, "sensor_1_w5_mean"]
        != modified.loc[future_index, "sensor_1_w5_mean"]
    )

    preprocessor = FoldPreprocessor(scale=True).fit(original)
    transformed = preprocessor.transform(original)
    assert transformed.shape[1] == len(preprocessor.output_features or ())
    with pytest.raises(FeatureContractError, match="Feature order mismatch"):
        preprocessor.transform(original.loc[:, reversed(original.columns)])


def test_feature_source_and_preprocessor_fail_closed() -> None:
    source = _sensor_frame([1], cycles=4, targets=False)
    with pytest.raises(FeatureContractError, match="ordered"):
        build_causal_features(source.iloc[::-1])
    duplicate = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    with pytest.raises(FeatureContractError, match="duplicate"):
        build_causal_features(duplicate)
    with pytest.raises(FeatureContractError, match="fitted"):
        FoldPreprocessor(scale=False).transform(build_causal_features(source))

    constant = build_causal_features(source).copy()
    constant.loc[:, :] = 1.0
    with pytest.raises(FeatureContractError, match="no non-constant"):
        FoldPreprocessor(scale=False).fit(constant)


def test_metrics_weights_and_finite_sample_interval_contract() -> None:
    truth = np.array([10.0, 20.0, 40.0, 70.0, 130.0])
    prediction = np.array([12.0, 18.0, 35.0, 80.0, 125.0])
    engines = np.array([1, 1, 2, 2, 3])
    expected_penalties = np.where(
        prediction - truth < 0,
        np.expm1(-(prediction - truth) / 13),
        np.expm1((prediction - truth) / 10),
    )
    expected_nasa = (
        expected_penalties[:2].mean()
        + expected_penalties[2:4].mean()
        + expected_penalties[4]
    )
    assert nasa_score(truth, prediction, engines) == pytest.approx(expected_nasa)
    metrics = prediction_metrics(truth, prediction, engines)
    assert metrics["rul_bands"][">125"]["rows"] == 1
    assert metrics["critical_rul"]["true_positive"] == 2
    assert metrics["rmse"] > metrics["mae"]

    weights = engine_equal_weights(np.array([1, 1, 2]))
    assert weights[:2].sum() == pytest.approx(weights[2:].sum())
    assert weights.mean() == pytest.approx(1.0)
    assert finite_sample_quantile(np.arange(1, 21), 0.9) == 19
    with pytest.raises(ModelingError, match="at least one"):
        finite_sample_quantile(np.array([]), 0.9)


def test_train_lock_and_official_evaluation_are_reproducible(
    tmp_path: Path,
) -> None:
    _write_phase1_fixture(tmp_path)
    config = ModelingConfig(
        seed=42,
        group_folds=2,
        critical_rul_threshold=30,
        nominal_interval_coverage=0.9,
        xgboost_candidates=1,
        xgboost_max_trees=20,
        xgboost_early_stopping_rounds=5,
        xgboost_n_jobs=1,
    )
    result = train_and_lock(tmp_path, run_id="fixture-m2", config=config)
    assert result.model_lock.is_file()
    lock = json.loads(result.model_lock.read_text(encoding="utf-8"))
    assert lock["official_test_evaluation"].startswith("not opened")
    assert lock["features"]["order"] == list(FEATURE_NAMES)
    assert lock["calibration"]["score_count"] == 2
    assert lock["environment"]["python"]["version"].startswith("3.11")
    assert lock["environment"]["packages"]["numpy"] == np.__version__
    assert lock["environment"]["constraints"]["sha256"] == sha256_file(
        tmp_path / "constraints" / "python311-tested.txt"
    )
    run_manifest = json.loads(
        (result.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["environment"] == lock["environment"]
    assert set(lock["development_engine_ids"]).isdisjoint(
        lock["calibration_engine_ids"]
    )
    decision = json.loads(
        (result.run_dir / "champion_decision.json").read_text(encoding="utf-8")
    )
    assert decision["champion"] in {"ridge", "xgboost"}
    assert (
        len(
            json.loads(
                (result.run_dir / "model_comparison.json").read_text(encoding="utf-8")
            )["xgboost_candidates"]
        )
        == 1
    )

    first = evaluate_locked(tmp_path, run_id="fixture-m2")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in first.output_dir.iterdir()
    }
    second = evaluate_locked(tmp_path, run_id="fixture-m2")
    assert first.metrics == second.metrics
    assert hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in second.output_dir.iterdir()
    }
    predictions = pd.read_parquet(first.output_dir / "predictions.parquet")
    assert len(predictions) == 8
    assert predictions["interval_low"].le(predictions["interval_high"]).all()
    assert set(predictions["risk_band"]) <= {
        "critical",
        "elevated",
        "routine",
    }

    with pytest.raises(ModelingError, match="already exists"):
        train_and_lock(tmp_path, run_id="fixture-m2", config=config)
    with pytest.raises(ModelingError, match="safe path"):
        train_and_lock(tmp_path, run_id="../unsafe", config=config)

    feature_manifest = result.run_dir / "feature_manifest.json"
    original = feature_manifest.read_bytes()
    feature_manifest.write_bytes(original + b"changed")
    with pytest.raises(ModelingError, match="Locked artifact hash mismatch"):
        evaluate_locked(tmp_path, run_id="fixture-m2")


def test_evaluation_requires_lock_before_official_label_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(ModelingError, match="requires model_lock"):
        evaluate_locked(tmp_path, run_id="missing")
