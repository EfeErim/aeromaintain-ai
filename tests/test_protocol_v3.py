from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cmapss_fixture import sensor_frame, small_config, write_prepared_fixture

from aeromaintain.features import (
    CONDITION_FEATURE,
    ConditionNormalizer,
    build_condition_features,
)
from aeromaintain.features.causal import FeatureContractError
from aeromaintain.files import sha256_file
from aeromaintain.models import compare_runs, evaluate_locked, rul, run_seed_sweep
from aeromaintain.models.rul import (
    calibrate_interval,
    train_and_lock,
)

_ORIGINAL_CALIBRATE = calibrate_interval
REPOSITORY = Path(__file__).resolve().parents[1]
SETTINGS = [(0.0, 0.0, 100.0), (10.0, 0.25, 100.0), (20.0, 0.7, 60.0)]


def _condition_frame(engines: list[int], *, cycles: int = 12) -> pd.DataFrame:
    frame = sensor_frame(engines, cycles=cycles, targets=False)
    labels = (frame["unit_id"] + frame["cycle"]).to_numpy() % len(SETTINGS)
    frame[["setting_1", "setting_2", "setting_3"]] = np.asarray(SETTINGS)[labels]
    for index in range(1, 22):
        frame[f"sensor_{index}"] += labels * 100.0
    return frame


def test_condition_normalizer_removes_condition_shift_and_is_causal() -> None:
    frame = _condition_frame([1, 2, 3, 4])
    normalizer = ConditionNormalizer(conditions=3, seed=1).fit(frame)

    normalized = normalizer.normalized_frame(frame)
    labels = normalizer.labels(frame)
    assert len(set(labels)) == 3
    for label in range(3):
        means = normalized.loc[labels == label, "sensor_1"].mean()
        assert means == pytest.approx(0.0, abs=1e-9)
    assert frame["sensor_1"].std() > 10 * normalized["sensor_1"].std()

    features = build_condition_features(frame, normalizer)
    changed = frame.copy()
    future = changed.index[changed["unit_id"].eq(1) & changed["cycle"].eq(10)][0]
    changed.loc[future, "sensor_1"] += 5.0
    modified = build_condition_features(changed, normalizer)
    past = frame["unit_id"].eq(1) & frame["cycle"].lt(10)
    pd.testing.assert_frame_equal(features.loc[past], modified.loc[past])


def test_cycles_since_condition_change_restarts_per_engine_and_condition() -> None:
    frame = _condition_frame([1, 2], cycles=4)
    settings = np.asarray(SETTINGS)[[0, 0, 1, 1, 0, 0, 0, 2]]
    frame[["setting_1", "setting_2", "setting_3"]] = settings
    normalizer = ConditionNormalizer(conditions=3, seed=1).fit(frame)

    features = build_condition_features(frame, normalizer)

    assert features[CONDITION_FEATURE].tolist() == [0, 1, 0, 1, 0, 1, 2, 0]


def test_normalizer_fails_closed_on_inseparable_settings() -> None:
    frame = _condition_frame([1, 2, 3])
    generator = np.random.default_rng(0)
    frame["setting_1"] = generator.uniform(0, 1, size=len(frame))
    frame["setting_2"] = generator.uniform(0, 1, size=len(frame))

    with pytest.raises(FeatureContractError, match="do not separate"):
        ConditionNormalizer(conditions=3, seed=1).fit(frame)
    with pytest.raises(FeatureContractError, match="fitted"):
        ConditionNormalizer(conditions=3, seed=1).labels(frame)


def test_condition_scales_come_from_training_engines_only() -> None:
    frame = _condition_frame([1, 2, 3, 4, 5, 6])
    config = replace(small_config(), operating_conditions=3)
    spec = {"family": "ridge", "condition": True}
    train_rows = np.flatnonzero(frame["unit_id"].le(4))
    held_out = frame["unit_id"].gt(4)

    features, normalizer = rul.partition_features(
        spec, frame, frame, train_rows, config
    )
    changed = frame.copy()
    changed.loc[held_out, "sensor_1"] += 10_000.0
    new_features, new_normalizer = rul.partition_features(
        spec, changed, changed, train_rows, config
    )

    np.testing.assert_array_equal(normalizer.sensor_mean, new_normalizer.sensor_mean)
    np.testing.assert_array_equal(normalizer.sensor_scale, new_normalizer.sensor_scale)
    pd.testing.assert_frame_equal(features.loc[~held_out], new_features.loc[~held_out])


def _write_condition_fixture(root: Path) -> None:
    write_prepared_fixture(root)
    processed = root / "data" / "processed" / "fd001"
    train = pd.read_parquet(processed / "train.parquet")
    test = pd.read_parquet(processed / "test.parquet")
    for frame in (train, test):
        labels = (frame["unit_id"] + frame["cycle"]).to_numpy() % len(SETTINGS)
        frame[["setting_1", "setting_2", "setting_3"]] = np.asarray(SETTINGS)[labels]
    train.to_parquet(processed / "train.parquet", index=False)
    test.to_parquet(processed / "test.parquet", index=False)
    manifest_path = processed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["train.parquet"] = sha256_file(processed / "train.parquet")
    manifest["artifacts"]["test.parquet"] = sha256_file(processed / "test.parquet")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_all_v3_stages_lock_evaluate_and_repeat_with_forced_adoption(
    monkeypatch,
    tmp_path: Path,
) -> None:
    original = rul._run_stage

    def adopt_first(current, alternatives, data, stage):
        if stage in {"features", "cap"}:
            return original(current, alternatives, data, stage)
        record = rul._evaluate_candidate(alternatives[0], data)
        return record, [{"stage": stage, "accepted": True}]

    def force_xgboost(ridge, xgboost, config):
        decision = rul_select(ridge, xgboost, config)
        return {**decision, "champion": "xgboost"}

    rul_select = rul._select_champion
    monkeypatch.setattr(rul, "_run_stage", adopt_first)
    monkeypatch.setattr(rul, "_select_champion", force_xgboost)
    monkeypatch.setattr(
        rul,
        "calibrate_interval",
        lambda cross_fit, frame, config, *, seed, forced_method=None: calibrate(
            cross_fit, frame, config, seed=seed, forced_method="cqr_uncapped"
        ),
    )
    calibrate = _ORIGINAL_CALIBRATE
    _write_condition_fixture(tmp_path)
    config = replace(
        small_config(),
        operating_conditions=3,
        ensemble_members=2,
        condition_max_within_spread=1.0,
    )

    result = train_and_lock(tmp_path, run_id="all", config=config)
    lock = json.loads(result.model_lock.read_text(encoding="utf-8"))
    assert lock["schema_version"] == 3
    assert lock["champion"]["condition"] is True
    assert lock["champion"]["emphasis"] == "window_focus"
    assert len(lock["champion"]["member_paths"]) == 2
    assert CONDITION_FEATURE in lock["features"]["order"]
    assert lock["calibration"]["method"] == "mondrian-cqr-cross-fit"
    assert lock["calibration"]["quantile_target"] == "uncapped"
    assert (result.run_dir / "condition_normalizer.joblib").is_file()
    assert (result.run_dir / "quantile_model.json").is_file()

    evaluated = evaluate_locked(tmp_path, run_id="all")
    assert evaluated.metrics["nasa_score_mean_per_engine"] == pytest.approx(
        evaluated.metrics["nasa_score_motor_normalized"] / evaluated.metrics["engines"]
    )
    frame = pd.read_parquet(result.run_dir / "official_test" / "predictions.parquet")
    assert frame["interval_low"].le(frame["interval_high"]).all()
    assert len(frame) == 8

    monkeypatch.undo()
    report = run_seed_sweep(tmp_path, base_run_id="all", seeds=(1,))
    assert report["per_seed"][0]["risk_threshold"] >= 0
    assert (
        compare_runs(tmp_path, first_run_id="all", second_run_id="all")[
            "different_files"
        ]
        == []
    )


# The v2 model locks record this hash, and so does every prepared C-MAPSS data
# manifest, which the v2 and v3 locks hash in turn. Any edit to the file, even
# removing an unused key, stops every published C-MAPSS run from verifying.
PROJECT_CONFIG_SHA256 = (
    "10744394d2a9e26195db590fbae51352e23c475be68a83db4ecd921c49bfe027"
)


def test_project_config_stays_unchanged_for_earlier_locks() -> None:
    assert sha256_file(REPOSITORY / "configs" / "project.yaml") == PROJECT_CONFIG_SHA256
