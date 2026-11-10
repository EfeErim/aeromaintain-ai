from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from test_rul_modeling import _sensor_frame, _small_config, _write_prepared_fixture

from aeromaintain.data.pipeline import DATASET_SHAPES, load_data_contract, sha256_file
from aeromaintain.evaluation.protocol import (
    apply_mondrian_cqr,
    fit_mondrian_cqr,
    leave_engines_out_cqr,
)
from aeromaintain.features import (
    CONDITION_FEATURE,
    ConditionNormalizer,
    build_condition_features,
)
from aeromaintain.features.causal import FeatureContractError
from aeromaintain.models import compare_runs, evaluate_locked, rul, run_seed_sweep
from aeromaintain.models.evaluation import _validate_lock
from aeromaintain.models.rul import (
    ModelingConfig,
    ModelingError,
    _run_stage,
    _SelectionData,
    calibrate_interval,
    load_modeling_config,
    load_test_status,
    row_weights,
    train_and_lock,
)

_ORIGINAL_CALIBRATE = calibrate_interval
REPOSITORY = Path(__file__).resolve().parents[1]
SETTINGS = [(0.0, 0.0, 100.0), (10.0, 0.25, 100.0), (20.0, 0.7, 60.0)]


def _condition_frame(engines: list[int], *, cycles: int = 12) -> pd.DataFrame:
    frame = _sensor_frame(engines, cycles=cycles, targets=False)
    labels = (frame["unit_id"] + frame["cycle"]).to_numpy() % len(SETTINGS)
    frame[["setting_1", "setting_2", "setting_3"]] = np.asarray(SETTINGS)[labels]
    for index in range(1, 22):
        frame[f"sensor_{index}"] += labels * 100.0
    return frame


def test_v3_config_marks_only_unopened_subsets_clean() -> None:
    assert load_test_status(REPOSITORY, "FD002") == "clean_single_open"
    assert load_test_status(REPOSITORY, "FD004") == "clean_single_open"
    assert load_test_status(REPOSITORY, "FD001") == "previously_seen"
    assert load_test_status(REPOSITORY, "FD003") == "previously_seen"
    six = load_modeling_config(REPOSITORY, "FD004")
    assert six.operating_conditions == 6
    assert six.rmse_target is None
    assert six.baseline_rmse_reduction_target == 0.10
    assert (six.xgboost_candidates, six.cqr_enabled) == (4, False)
    one = load_modeling_config(REPOSITORY, "FD001")
    assert (one.operating_conditions, one.rmse_target) == (1, 16.6)
    assert (one.xgboost_candidates, one.cqr_enabled) == (12, True)
    assert one.scoring_rul_window == 150


@pytest.mark.parametrize(
    ("dataset", "development", "calibration", "labels"),
    [("FD001", 80, 20, 100), ("FD002", 208, 52, 259), ("FD004", 199, 50, 248)],
)
def test_contract_splits_each_subset_by_its_own_shape(
    dataset: str,
    development: int,
    calibration: int,
    labels: int,
) -> None:
    contract = load_data_contract(REPOSITORY, dataset)

    assert contract.development_engines == development
    assert contract.calibration_engines == calibration
    assert contract.test_labels == labels
    assert development + calibration == DATASET_SHAPES[dataset]["train_engines"]


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


def test_window_focus_lowers_only_rows_above_the_window() -> None:
    frame = pd.DataFrame(
        {"unit_id": [1, 1, 2, 2], "rul_true": [200, 100, 151, 150]},
    )
    config = ModelingConfig()

    uniform = row_weights({"family": "xgboost"}, frame, config)
    focused = row_weights(
        {"family": "xgboost", "emphasis": "window_focus"}, frame, config
    )

    np.testing.assert_allclose(focused, uniform * [0.25, 1.0, 0.25, 1.0])


def test_later_stage_needs_rmse_gain_and_nasa_not_worse(monkeypatch) -> None:
    current = {
        "spec": {"name": "current"},
        "metrics": {
            "rmse": {"median": 20.0},
            "nasa_score_motor_normalized": {"median": 100.0},
        },
    }

    def evaluate(spec, data):
        rmse, nasa = spec["result"]
        return {
            "spec": spec,
            "metrics": {
                "rmse": {"median": rmse},
                "nasa_score_motor_normalized": {"median": nasa},
            },
        }

    monkeypatch.setattr("aeromaintain.models.rul._evaluate_candidate", evaluate)
    data = _SelectionData(None, None, [], None, ModelingConfig())

    best, _ = _run_stage(current, [{"result": (19.0, 120.0)}], data, "emphasis")
    assert best is current
    best, _ = _run_stage(current, [{"result": (19.9, 90.0)}], data, "emphasis")
    assert best is current
    best, log = _run_stage(current, [{"result": (19.0, 95.0)}], data, "emphasis")
    assert best["spec"]["result"] == (19.0, 95.0)
    assert log[0]["accepted"] is True


def test_mondrian_cqr_covers_with_narrower_intervals_on_heteroscedastic_data() -> None:
    generator = np.random.default_rng(0)
    engines = np.repeat(np.arange(200), 5)
    prediction = generator.uniform(0, 140, size=len(engines))
    scale = 2 + prediction / 5
    truth = prediction + generator.normal(0, scale)
    quantiles = np.column_stack(
        [prediction - 1.645 * scale, prediction + 1.645 * scale]
    )

    interval = fit_mondrian_cqr(
        prediction,
        quantiles,
        truth,
        engines,
        coverage=0.9,
        edges=(30, 60, 100),
        min_engines=20,
    )
    low, high = apply_mondrian_cqr(prediction, quantiles, interval)

    assert (low >= 0).all()
    assert (high >= low).all()
    assert 0.87 <= np.mean((truth >= low) & (truth <= high)) <= 0.95
    assert abs(interval["bands"][0]["margin"]) < 5

    folds = engines % 5
    arguments = {"coverage": 0.9, "edges": (30, 60, 100), "min_engines": 20}
    first_low, _ = leave_engines_out_cqr(
        prediction,
        quantiles,
        truth,
        engines,
        np.ones(len(truth), dtype=bool),
        folds,
        **arguments,
    )
    changed = truth.copy()
    changed[folds == 0] += 1_000
    second_low, _ = leave_engines_out_cqr(
        prediction,
        quantiles,
        changed,
        engines,
        np.ones(len(truth), dtype=bool),
        folds,
        **arguments,
    )
    np.testing.assert_array_equal(first_low[folds == 0], second_low[folds == 0])


def _write_condition_fixture(root: Path) -> None:
    _write_prepared_fixture(root)
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
        _small_config(),
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


def test_v2_lock_schema_still_validates_but_unknown_schema_is_refused(
    tmp_path: Path,
) -> None:
    _write_prepared_fixture(tmp_path)
    result = train_and_lock(tmp_path, run_id="fixture", config=_small_config())
    _validate_lock(tmp_path, "fixture")

    lock = json.loads(result.model_lock.read_text(encoding="utf-8"))
    lock["schema_version"] = 99
    result.model_lock.write_text(json.dumps(lock), encoding="utf-8")
    manifest_path = result.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_lock_sha256"] = sha256_file(result.model_lock)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelingError, match="identity or schema"):
        _validate_lock(tmp_path, "fixture")


def test_project_config_stays_unchanged_for_earlier_locks() -> None:
    payload = yaml.safe_load(
        (REPOSITORY / "configs" / "project.yaml").read_text(encoding="utf-8")
    )
    assert payload["data"]["datasets"]["FD003"]["test_status"] == "clean_single_open"
