from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cmapss_fixture import sensor_frame, small_config, write_prepared_fixture
from streamlit.testing.v1 import AppTest

from aeromaintain.evaluation import (
    alert_rule_metrics,
    nasa_score,
    prediction_metrics,
    wilson_interval,
)
from aeromaintain.features import (
    FEATURE_NAMES,
    FoldPreprocessor,
    build_causal_features,
)
from aeromaintain.features.causal import FeatureContractError
from aeromaintain.files import sha256_file
from aeromaintain.models.rul import (
    RISK_THRESHOLD_GRID,
    ModelingError,
    _select_champion,
    engine_equal_weights,
    evaluate_locked,
    train_and_lock,
)


def test_causal_features_do_not_change_past_and_keep_order() -> None:
    source = sensor_frame([1, 2], cycles=25, targets=False)
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


def test_rolling_statistics_match_a_known_linear_trend() -> None:
    # In the fixture, sensor_k rises by 0.05 * k per cycle on every engine.
    source = sensor_frame([1, 2], cycles=12, targets=False)
    features = build_causal_features(source)
    row = source.index[source["unit_id"].eq(2) & source["cycle"].eq(10)][0]
    value = source.set_index(["unit_id", "cycle"])["sensor_3"]

    assert features.loc[row, "sensor_1_w5_slope"] == pytest.approx(0.05)
    assert features.loc[row, "sensor_3_w10_slope"] == pytest.approx(0.15)
    assert features.loc[row, "sensor_3_w5_mean"] == pytest.approx(value[(2, 8)])
    assert features.loc[row, "sensor_3_w5_min"] == pytest.approx(value[(2, 6)])
    assert features.loc[row, "sensor_3_w5_max"] == pytest.approx(value[(2, 10)])
    assert features.loc[row, "sensor_3_w5_std"] == pytest.approx(0.15 * np.sqrt(2))
    assert features.loc[row, "sensor_3_w5_delta_mean"] == pytest.approx(0.3)
    first = source.index[source["unit_id"].eq(2) & source["cycle"].eq(1)][0]
    assert features.loc[first, "sensor_1_w5_slope"] == 0.0


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


def test_alert_rule_reports_both_error_costs_with_wilson_intervals() -> None:
    truth = np.array([10.0, 25.0, 30.0, 31.0, 80.0, 120.0])
    interval_low = np.array([0.0, 20.0, 45.0, 25.0, 70.0, 90.0])

    rule = alert_rule_metrics(truth, interval_low)

    assert (rule["true_positive"], rule["false_negative"]) == (2, 1)
    assert (rule["false_positive"], rule["true_negative"]) == (1, 2)
    assert rule["precision"] == pytest.approx(2 / 3)
    assert rule["recall"] == pytest.approx(2 / 3)
    assert rule["false_alarm_rate"] == pytest.approx(1 / 3)
    assert (
        rule["wilson_95"]["recall"]["low"] < 2 / 3 < rule["wilson_95"]["recall"]["high"]
    )

    # The published 25/25 reference recall has a lower bound near 87%, not 100%.
    perfect = wilson_interval(25, 25)
    assert perfect == {"low": pytest.approx(0.8668, abs=1e-4), "high": 1.0}
    # One success in ten keeps a positive lower bound; only zero successes clamp to 0.
    assert wilson_interval(1, 10) == {
        "low": pytest.approx(0.0179, abs=1e-4),
        "high": pytest.approx(0.4042, abs=1e-4),
    }


def test_train_lock_and_official_evaluation_are_reproducible(
    tmp_path: Path,
) -> None:
    write_prepared_fixture(tmp_path)
    config = small_config()
    result = train_and_lock(tmp_path, run_id="fixture-run", config=config)
    assert result.model_lock.is_file()
    lock = json.loads(result.model_lock.read_text(encoding="utf-8"))
    assert lock["official_test_evaluation"].startswith("not opened")
    assert set(lock["features"]["order"]) <= set(FEATURE_NAMES)
    assert lock["schema_version"] == 3
    assert lock["protocol"]["test_status"] == "previously_seen"
    assert lock["calibration"]["engines"] == 8
    assert lock["calibration"]["risk_threshold"] in RISK_THRESHOLD_GRID
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

    first = evaluate_locked(tmp_path, run_id="fixture-run")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in first.output_dir.iterdir()
    }
    second = evaluate_locked(tmp_path, run_id="fixture-run")
    assert first.metrics == second.metrics
    assert hashes == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in second.output_dir.iterdir()
    }
    predictions = pd.read_parquet(first.output_dir / "predictions.parquet")
    assert len(predictions) == 8
    assert predictions["interval_low"].le(predictions["interval_high"]).all()
    rules = first.metrics["alert_rules"]
    assert rules["interval_low"]["recall"] >= rules["point_prediction"]["recall"]

    with pytest.raises(ModelingError, match="already exists"):
        train_and_lock(tmp_path, run_id="fixture-run", config=config)
    with pytest.raises(ModelingError, match="safe path"):
        train_and_lock(tmp_path, run_id="../unsafe", config=config)

    feature_manifest = result.run_dir / "feature_manifest.json"
    original = feature_manifest.read_bytes()
    feature_manifest.write_bytes(original + b"changed")
    with pytest.raises(ModelingError, match="Model artefact SHA-256 mismatch"):
        evaluate_locked(tmp_path, run_id="fixture-run")


def test_evaluation_requires_lock_before_official_label_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(ModelingError, match="Run directory does not exist"):
        evaluate_locked(tmp_path, run_id="missing")


def test_evaluation_refuses_changed_or_unanchored_lock(tmp_path: Path) -> None:
    write_prepared_fixture(tmp_path)
    result = train_and_lock(tmp_path, run_id="fixture-lock", config=small_config())
    official_dir = result.run_dir / "official_test"
    original_lock = result.model_lock.read_bytes()

    changed = json.loads(original_lock)
    changed["calibration"]["risk_threshold"] = 9999.0
    result.model_lock.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ModelingError, match="Model lock SHA-256 mismatch"):
        evaluate_locked(tmp_path, run_id="fixture-lock")
    assert not official_dir.exists()

    result.model_lock.write_bytes(original_lock)
    manifest_path = result.run_dir / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest_path.unlink()
    with pytest.raises(ModelingError, match="Required artefact is missing"):
        evaluate_locked(tmp_path, run_id="fixture-lock")
    assert not official_dir.exists()

    manifest_path.write_bytes(original_manifest)
    assert evaluate_locked(tmp_path, run_id="fixture-lock").output_dir == official_dir


def test_xgboost_champion_runs_through_evaluation_and_application(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def force_xgboost(ridge, xgboost, config):
        decision = _select_champion(ridge, xgboost, config)
        return {**decision, "champion": "xgboost"}

    monkeypatch.setattr("aeromaintain.models.rul._select_champion", force_xgboost)
    write_prepared_fixture(tmp_path)
    result = train_and_lock(tmp_path, run_id="fixture-xgb", config=small_config())
    evaluate_locked(tmp_path, run_id="fixture-xgb")

    lock = json.loads(result.model_lock.read_text(encoding="utf-8"))
    explanation = json.loads(
        (result.run_dir / "explanation.json").read_text(encoding="utf-8")
    )
    assert lock["champion"]["kind"] == "xgboost"
    assert explanation["method"] == "SHAP TreeExplainer"
    assert "mean_abs_shap" in explanation["global_importance"][0]

    monkeypatch.setenv("AEROMAINTAIN_RUN_ID", "fixture-xgb")
    monkeypatch.setenv("AEROMAINTAIN_PROJECT_ROOT", str(tmp_path))
    script = (
        Path(__file__).resolve().parents[1] / "src" / "aeromaintain" / "app" / "main.py"
    )
    tested = AppTest.from_file(str(script), default_timeout=30).run()
    assert not tested.exception
    tested.sidebar.radio[0].set_value("Engine risk")
    tested.run()
    assert not tested.exception
    assert tested.expander[0].label == "Model SHAP review"
