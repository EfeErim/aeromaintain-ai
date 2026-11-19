from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cmapss_fixture import sensor_frame, small_config, write_prepared_fixture

from aeromaintain.evaluation.protocol import (
    apply_mondrian_interval,
    draw_engine_rows,
    eligible_rows,
    fit_mondrian_interval,
    leave_engines_out_interval,
)
from aeromaintain.files import sha256_file
from aeromaintain.models import compare_runs, run_seed_sweep
from aeromaintain.models.escalation import (
    build_sequences,
    run_cqr_study,
    run_lstm_study,
)
from aeromaintain.models.evaluation import evaluate_locked
from aeromaintain.models.rul import (
    ModelingConfig,
    ModelingError,
    _select_champion,
    group_fold_plan,
    inner_engine_split,
    train_and_lock,
)


def _draw_record(rmse: float, nasa: float, rmse_by_draw: list[float]) -> dict:
    return {
        "metrics": {
            "rmse": {"median": rmse},
            "nasa_score_motor_normalized": {"median": nasa},
        },
        "_rmse_by_draw": np.asarray(rmse_by_draw),
    }


@pytest.mark.parametrize(
    ("rmse", "nasa", "champion"),
    [
        (18.9, 90.0, "xgboost"),  # 5.5% better and NASA not worse.
        (19.5, 90.0, "ridge"),  # NASA improves, but RMSE gain is below 5%.
        (18.0, 101.0, "ridge"),  # RMSE gain is enough, but NASA is worse.
    ],
)
def test_champion_rule_is_the_written_rule(
    rmse: float,
    nasa: float,
    champion: str,
) -> None:
    decision = _select_champion(
        _draw_record(20.0, 100.0, [20.0, 20.0]),
        _draw_record(rmse, nasa, [rmse, 21.0]),
        ModelingConfig(),
    )

    assert decision["champion"] == champion
    assert decision["xgboost_win_rate_over_paired_draws"] == 0.5
    assert decision["rmse_target_met"] is False


def test_truncation_picks_one_eligible_row_per_engine_reproducibly() -> None:
    engines = np.array([1, 1, 1, 2, 2, 3])
    truth = np.array([200.0, 100.0, 5.0, 149.0, 150.0, 400.0])
    eligible = eligible_rows(truth, 150)

    plan = draw_engine_rows(engines, eligible, draws=30, seed=7)

    # Engine 3 has no eligible row, so it cannot enter any draw.
    assert plan.shape == (30, 2)
    assert np.isin(plan[:, 0], [1, 2]).all()
    assert np.isin(plan[:, 1], [3, 4]).all()
    assert set(plan[:, 0]) == {1, 2}
    np.testing.assert_array_equal(
        plan, draw_engine_rows(engines, eligible, draws=30, seed=7)
    )
    assert eligible_rows(truth, None).all()
    with pytest.raises(ValueError, match="at least one draw"):
        draw_engine_rows(engines, eligible, draws=0, seed=7)
    with pytest.raises(ValueError, match="eligible"):
        draw_engine_rows(engines, np.zeros(6, dtype=bool), draws=1, seed=7)


def test_early_stopping_engines_stay_out_of_fit_and_outer_folds() -> None:
    engines = np.repeat(np.arange(1, 11), 3)
    outer = group_fold_plan(engines, folds=5, seed=42)
    assert len(outer) == 5
    for train_index, validation_index in outer:
        fit_rows, stop_rows = inner_engine_split(
            engines[train_index],
            fraction=0.2,
            seed=1,
        )
        fit_engines = set(engines[train_index][fit_rows])
        stop_engines = set(engines[train_index][stop_rows])
        scored_engines = set(engines[validation_index])
        assert fit_engines and stop_engines
        assert not fit_engines & stop_engines
        assert not (fit_engines | stop_engines) & scored_engines
        assert len(fit_rows) + len(stop_rows) == len(train_index)
    with pytest.raises(ModelingError, match="two training engines"):
        inner_engine_split(np.array([1, 1]), fraction=0.2, seed=1)


def _heteroscedastic_rows() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(0)
    engines = np.repeat(np.arange(200), 5)
    prediction = generator.uniform(0, 140, size=len(engines))
    noise = generator.normal(0, 3 + prediction / 5, size=len(engines))
    return prediction, prediction + noise, engines


def test_mondrian_interval_widens_with_predicted_rul_and_covers() -> None:
    prediction, truth, engines = _heteroscedastic_rows()

    interval = fit_mondrian_interval(
        prediction,
        truth,
        engines,
        coverage=0.9,
        edges=(30, 60, 100),
        min_engines=20,
    )
    low, high = apply_mondrian_interval(prediction, interval)

    assert [band["upper_predicted_rul"] for band in interval["bands"]] == [
        30,
        60,
        100,
        None,
    ]
    widths = [band["q_high"] - band["q_low"] for band in interval["bands"]]
    assert widths == sorted(widths)
    assert all(band["q_low"] <= 0 <= band["q_high"] for band in interval["bands"])
    assert (low >= 0).all()
    assert np.mean((truth >= low) & (truth <= high)) >= 0.89


def test_mondrian_merges_small_bands_and_still_covers_large_predictions() -> None:
    prediction = np.array([5.0, 10.0, 200.0, 210.0])
    truth = np.array([6.0, 12.0, 190.0, 220.0])
    engines = np.array([1, 2, 3, 4])

    interval = fit_mondrian_interval(
        prediction,
        truth,
        engines,
        coverage=0.9,
        edges=(30, 60),
        min_engines=2,
    )
    low, high = apply_mondrian_interval(np.array([1000.0]), interval)

    assert len(interval["bands"]) == 2
    assert [band["engines"] for band in interval["bands"]] == [2, 2]
    assert low[0] <= 1000.0 <= high[0]

    single = fit_mondrian_interval(
        prediction,
        truth,
        engines,
        coverage=0.9,
        edges=(30, 60),
        min_engines=50,
    )
    assert len(single["bands"]) == 1
    assert single["bands"][0]["upper_predicted_rul"] is None


def test_each_engine_weighs_the_same_regardless_of_row_count() -> None:
    # Engine 1 dominates rows but must not dominate the quantile.
    prediction = np.zeros(1_020)
    truth = np.concatenate([np.full(1_000, 1.0), np.arange(1.0, 21.0)])
    engines = np.concatenate([np.ones(1_000), np.arange(2, 22)])

    interval = fit_mondrian_interval(
        prediction,
        truth,
        engines,
        coverage=0.5,
        edges=(),
        min_engines=1,
    )

    assert interval["bands"][0]["q_high"] > 10.0


def test_leave_engines_out_does_not_use_the_held_out_engine() -> None:
    prediction, truth, engines = _heteroscedastic_rows()
    folds = engines % 5
    eligible = np.ones(len(prediction), dtype=bool)
    arguments = {
        "coverage": 0.9,
        "edges": (30, 60, 100),
        "min_engines": 20,
    }
    low, high = leave_engines_out_interval(
        prediction, truth, engines, eligible, folds, **arguments
    )

    changed = truth.copy()
    changed[folds == 0] += 10_000
    new_low, new_high = leave_engines_out_interval(
        prediction, changed, engines, eligible, folds, **arguments
    )

    np.testing.assert_array_equal(low[folds == 0], new_low[folds == 0])
    np.testing.assert_array_equal(high[folds == 0], new_high[folds == 0])
    assert np.mean((truth >= low) & (truth <= high)) >= 0.87


def test_sequences_are_causal_and_pad_with_the_first_row() -> None:
    frame = sensor_frame([1, 2], cycles=6, targets=False)

    sequences = build_sequences(frame, length=4)
    assert sequences.shape == (12, 4, 24)
    first = frame.loc[:, ["setting_1", "sensor_1"]].to_numpy()
    # Engine 1, cycle 2: padded with cycle 1, ending at cycle 2.
    sensor_position = 3
    np.testing.assert_allclose(sequences[1, :3, 0], first[0, 0])
    np.testing.assert_allclose(sequences[1, 3, sensor_position], first[1, 1])

    changed = frame.copy()
    changed.loc[5, "sensor_1"] += 1_000
    np.testing.assert_array_equal(sequences[:5], build_sequences(changed, length=4)[:5])
    # A new engine never inherits the previous engine's rows.
    np.testing.assert_allclose(sequences[6, 0, sensor_position], first[6, 1])


def test_seed_sweep_and_repeat_run_comparison(tmp_path: Path) -> None:
    write_prepared_fixture(tmp_path)
    config = small_config()
    train_and_lock(tmp_path, run_id="one", config=config)
    train_and_lock(tmp_path, run_id="two", config=config)

    report = run_seed_sweep(tmp_path, base_run_id="one", seeds=(1, 2))
    assert report["seeds"] == [1, 2]
    assert len(report["per_seed"]) == 2
    summary = report["summary"]["rmse"]
    assert summary["min"] <= summary["mean"] <= summary["max"]
    assert "Selection is not rerun" in report["scope"]
    with pytest.raises(ModelingError, match="at least one seed"):
        run_seed_sweep(tmp_path, base_run_id="one", seeds=())

    comparison = compare_runs(tmp_path, first_run_id="one", second_run_id="two")
    assert "model_comparison.json" in comparison["identical_files"]
    assert "cross_fit_predictions.parquet" in comparison["identical_files"]
    assert comparison["changed_lock_fields_ignoring_run_id"] == []
    with pytest.raises(ModelingError, match="no model lock"):
        compare_runs(tmp_path, first_run_id="one", second_run_id="missing")


def test_seed_sweep_refuses_a_clean_holdout(tmp_path: Path) -> None:
    write_prepared_fixture(tmp_path)
    train_and_lock(tmp_path, run_id="one", config=small_config())
    run_dir = tmp_path / "runs" / "one"
    lock_path = run_dir / "model_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["protocol"]["test_status"] = "clean_single_open"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_lock_sha256"] = sha256_file(lock_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelingError, match="already seen"):
        run_seed_sweep(tmp_path, base_run_id="one", seeds=(1,))


def test_cqr_study_compares_without_opening_labels(tmp_path: Path) -> None:
    write_prepared_fixture(tmp_path)

    def force_xgboost(ridge, xgboost, config):
        decision = _select_champion(ridge, xgboost, config)
        return {**decision, "champion": "xgboost"}

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr("aeromaintain.models.rul._select_champion", force_xgboost)
        train_and_lock(tmp_path, run_id="base", config=small_config())
    labels = (
        tmp_path / "data" / "processed" / "fd001" / "evaluation" / "test_rul.parquet"
    )
    before = labels.stat().st_atime_ns

    report = run_cqr_study(tmp_path, base_run_id="base")

    assert report["scope"].endswith("test labels not opened")
    assert isinstance(report["cqr_replaces_mondrian"], bool)
    assert 0.0 <= report["cqr"]["median_coverage"] <= 1.0
    assert labels.stat().st_atime_ns >= before


def test_lstm_study_uses_the_same_folds_and_rule(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    write_prepared_fixture(tmp_path)
    train_and_lock(tmp_path, run_id="base", config=small_config())

    report = run_lstm_study(
        tmp_path,
        base_run_id="base",
        configurations=({"hidden_size": 4, "layers": 1, "dropout": 0.0},),
    )

    assert report["records"][0]["metrics"]["draws"] == 5
    assert isinstance(report["lstm_replaces_xgboost"], bool)


def test_clean_holdout_opens_for_one_lock_only(tmp_path: Path) -> None:
    write_prepared_fixture(tmp_path)
    config_path = tmp_path / "configs" / "project.yaml"
    config_text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        config_text
        + "data:\n  datasets:\n    FD001:\n      test_status: clean_single_open\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "data" / "processed" / "fd001" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_sha256"] = sha256_file(config_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    first = train_and_lock(tmp_path, run_id="first", config=small_config())
    second = train_and_lock(tmp_path, run_id="second", config=small_config())
    locked = json.loads(first.model_lock.read_text(encoding="utf-8"))
    assert locked["protocol"]["label"] == "protocol v3, clean single-open test"

    evaluated = evaluate_locked(tmp_path, run_id="first")
    assert evaluated.metrics["protocol"]["test_status"] == "clean_single_open"
    # Re-reading the same lock's result is harmless and byte-identical.
    assert evaluate_locked(tmp_path, run_id="first").metrics == evaluated.metrics

    with pytest.raises(ModelingError, match="already opened by run first"):
        evaluate_locked(tmp_path, run_id="second")
    assert not (second.run_dir / "official_test").exists()
    frame = pd.read_parquet(first.run_dir / "official_test" / "predictions.parquet")
    assert {"interval_low", "interval_high", "risk_band"} <= set(frame.columns)
