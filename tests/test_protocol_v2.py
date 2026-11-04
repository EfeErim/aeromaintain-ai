from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_rul_modeling import _sensor_frame, _small_config, _write_phase1_fixture

from aeromaintain.data.pipeline import (
    expected_members,
    load_data_contract,
    sha256_file,
)
from aeromaintain.evaluation import alert_rule_metrics, prediction_metrics
from aeromaintain.evaluation.protocol import (
    apply_mondrian_interval,
    bootstrap_official_metrics,
    draw_engine_rows,
    eligible_rows,
    fit_mondrian_interval,
    interval_report,
    leave_engines_out_interval,
    risk_threshold_table,
    select_risk_threshold,
    summarize_draws,
)
from aeromaintain.features import FEATURE_NAMES, FEATURE_SETS
from aeromaintain.models import compare_runs, run_seed_sweep, write_report
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
    load_test_status,
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


def test_development_draws_use_the_official_metric_definition() -> None:
    engines = np.array([1, 1, 2, 2])
    truth = np.array([40.0, 20.0, 80.0, 10.0])
    prediction = np.array([50.0, 25.0, 60.0, 30.0])
    plan = np.array([[0, 2], [1, 3]])

    summary, rmse_by_draw = summarize_draws(
        truth, prediction, engines, plan, critical_threshold=30
    )

    official = [
        prediction_metrics(truth[rows], prediction[rows], engines[rows])
        for rows in plan
    ]
    np.testing.assert_allclose(rmse_by_draw, [item["rmse"] for item in official])
    assert summary["engines"] == 2
    assert summary["nasa_score_motor_normalized"]["median"] == pytest.approx(
        np.median([item["nasa_score_motor_normalized"] for item in official])
    )


def test_early_stopping_engines_stay_out_of_fit_and_outer_folds() -> None:
    engines = np.repeat(np.arange(1, 11), 3)
    outer = group_fold_plan(engines, folds=5, seed=42)
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
    with pytest.raises(ModelingError, match="two training engines"):
        inner_engine_split(np.array([1, 1]), fraction=0.2, seed=1)


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


def test_feature_sets_are_ordered_subsets() -> None:
    for name, columns in FEATURE_SETS.items():
        positions = [FEATURE_NAMES.index(column) for column in columns]
        assert positions == sorted(positions), name
    assert "engine_age" not in FEATURE_SETS["no_age"]
    assert "setting_1" not in FEATURE_SETS["no_settings"]
    assert len(FEATURE_SETS["compact"]) == 64


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


def test_interval_report_splits_coverage_and_width_by_band() -> None:
    truth = np.array([10.0, 50.0, 90.0, 140.0])
    prediction = np.array([12.0, 55.0, 80.0, 130.0])
    low = prediction - 5.0
    high = prediction + 5.0

    report = interval_report(truth, prediction, low, high, edges=(30, 60, 100))

    assert report["coverage"] == 0.5
    assert report["mean_width"] == 10.0
    assert set(report["by_predicted_rul"]) == {"<=30", "31-60", "61-100", ">100"}
    assert report["by_predicted_rul"]["<=30"]["coverage"] == 1.0
    assert report["by_true_rul"]["0-30"]["rows"] == 1
    assert report["by_true_rul"][">125"]["coverage"] == 0.0


def test_risk_threshold_takes_the_smallest_meeting_recall_and_shows_costs() -> None:
    truth = np.array([10.0, 25.0, 40.0, 90.0])
    low = np.array([0.0, 20.0, 18.0, 70.0])
    plan = np.array([[0, 1, 2, 3]])

    table = risk_threshold_table(
        truth,
        low,
        plan,
        critical_threshold=30,
        thresholds=(0.0, 10.0, 20.0, 30.0),
    )

    assert [row["median_missed"] for row in table] == [1.0, 1.0, 0.0, 0.0]
    assert [row["median_false_alarms"] for row in table] == [0.0, 0.0, 1.0, 1.0]
    selected = select_risk_threshold(table, recall_target=0.95)
    assert selected["threshold"] == 20.0
    assert selected["target_met"] is True

    unreachable = select_risk_threshold(table[:2], recall_target=0.95)
    assert unreachable["threshold"] == 0.0
    assert unreachable["target_met"] is False


def test_bootstrap_intervals_contain_the_point_values_and_repeat() -> None:
    generator = np.random.default_rng(3)
    truth = generator.uniform(5, 140, size=100)
    prediction = truth + generator.normal(0, 10, size=100)
    low = np.maximum(0.0, prediction - 20)
    high = prediction + 20

    first = bootstrap_official_metrics(
        truth,
        prediction,
        low,
        high,
        critical_threshold=30,
        alert_threshold=30,
        resamples=500,
        seed=1,
    )
    second = bootstrap_official_metrics(
        truth,
        prediction,
        low,
        high,
        critical_threshold=30,
        alert_threshold=30,
        resamples=500,
        seed=1,
    )

    assert first == second
    intervals = first["intervals_95"]
    point = prediction_metrics(truth, prediction, np.arange(100))
    assert intervals["rmse"][0] <= point["rmse"] <= intervals["rmse"][1]
    rule = alert_rule_metrics(truth, low)
    low_recall = intervals["interval_low_rule_recall"]
    assert low_recall[0] <= rule["recall"] <= low_recall[1]


def test_bootstrap_reports_none_when_no_resample_has_a_critical_engine() -> None:
    truth = np.array([100.0, 120.0])
    result = bootstrap_official_metrics(
        truth,
        truth,
        truth,
        truth,
        critical_threshold=30,
        alert_threshold=30,
        resamples=10,
        seed=1,
    )

    assert result["intervals_95"]["point_rule_recall"] is None


def test_seed_sweep_and_repeat_run_comparison(tmp_path: Path) -> None:
    _write_phase1_fixture(tmp_path)
    config = _small_config()
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

    output = tmp_path / "report.json"
    write_report(output, comparison)
    write_report(output, comparison)
    with pytest.raises(ModelingError, match="Refusing to overwrite"):
        write_report(output, {"changed": True})


def test_seed_sweep_refuses_a_clean_holdout(tmp_path: Path) -> None:
    _write_phase1_fixture(tmp_path)
    train_and_lock(tmp_path, run_id="one", config=_small_config())
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


def test_sequences_are_causal_and_pad_with_the_first_row() -> None:
    frame = _sensor_frame([1, 2], cycles=6, targets=False)

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


def test_cqr_study_compares_without_opening_labels(tmp_path: Path) -> None:
    _write_phase1_fixture(tmp_path)

    def force_xgboost(ridge, xgboost, config):
        decision = _select_champion(ridge, xgboost, config)
        return {**decision, "champion": "xgboost"}

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr("aeromaintain.models.rul._select_champion", force_xgboost)
        train_and_lock(tmp_path, run_id="base", config=_small_config())
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
    _write_phase1_fixture(tmp_path)
    train_and_lock(tmp_path, run_id="base", config=_small_config())

    report = run_lstm_study(
        tmp_path,
        base_run_id="base",
        configurations=({"hidden_size": 4, "layers": 1, "dropout": 0.0},),
    )

    assert report["records"][0]["metrics"]["draws"] == 5
    assert isinstance(report["lstm_replaces_xgboost"], bool)


def test_fd003_contract_uses_its_own_members_shape_and_clean_status() -> None:
    root = Path(__file__).resolve().parents[1]

    contract = load_data_contract(root, "FD003")

    assert contract.dataset == "FD003"
    assert (contract.train_rows, contract.test_rows) == (24_720, 16_596)
    assert expected_members("FD003")["test_rul"] == "RUL_FD003.txt"
    assert load_test_status(root, "FD003") == "clean_single_open"
    assert load_test_status(root, "FD001") == "previously_seen"


def test_broken_json_is_reported_as_a_modeling_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "broken"
    run_dir.mkdir(parents=True)
    (run_dir / "model_lock.json").write_text("{not json", encoding="utf-8")
    (run_dir / "manifest.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ModelingError, match="Run manifest is unreadable"):
        evaluate_locked(tmp_path, run_id="broken")

    (run_dir / "manifest.json").write_text(
        json.dumps({"model_lock_sha256": sha256_file(run_dir / "model_lock.json")}),
        encoding="utf-8",
    )
    with pytest.raises(ModelingError, match="Model lock is unreadable"):
        evaluate_locked(tmp_path, run_id="broken")

    (run_dir / "model_lock.json").write_text(
        json.dumps({"run_id": "broken", "schema_version": 2}),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"model_lock_sha256": sha256_file(run_dir / "model_lock.json")}),
        encoding="utf-8",
    )
    with pytest.raises(ModelingError, match="structure is invalid"):
        evaluate_locked(tmp_path, run_id="broken")


def test_clean_holdout_opens_for_one_lock_only(tmp_path: Path) -> None:
    _write_phase1_fixture(tmp_path)
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

    first = train_and_lock(tmp_path, run_id="first", config=_small_config())
    second = train_and_lock(tmp_path, run_id="second", config=_small_config())
    locked = json.loads(first.model_lock.read_text(encoding="utf-8"))
    assert locked["protocol"]["label"] == "protocol v2, clean single-open test"

    evaluated = evaluate_locked(tmp_path, run_id="first")
    assert evaluated.metrics["protocol"]["test_status"] == "clean_single_open"
    # Re-reading the same lock's result is harmless and byte-identical.
    assert evaluate_locked(tmp_path, run_id="first").metrics == evaluated.metrics

    with pytest.raises(ModelingError, match="already opened by run first"):
        evaluate_locked(tmp_path, run_id="second")
    assert not (second.run_dir / "official_test").exists()
    frame = pd.read_parquet(first.run_dir / "official_test" / "predictions.parquet")
    assert {"interval_low", "interval_high", "risk_band"} <= set(frame.columns)
