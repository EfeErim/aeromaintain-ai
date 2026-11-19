from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cmapss_fixture import small_config, write_prepared_fixture

from aeromaintain.files import sha256_file
from aeromaintain.models import compare_runs, run_seed_sweep
from aeromaintain.models.escalation import run_cqr_study, run_lstm_study
from aeromaintain.models.evaluation import evaluate_locked
from aeromaintain.models.rul import (
    ModelingConfig,
    ModelingError,
    _select_champion,
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
