from __future__ import annotations

import json
import multiprocessing

import numpy as np
import pandas as pd
import pytest

from aeromaintain.evaluation.ledger import claim_holdout
from aeromaintain.evaluation.v4 import (
    acceptance,
    interval_metrics,
    paired_bootstrap,
    point_metrics,
    replay_alarms,
)
from aeromaintain.models.v4 import split_engines


def _claim_worker(path, name, ready, start, results):
    ready.put(name)
    start.wait(15)
    try:
        claim_holdout(path, "new_holdout", {"model": name})
        results.put("won")
    except ValueError:
        results.put("refused")


def test_two_processes_cannot_claim_same_holdout(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready, results, start = context.Queue(), context.Queue(), context.Event()
    path = tmp_path / "ledger.json"
    workers = [
        context.Process(target=_claim_worker, args=(path, name, ready, start, results))
        for name in ("a", "b")
    ]
    for worker in workers:
        worker.start()
    try:
        ready.get(timeout=30)
        ready.get(timeout=30)
        start.set()
        assert sorted([results.get(timeout=30), results.get(timeout=30)]) == [
            "refused",
            "won",
        ]
        for worker in workers:
            worker.join(30)
            assert worker.exitcode == 0
        assert json.loads(path.read_text())["new_holdout"]["model"] in {"a", "b"}
        assert not path.with_suffix(".json.lock").exists()
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()


def test_ledger_resume_corruption_and_atomic_failure(tmp_path, monkeypatch):
    path = tmp_path / "ledger.json"
    claim_holdout(path, "one", {"model": "a"})
    claim_holdout(path, "one", {"model": "a"})
    with pytest.raises(ValueError, match="already opened"):
        claim_holdout(path, "one", {"model": "b"})
    original = path.read_bytes()

    def fail(*args):
        raise OSError("replacement failed")

    monkeypatch.setattr("aeromaintain.evaluation.ledger.os.replace", fail)
    with pytest.raises(OSError, match="replacement"):
        claim_holdout(path, "two", {"model": "b"})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    path.write_text("[]")
    with pytest.raises(ValueError, match="object"):
        claim_holdout(path, "two", {})


def test_full_range_metrics_and_engine_paired_uncertainty():
    truth = np.array([[20.0, 200.0, 100.0], [10.0, 300.0, 110.0]])
    reference = np.array([[40.0, 125.0, 100.0], [40.0, 125.0, 100.0]])
    candidate = truth.copy()
    baseline = point_metrics(truth, reference)
    improved = point_metrics(truth, candidate)
    assert baseline["bands"][">150"] == {"sampled_cuts": 2, "engines": 1}
    assert baseline["critical_miss_rate"] == 1
    assert baseline["mean_critical_misses_per_draw"] == 1
    assert improved["rmse"] == 0
    assert acceptance(baseline, improved)["accepted"]
    intervals = paired_bootstrap(truth, reference, candidate, resamples=100)
    assert intervals["intervals_95"]["rmse"][1] < 0
    identical = paired_bootstrap(truth, reference, reference, resamples=100)
    assert identical["intervals_95"]["rmse"] == [0, 0]
    # Repeating the same cuts cannot create additional independent engines.
    repeated = paired_bootstrap(
        np.tile(truth, (3, 1)),
        np.tile(reference, (3, 1)),
        np.tile(candidate, (3, 1)),
        resamples=100,
    )
    assert repeated["intervals_95"]["rmse"] == pytest.approx(
        intervals["intervals_95"]["rmse"]
    )
    no_long = point_metrics(truth[:, [0, 2]], candidate[:, [0, 2]])
    assert not acceptance(no_long, no_long)["accepted"]


def test_interval_score_tails_and_small_groups():
    report = interval_metrics(
        np.array([0.0, 5.0, 20.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([10.0, 10.0, 10.0]),
        np.array([1, 2, 2]),
    )
    assert report["coverage"] == pytest.approx(1 / 3)
    assert report["below_lower"] == report["above_upper"] == 1
    assert report["interval_score"] == pytest.approx(9 + 20 * 11 / 3)
    assert report["engines"] == 2 and report["small_group"]
    assert interval_metrics(np.array([]), None, None, None)["cuts"] == 0


def test_replay_confirmation_reset_and_ongoing_critical_alarm():
    frame = pd.DataFrame(
        {
            "unit_id": [1] * 7 + [2] * 3,
            "cycle": [*range(1, 8), 1, 2, 4],
            "rul_true": [40, 39, 38, 37, 30, 29, 28, 2, 1, 0],
        }
    )
    report = replay_alarms(frame, np.array([10, 10, 10, 50, 10, 10, 10, 10, 10, 10]))
    assert report["records"][0]["first_confirmed_lead"] == 38
    assert report["false_events_per_engine"] == 0.5
    assert report["missed_engines"] == 1  # gap resets engine 2
    ongoing = replay_alarms(frame.iloc[:7], np.zeros(7))
    assert ongoing["missed_engines"] == 0
    assert ongoing["records"][0]["false_events"] == 1
    with pytest.raises(ValueError):
        replay_alarms(frame, np.zeros(10), persistence=0)


def test_engine_partitions_are_disjoint_and_stable():
    frame = pd.DataFrame({"unit_id": np.repeat(np.arange(100), 3)})
    split = split_engines(frame)
    assert [len(ids) for ids in split.values()] == [60, 20, 20]
    assert len({unit for ids in split.values() for unit in ids}) == 100
    assert split == split_engines(frame.sample(frac=1, random_state=7))


def test_v4_orchestration_uses_shared_cuts_and_persists_evidence(tmp_path, monkeypatch):
    from test_rul_modeling import _sensor_frame

    from aeromaintain.models import v4

    # 45 engines split 27/9/9; nine is the minimum for a finite 90% conformal rank.
    frame = _sensor_frame(list(range(1, 46)), cycles=12, targets=False)
    settings = np.array(
        [
            (0, 0, 100),
            (10, 0.25, 100),
            (20, 0.7, 60),
            (30, 0.4, 50),
            (40, 0.8, 40),
            (50, 0.9, 30),
        ]
    )
    frame[["setting_1", "setting_2", "setting_3"]] = settings[
        frame.cycle.to_numpy() % 6
    ]
    frame["rul_true"] = (12 - frame.cycle) * 20
    (tmp_path / "configs").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/protocol_v4.md").write_text("Fixture protocol")
    config = {
        "seed": 42,
        "folds": 2,
        "draws": 10,
        "bootstrap_resamples": 20,
        "critical_miss_tolerance": 0.02,
        "datasets": ["FD002", "FD004"],
        "xgboost": {},
    }
    (tmp_path / "configs/protocol_v4.json").write_text(json.dumps(config))
    for dataset in config["datasets"]:
        folder = tmp_path / "data/processed" / dataset.lower()
        folder.mkdir(parents=True)
        frame.to_parquet(folder / "train.parquet")

    def fake_fit(data, rows, spec, parameters, **kwargs):
        # Orchestration only: estimator behavior has separate real-fitting tests.
        assert len(rows) < len(data)
        pred = np.minimum(data.rul_true.to_numpy() + 5, 125)
        return pred, {"spec": spec}, 10 if spec["family"] == "xgboost" else None

    monkeypatch.setattr(v4, "fit_partition", fake_fit)
    output = tmp_path / "output"
    v4.run(tmp_path, output)
    result = json.loads((output / "evidence.json").read_text())
    assert result["chosen"] == "reference"
    for dataset in config["datasets"]:
        records = result["selection"][dataset]
        assert len(records) == 5
        assert records["reference"]["metrics"]["engines"] == 27
        assert result["evaluation"][dataset]["point"]["engines"] == 9
        assert (output / dataset / "evaluation.parquet").is_file()
    with pytest.raises(FileExistsError):
        v4.run(tmp_path, output)


def test_v4_fit_excludes_outer_validation_targets():
    from test_rul_modeling import _sensor_frame

    from aeromaintain.models.v4 import fit_partition

    frame = _sensor_frame(list(range(1, 11)), cycles=12, targets=False)
    frame["rul_true"] = (12 - frame.cycle) * 20
    rows = np.flatnonzero(frame.unit_id <= 8)
    spec = {"family": "ridge", "cap": None, "condition": False}
    pred, bundle, count = fit_partition(frame, rows, spec, {})
    changed = frame.copy()
    changed.loc[changed.unit_id > 8, "rul_true"] = 99999
    other, _, _ = fit_partition(changed, rows, spec, {})
    np.testing.assert_array_equal(pred, other)
    assert count is None
    assert bundle["normalizer"] is None
