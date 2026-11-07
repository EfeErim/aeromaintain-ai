import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aeromaintain.evaluation.v4 import replay_alarms
from aeromaintain.models import v5

ROOT = Path(__file__).resolve().parents[1]


def test_hybrid_switches_at_threshold():
    reference = np.array([10.0, 114.9, 115.0, 126.0])
    uncapped = np.array([50.0, 50.0, 160.0, 180.0])
    assert v5.hybrid_prediction(reference, uncapped, 115).tolist() == [
        10.0,
        114.9,
        160.0,
        180.0,
    ]


def _engine(values, rul):
    frame = pd.DataFrame(
        {"unit_id": 1, "cycle": np.arange(len(rul)), "rul_true": np.asarray(rul)}
    )
    return frame, np.asarray(values, dtype=float)


def test_late_alarm_is_not_timely():
    rul = np.arange(40, -1, -1)
    # The prediction only drops below 30 when five cycles remain.
    frame, values = _engine(np.where(rul <= 5, 0, 80), rul)
    assert replay_alarms(frame, values)["missed_engines"] == 0
    assert replay_alarms(frame, values, min_rul=10)["missed_engines"] == 1


def test_timely_alarm_is_detected():
    rul = np.arange(40, -1, -1)
    frame, values = _engine(np.where(rul <= 25, 0, 80), rul)
    assert replay_alarms(frame, values, min_rul=10)["missed_engines"] == 0


def _synthetic(engines=12, cycles=30):
    rng = np.random.default_rng(0)
    ids = np.repeat(np.arange(engines), cycles)
    rul = np.tile(np.arange(cycles)[::-1], engines).astype(float)
    x = np.column_stack([rul + rng.normal(0, 2, len(rul)), rng.normal(size=len(rul))])
    return x, rul, ids


@pytest.mark.parametrize("name", v5.CANDIDATES)
def test_candidates_fit(name):
    x, y, engines = _synthetic()
    rows = np.arange(len(y))
    params = json.loads((ROOT / v5.CONFIG).read_text())["xgboost"]
    model, count = v5.fit_candidate(name, x, y, engines, rows, params, seed=42)
    assert (count is None) == (name == "ridge")
    assert np.sqrt(np.mean((v5.predict(model, x) - y) ** 2)) < 5


def test_split_margin_needs_enough_engines():
    margin, rank = v5.split_margin(np.zeros(64), np.arange(64.0), 0.9)
    assert rank == 59 and margin == 58.0
    with pytest.raises(ValueError):
        v5.split_margin(np.zeros(5), np.zeros(5), 0.9)


def test_gates():
    gates = {"min_mondrian_coverage": 0.85, "max_untimely_fraction": 0.1}
    replay = {"missed_fraction": 3 / 35}
    result = v5.final_gates({"rmse": 10}, {"rmse": 12}, 0.85, replay, gates)
    assert result["passed"]
    result = v5.final_gates(
        {"rmse": 10}, {"rmse": 12}, 0.85, {"missed_fraction": 4 / 35}, gates
    )
    assert not result["passed"]
    assert not result["checks"]["G3_timely_point_alarm"]


def test_reserved_engines_are_namespaced():
    reservation = json.loads((ROOT / v5.RESERVATION).read_text())
    config = json.loads((ROOT / v5.CONFIG).read_text())
    engines = v5.reserved_test_engines(reservation, config["test_files"])
    flat = [e for parts in engines.values() for split in parts.values() for e in split]
    assert len(flat) == len(set(flat)) == v5.TEST_ENGINES
    assert min(flat) > 10_000
    assert not set(config["training_files"]) & set(config["test_files"])


def test_evaluate_refuses_uncommitted_lock(tmp_path, monkeypatch):
    output = tmp_path / "final"
    output.mkdir()
    (output / "model_lock.json").write_bytes(b'{"lock": 1}\n')
    monkeypatch.setattr(v5, "require_clean_tree", lambda root: "abc")
    monkeypatch.setattr(v5, "_git", lambda root, *args: b'{"lock": 2}\n')
    with pytest.raises(RuntimeError, match="committed"):
        v5.evaluate(tmp_path, output, {})
    assert not (tmp_path / v5.LEDGER).exists()


def test_engine_bootstrap_bounds():
    values = np.array([1.0, 0.0, 1.0, 1.0])
    result = v5.engine_bootstrap({"rate": (values, lambda v: v)}, 200, seed=1)
    low, high = result["rate"]
    assert 0 <= low <= high <= 1
