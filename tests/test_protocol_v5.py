import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aeromaintain.evaluation.v4 import replay_alarms
from aeromaintain.models import v5

ROOT = Path(__file__).resolve().parents[1]


def _engine(values, rul):
    frame = pd.DataFrame(
        {"unit_id": 1, "cycle": np.arange(len(rul)), "rul_true": np.asarray(rul)}
    )
    return frame, np.asarray(values, dtype=float)


def test_alarm_counts_as_timely_only_with_ten_cycles_left():
    rul = np.arange(40, -1, -1)
    late, values = _engine(np.where(rul <= 5, 0, 80), rul)
    assert replay_alarms(late, values)["missed_engines"] == 0
    assert replay_alarms(late, values, min_rul=10)["missed_engines"] == 1
    timely, values = _engine(np.where(rul <= 25, 0, 80), rul)
    assert replay_alarms(timely, values, min_rul=10)["missed_engines"] == 0


def test_reserved_engines_are_namespaced():
    reservation = json.loads((ROOT / v5.RESERVATION).read_text())
    config = json.loads((ROOT / v5.CONFIG).read_text())
    engines = v5.reserved_test_engines(reservation, config["test_files"])
    flat = [e for parts in engines.values() for split in parts.values() for e in split]
    assert len(flat) == len(set(flat)) == v5.TEST_ENGINES
    assert min(flat) > 10_000
    assert not set(config["training_files"]) & set(config["test_files"])


def _lock():
    return json.loads((ROOT / v5.PUBLISHED_LOCK).read_text(encoding="utf-8"))


def test_hybrid_switches_at_threshold():
    reference = np.array([10.0, 114.9, 115.0, 126.0])
    uncapped = np.array([50.0, 50.0, 160.0, 180.0])
    assert v5.hybrid_prediction(reference, uncapped, 115).tolist() == [
        10.0,
        114.9,
        160.0,
        180.0,
    ]


@pytest.mark.parametrize(
    ("missed", "passed"), [(0, True), (3, True), (4, False), (35, False)]
)
def test_alarm_gate_allows_at_most_three_of_35(missed, passed):
    point = {"rmse": 13.0}
    baseline = {"rmse": 22.0}
    replay = {"missed_fraction": missed / v5.TEST_ENGINES}
    result = v5.final_gates(point, baseline, 0.9, replay, _lock()["gates"])
    assert result["checks"]["G3_timely_point_alarm"] is passed


def test_published_lock_still_matches_frozen_files():
    lock = _lock()
    assert v5.source_hashes(ROOT) == lock["source_sha256"]
    for key, name in [
        ("protocol_sha256", v5.PROTOCOL),
        ("config_sha256", v5.CONFIG),
        ("reservation_sha256", v5.RESERVATION),
        ("development_sha256", v5.DEVELOPMENT),
    ]:
        assert v5.sha256_file(ROOT / name) == lock[key], name


RELEASE_TAG = "v1.0.0"
LF_NORMALIZED_AT_TAG = {v5.DEVELOPMENT}


def _tagged_bytes(path: str) -> bytes | None:
    """Return a file's bytes at the release tag, or None when the tag is absent."""
    result = subprocess.run(
        ["git", "show", f"{RELEASE_TAG}:{path}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def test_published_lock_matches_release_tag():
    """The published tag, not only the working tree, holds the code the lock names.

    This is the provenance check: it keeps holding even if a pinned file is later
    refactored on main. CI fetches tags, so it never skips there.
    """
    lock = _lock()
    expected = dict(lock["source_sha256"])
    for key, name in [
        ("protocol_sha256", v5.PROTOCOL),
        ("config_sha256", v5.CONFIG),
        ("reservation_sha256", v5.RESERVATION),
        ("development_sha256", v5.DEVELOPMENT),
    ]:
        expected[name] = lock[key]
    tagged_lock = _tagged_bytes(v5.PUBLISHED_LOCK)
    if tagged_lock is None:
        if os.environ.get("CI"):
            pytest.fail(f"Release tag {RELEASE_TAG} is not available in CI")
        pytest.skip(f"Release tag {RELEASE_TAG} is not available")
    assert (
        hashlib.sha256(tagged_lock).hexdigest()
        == hashlib.sha256((ROOT / v5.PUBLISHED_LOCK).read_bytes()).hexdigest()
    )
    for name, digest in expected.items():
        content = _tagged_bytes(name)
        assert content is not None, name
        if name in LF_NORMALIZED_AT_TAG:
            # The lock hashed CRLF bytes; the tag stored them LF-normalized. Fixed in
            # ab390b5 with no content change: the tag must equal the locked bytes
            # with only line endings normalized.
            locked = (ROOT / name).read_bytes()
            assert hashlib.sha256(locked).hexdigest() == digest, name
            assert content == locked.replace(b"\r\n", b"\n"), name
            continue
        assert hashlib.sha256(content).hexdigest() == digest, name


def test_gates_fail_on_each_threshold():
    gates = _lock()["gates"]
    replay = {"missed_fraction": 0.0}
    for missed, passed in [(3, True), (4, False)]:
        late = {"missed_fraction": missed / v5.TEST_ENGINES}
        result = v5.final_gates({"rmse": 1.0}, {"rmse": 2.0}, 0.9, late, gates)
        assert result["checks"]["G3_timely_point_alarm"] is passed
    assert v5.final_gates({"rmse": 1.0}, {"rmse": 1.0}, 0.85, replay, gates)["passed"]
    worse = v5.final_gates({"rmse": 1.01}, {"rmse": 1.0}, 0.9, replay, gates)
    assert worse["checks"] == {
        "G1_rmse_not_worse_than_baseline": False,
        "G2_mondrian_coverage": True,
        "G3_timely_point_alarm": True,
    }
    assert not v5.final_gates({"rmse": 1.0}, {"rmse": 2.0}, 0.849, replay, gates)[
        "passed"
    ]
    no_critical = {"missed_fraction": None}
    assert not v5.final_gates({"rmse": 1.0}, {"rmse": 2.0}, 0.9, no_critical, gates)[
        "passed"
    ]


def test_published_gate_decisions_follow_from_published_numbers():
    final = json.loads(
        (ROOT / "docs/protocol_v5_evidence.json").read_text(encoding="utf-8")
    )["part_b_final_test"]
    replay = final["replay"]["point_timely"]
    assert replay["missed_engines"] == 3
    recomputed = v5.final_gates(
        final["point"],
        final["baseline"],
        final["intervals"]["mondrian"]["overall"]["coverage"],
        replay,
        _lock()["gates"],
    )
    assert recomputed == final["gates"]
