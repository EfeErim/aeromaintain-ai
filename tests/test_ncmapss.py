"""Small HDF5 contracts: no simulator-state inputs and label opening after lock."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aeromaintain.data import ncmapss


def make_hdf(path, *, units=4):
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        for name, values in [
            ("A_var", ("unit", "cycle", "Fc", "hs")),
            ("X_s_var", ncmapss.MEASURED_CHANNELS),
            ("W_var", ncmapss.OPERATING_CHANNELS),
        ]:
            handle[name] = np.asarray(values, dtype="S")
        for split, ids in [("dev", range(1, units)), ("test", [units])]:
            rows = [
                (unit, cycle, sample)
                for unit in ids
                for cycle in range(1, 7)
                for sample in range(3)
            ]
            aux = np.array([[unit, cycle, 999, 999] for unit, cycle, _ in rows])
            sensor = np.array(
                [
                    [unit + cycle + sample + j for j in range(14)]
                    for unit, cycle, sample in rows
                ],
                dtype=float,
            )
            settings = np.array(
                [[cycle + sample + j for j in range(4)] for _, cycle, sample in rows],
                dtype=float,
            )
            handle[f"A_{split}"] = aux
            handle[f"X_s_{split}"] = sensor
            handle[f"W_{split}"] = settings
            handle[f"Y_{split}"] = np.array([[6 - cycle] for _, cycle, _ in rows])
            # X_v and T deliberately absent: the pipeline must never access them.


def test_measured_cycle_aggregation_is_chunk_invariant(tmp_path):
    path = tmp_path / "test.h5"
    make_hdf(path)
    small = ncmapss.cycle_features(path, chunk_size=4)
    big = ncmapss.cycle_features(path, chunk_size=100)
    pd.testing.assert_frame_equal(small, big, atol=1e-10)
    assert len(small) == 24
    assert small.loc[0, "sensor_0_mean"] == 3
    assert small.loc[0, "sensor_0_std"] == pytest.approx(np.std([2, 3, 4]))
    assert not any(
        name.startswith(("Y", "X_v", "T_", "hs", "Fc")) for name in small.columns
    )
    assert ncmapss.read_targets(path, small.iloc[[0, 5]]).tolist() == [5, 0]
    bad = small.iloc[[0]].copy()
    bad["cycle"] = 999
    with pytest.raises(ValueError, match="no HDF5"):
        ncmapss.read_targets(path, bad)


def test_hdf_schema_rejects_channel_reordering(tmp_path):
    path = tmp_path / "bad.h5"
    make_hdf(path)
    with ncmapss._h5py().File(path, "r+") as handle:
        handle["W_var"][0] = b"bad"
    with pytest.raises(ValueError, match="channel"):
        ncmapss.cycle_features(path)


def test_pilot_publishes_lock_before_evaluation_labels(tmp_path, monkeypatch):
    path = tmp_path / "ds02.h5"
    make_hdf(path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/protocol_v4.md").write_text("Frozen fixture protocol")
    original = ncmapss.read_targets

    def checked(source, requested):
        if requested.source_split.eq("test").any():
            assert (tmp_path / "docs/ncmapss_locks/ds02.json").exists()
            ledger = json.loads(
                (tmp_path / "runs/ncmapss_holdout_ledger.json").read_text()
            )
            assert "DS02" in ledger
        return original(source, requested)

    monkeypatch.setattr(ncmapss, "read_targets", checked)
    report = ncmapss.run_pilot([path], tmp_path / "run", tmp_path)
    assert report["point"]["engines"] == 1
    assert len(report["lock"]["features"]) == 72
    assert report["lock"]["evaluation_engines"] == [4]
    assert not set(report["lock"]["features"]) & {"unit_id", "cycle", "archive_unit_id"}


def test_multiple_files_namespace_engines_and_require_all_sources():
    requested = pd.DataFrame({"source_file": ["missing.h5"], "archive_unit_id": [1]})
    with pytest.raises(ValueError, match="Missing source"):
        ncmapss.read_multiple_targets([Path("another.h5")], requested)
