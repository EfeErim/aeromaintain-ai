"""A tiny synthetic N-CMAPSS project for running v5 end to end.

The real archive is several gigabytes and its test labels are reserved, so the v5
flow is exercised on small HDF5 files with the same layout, inside a temporary git
repository that holds byte-identical copies of the frozen v5 sources.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from aeromaintain.data.ncmapss import MEASURED_CHANNELS, OPERATING_CHANNELS
from aeromaintain.data.pipeline import sha256_file
from aeromaintain.models import v5

REPO = Path(__file__).resolve().parents[1]
# 64 training engines over six files, as v5 requires.
TRAINING_SPLITS = {
    "N-CMAPSS_DS02-006.h5": (6, 5),
    "N-CMAPSS_DS03-012.h5": (6, 5),
    "N-CMAPSS_DS04.h5": (6, 5),
    "N-CMAPSS_DS05.h5": (6, 5),
    "N-CMAPSS_DS06.h5": (5, 5),
    "N-CMAPSS_DS07.h5": (5, 5),
}
# 35 test engines, split into dev and test units like the real reservation.
TEST_SPLITS = {
    "N-CMAPSS_DS01-005.h5": (6, 4),
    "N-CMAPSS_DS08a-009.h5": (9, 6),
    "N-CMAPSS_DS08c-008.h5": (6, 4),
}
ROWS_PER_CYCLE = 2


def _strings(values) -> np.ndarray:
    return np.array([value.encode() for value in values])


def write_h5(path: Path, dev_units: int, test_units: int, seed: int) -> None:
    import h5py

    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as handle:
        handle["X_s_var"] = _strings(MEASURED_CHANNELS)
        handle["W_var"] = _strings(OPERATING_CHANNELS)
        handle["A_var"] = _strings(["unit", "cycle", "Fc", "hs"])
        first = {"dev": 1, "test": dev_units + 1}
        counts = {"dev": dev_units, "test": test_units}
        for split in ("dev", "test"):
            xs, w, a, y = [], [], [], []
            for unit in range(first[split], first[split] + counts[split]):
                life = int(rng.integers(30, 55))
                for cycle in range(1, life + 1):
                    rul = life - cycle
                    for _ in range(ROWS_PER_CYCLE):
                        drift = rul / 50.0
                        xs.append(rng.normal(0, 0.05, 14) + drift * np.arange(1, 15))
                        w.append(rng.normal(0, 1, 4))
                        a.append([unit, cycle, 1, 0])
                        y.append([rul])
            handle[f"X_s_{split}"] = np.asarray(xs)
            handle[f"W_{split}"] = np.asarray(w)
            handle[f"A_{split}"] = np.asarray(a, dtype=float)
            handle[f"Y_{split}"] = np.asarray(y, dtype=float)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def commit_all(root: Path, message: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)


def build_project(root: Path) -> dict:
    """Create the project, commit it, and return the v5 config to use."""
    for name in v5.SOURCES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    for name in (v5.CONFIG, v5.PROTOCOL):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, root / name)
    config = json.loads((root / v5.CONFIG).read_text(encoding="utf-8"))

    raw = root / config["raw_dir"]
    raw.mkdir(parents=True)
    for offset, (name, (dev, test)) in enumerate(TRAINING_SPLITS.items()):
        write_h5(raw / name, dev, test, seed=offset)
    reservation = {"files": {}}
    for offset, (name, (dev, test)) in enumerate(TEST_SPLITS.items()):
        write_h5(raw / name, dev, test, seed=100 + offset)
        reservation["files"][name] = {
            "archive_dev_units": list(range(1, dev + 1)),
            "archive_test_units": list(range(dev + 1, dev + test + 1)),
            "engines": dev + test,
            "sha256": sha256_file(raw / name),
        }
    reservation_path = root / v5.RESERVATION
    reservation_path.parent.mkdir(parents=True, exist_ok=True)
    reservation_path.write_text(json.dumps(reservation, indent=2), encoding="utf-8")
    (root / ".gitignore").write_text("data/\nruns/\n", encoding="utf-8")

    _git(root, "init", "-q")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "fixture")
    # Locks hash exact bytes; line-ending conversion would break them.
    _git(root, "config", "core.autocrlf", "false")
    commit_all(root, "fixture project")
    return config
