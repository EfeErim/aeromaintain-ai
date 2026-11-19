"""A synthetic stand-in for the NASA archive, so the pipeline runs without the data.

``build_synthetic_project`` writes a separate project root holding a fake
``CMAPSSData.zip`` shaped exactly like FD001 (same row and engine counts), plus copies
of the configuration with the archive hash pointing at that fake file and lighter
training settings. The normal pipeline then runs on it unchanged: verification,
splits, training, the model lock, the single label opening and the report.

The numbers it produces say nothing about turbofan engines. The project root is
marked as synthetic and must be outside the repository, so synthetic runs can never
mix with the real ``runs/`` directory.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import zipfile
from pathlib import Path

import numpy as np
import yaml

from aeromaintain.data.pipeline import ARCHIVE_NAME, DATASET_SHAPES
from aeromaintain.files import SYNTHETIC_MARKER as MARKER

DATASET = "FD001"
# Lighter search settings so the demo finishes in a few minutes on a laptop CPU.
FAST_MODELING = {
    "xgboost_candidates": 2,
    "xgboost_n_jobs": 2,
    "scoring_draws": 20,
    "sensitivity_windows": [None],
    "cross_fit_folds": 5,
    "rul_cap_candidates": [125],
}


# Operating settings: two small jitters and a constant third setting, as in FD001.
SETTING_OFFSET = np.array([0.0, 0.0, 100.0])
SETTING_NOISE = np.array([0.002, 0.0003, 0.0])
SENSOR_NOISE = 0.5
ROW_FORMAT = "%d %d " + " ".join(["%.4f"] * 24)


class SyntheticProjectError(RuntimeError):
    """Raised when a synthetic project cannot be created safely."""


def _lengths(rng: np.random.Generator, count: int, total: int, low: int, high: int):
    """Draw ``count`` integer lengths in ``[low, high]`` that sum to ``total``."""
    lengths = rng.integers(low, high + 1, size=count)
    while (difference := total - int(lengths.sum())) != 0:
        step = 1 if difference > 0 else -1
        room = lengths < high if step > 0 else lengths > low
        index = rng.choice(np.flatnonzero(room), size=min(abs(difference), room.sum()))
        np.add.at(lengths, index, step)
        lengths = np.clip(lengths, low, high)
    return lengths


def _engine_rows(
    rng: np.random.Generator, unit: int, life: int, observed: int, wear: np.ndarray
) -> list[str]:
    """Rows for one engine: three settings and 21 sensors drifting with damage."""
    base = 500.0 + 40.0 * np.arange(21) + rng.normal(0, 2.0, 21)
    # One draw per cycle of three setting and 21 sensor values, in that order.
    noise = rng.standard_normal((observed, 24))
    cycles = np.arange(1, observed + 1)
    damage = (cycles / life) ** 2.5
    settings = SETTING_OFFSET + SETTING_NOISE * noise[:, :3]
    sensors = base + wear * damage[:, None] * 10.0 + SENSOR_NOISE * noise[:, 3:]
    return [
        ROW_FORMAT % (unit, cycle, *setting, *sensor)
        for cycle, setting, sensor in zip(
            cycles.tolist(), settings.tolist(), sensors.tolist(), strict=True
        )
    ]


def synthetic_archive_bytes(seed: int = 7) -> bytes:
    """Return a zip shaped like the FD001 members of the NASA archive."""
    shape = DATASET_SHAPES[DATASET]
    rng = np.random.default_rng(seed)
    # Some sensors carry no signal, as in the real data.
    wear = rng.normal(0.0, 1.0, 21) * (rng.random(21) > 0.3)

    lives = _lengths(rng, shape["train_engines"], shape["train_rows"], 128, 362)
    train = [
        row
        for unit, life in enumerate(lives, start=1)
        for row in _engine_rows(rng, unit, int(life), int(life), wear)
    ]

    observed = _lengths(rng, shape["test_engines"], shape["test_rows"], 31, 303)
    remaining = rng.integers(7, 146, size=shape["test_engines"])
    test = [
        row
        for unit, (seen, left) in enumerate(zip(observed, remaining, strict=True), 1)
        for row in _engine_rows(rng, unit, int(seen + left), int(seen), wear)
    ]

    # Fixed member timestamps keep the archive bytes, and so its hash, reproducible.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, lines in (
            (f"train_{DATASET}.txt", train),
            (f"test_{DATASET}.txt", test),
            (f"RUL_{DATASET}.txt", [str(int(value)) for value in remaining]),
        ):
            info = zipfile.ZipInfo(
                f"CMAPSSData/{name}", date_time=(2026, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, ("\n".join(lines) + "\n").encode("ascii"))
    return buffer.getvalue()


def build_synthetic_project(repo_root: Path, target: Path) -> Path:
    """Create a synthetic project root at ``target`` and return the archive path."""
    repo_root = repo_root.resolve()
    target = target.resolve()
    if target == repo_root or repo_root in target.parents:
        raise SyntheticProjectError(
            "The synthetic project must be outside the repository, so its runs never "
            "mix with real ones"
        )
    if target.exists() and any(target.iterdir()) and not (target / MARKER).is_file():
        raise SyntheticProjectError(
            f"{target} is not empty and is not a synthetic project"
        )
    config_path = repo_root / "configs" / "project.yaml"
    constraints = repo_root / "constraints" / "python311-tested.txt"
    if not config_path.is_file() or not constraints.is_file():
        raise SyntheticProjectError(
            "Run from the repository root (configs/ and constraints/ are needed)"
        )

    archive = synthetic_archive_bytes()
    (target / "configs").mkdir(parents=True, exist_ok=True)
    (target / "constraints").mkdir(exist_ok=True)
    (target / MARKER).write_text(
        "Every file under this directory comes from synthetic data generated by\n"
        "aeromaintain.synthetic. None of it describes real or NASA engines.\n",
        encoding="utf-8",
    )
    shutil.copyfile(constraints, target / "constraints" / constraints.name)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["project"]["synthetic_data"] = True
    config["data"]["source_url"] = "synthetic://aeromaintain"
    config["data"]["archive_size_bytes"] = len(archive)
    config["data"]["archive_sha256"] = hashlib.sha256(archive).hexdigest()
    config["modeling"].update(FAST_MODELING)
    (target / "configs" / "project.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    archive_path = target / "synthetic" / ARCHIVE_NAME
    archive_path.parent.mkdir(exist_ok=True)
    archive_path.write_bytes(archive)
    return archive_path
