"""Write-once files inside run directories, for code outside the frozen modules.

``models/rul.py`` has private copies of these helpers (and of ``json_bytes`` and
``parquet_bytes`` in ``data/pipeline.py``) and is pinned by the v5 lock, so it cannot
import from here. ``tests/test_model_boundary.py`` checks that both copies still
produce the same bytes and refuse the same overwrites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aeromaintain.data.pipeline import json_bytes
from aeromaintain.models.rul import ModelingError


def write_new_bytes(path: Path, payload: bytes) -> None:
    """Write once; an identical existing file is accepted, a different one refused."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise ModelingError(f"Refusing to overwrite different artifact: {path}")
    path.write_bytes(payload)


def write_json(path: Path, value: Any) -> None:
    write_new_bytes(path, json_bytes(value))


def processed_dir(project_root: Path, dataset: str) -> Path:
    return project_root / "data" / "processed" / dataset.lower()
