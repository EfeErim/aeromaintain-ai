"""Write-once files inside run directories, for code outside the frozen modules.

``models/rul.py`` has private copies of these helpers and is pinned by the v5 lock,
so it cannot import from here. ``tests/test_model_boundary.py`` checks that both
copies still produce the same bytes and refuse the same overwrites.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from aeromaintain.models.rul import ModelingError


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False, engine="pyarrow")
    return buffer.getvalue()


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
