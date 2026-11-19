"""File hashing, canonical bytes and write-once helpers.

Only the standard library and pandas, so the review app and the hosted demo can use
them without importing the modeling stack.

``models/rul.py`` keeps private copies of ``json_bytes``, ``parquet_bytes`` and the
write-once rule, and is pinned by the v5 lock, so it cannot import from here.
``tests/test_model_boundary.py`` checks that both copies produce the same bytes and
refuse the same overwrites.
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

SENSOR_COLUMNS = tuple(f"sensor_{index}" for index in range(1, 22))
# Written into every project root that ``aeromaintain pipeline --synthetic`` creates.
SYNTHETIC_MARKER = "SYNTHETIC_DATA.txt"


def sha256_file(path: Path) -> str:
    """Return a file's lowercase SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, engine="pyarrow")
    return buffer.getvalue()


def processed_dir(project_root: Path, dataset: str) -> Path:
    return project_root / "data" / "processed" / dataset.lower()


def write_replacing(path: Path, payload: bytes) -> None:
    """Replace ``path`` through a temporary file in the same directory."""
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}-", suffix=".tmp", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_once(path: Path, payload: bytes, error: type[Exception]) -> None:
    """Write ``payload`` once; accept identical existing bytes, refuse different ones.

    An identical file is what a resumed or repeated step finds, so it is accepted.
    A different one raises ``error``, the caller's domain error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise error(f"Refusing to overwrite a file with different content: {path}")
    write_replacing(path, payload)
