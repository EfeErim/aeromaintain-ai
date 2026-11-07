"""Cross-process exclusive claims with crash-safe ledger replacement."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def claim_holdout(path: Path, dataset: str, entry: dict[str, Any]) -> None:
    """Claim once; an existing identical claim may resume. Fail closed on contention.

    The exclusive sidecar intentionally survives a process crash. Its removal requires
    an operator to verify no claimant is active; elapsed time never grants access.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    guard = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError("Holdout ledger is locked by another claimant") from error
    temporary = None
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(str(os.getpid()))
            handle.flush()
            os.fsync(handle.fileno())
        ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(ledger, dict):
            raise ValueError("Holdout ledger must contain an object")
        if dataset in ledger:
            if ledger[dataset] != entry:
                owner = ledger[dataset].get("run_id", "another model")
                raise ValueError(
                    f"{dataset} test labels were already opened by run {owner}"
                )
            return
        ledger[dataset] = entry
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(ledger, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        guard.unlink()
