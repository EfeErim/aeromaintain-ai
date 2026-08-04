"""Runtime provenance recorded with reproducible AeroMaintain runs."""

from __future__ import annotations

import platform
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

from aeromaintain.data.pipeline import sha256_file

CONSTRAINT_PATH = Path("constraints/python311-tested.txt")


def _installed_distributions() -> dict[str, str]:
    observed = {
        name: distribution.version
        for distribution in distributions()
        if (name := distribution.metadata.get("Name"))
    }
    return dict(sorted(observed.items(), key=lambda item: item[0].casefold()))


def runtime_fingerprint(project_root: Path) -> dict[str, Any]:
    """Return interpreter, platform, and installed distribution provenance."""
    constraint_path = project_root / CONSTRAINT_PATH
    if not constraint_path.is_file():
        raise FileNotFoundError(
            f"Tested dependency constraints are missing: {constraint_path}"
        )
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "packages": _installed_distributions(),
        "constraints": {
            "path": CONSTRAINT_PATH.as_posix(),
            "sha256": sha256_file(constraint_path),
        },
    }
