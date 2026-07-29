"""Project-level path and configuration conventions."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT_ENV = "AEROMAINTAIN_PROJECT_ROOT"
REQUIRED_CONFIG_FILES = ("project.yaml", "scenario.yaml")
REQUIRED_LOCAL_DATA_DIRS = ("data/raw", "data/processed", "artifacts")


def resolve_project_root(project_root: Path | None = None) -> Path:
    """Resolve the local project root used by CLI commands."""
    if project_root is not None:
        return project_root.expanduser().resolve()

    configured_root = os.environ.get(PROJECT_ROOT_ENV)
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    return Path.cwd().resolve()
