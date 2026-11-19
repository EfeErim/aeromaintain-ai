"""Project-level path and configuration conventions."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT_ENV = "AEROMAINTAIN_PROJECT_ROOT"
# Read by the review app: a local run by ID (set by ``aeromaintain app``), or the
# row-free public bundle (set by ``streamlit_app.py`` for the hosted demo).
RUN_ID_ENV = "AEROMAINTAIN_RUN_ID"
PUBLIC_BUNDLE_ENV = "AEROMAINTAIN_PUBLIC_BUNDLE"


def resolve_project_root(project_root: Path | None = None) -> Path:
    """Resolve the local project root used by CLI commands."""
    if project_root is not None:
        return project_root.expanduser().resolve()

    configured_root = os.environ.get(PROJECT_ROOT_ENV)
    if configured_root:
        return Path(configured_root).expanduser().resolve()

    return Path.cwd().resolve()
