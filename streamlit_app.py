"""Hosted demo entrypoint: renders the row-free public bundle in ``demo/``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from aeromaintain.app.main import main  # noqa: E402
from aeromaintain.config import PUBLIC_BUNDLE_ENV  # noqa: E402

os.environ.setdefault(PUBLIC_BUNDLE_ENV, str(ROOT / "demo"))

main()
