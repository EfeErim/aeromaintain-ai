"""Hosted demo entrypoint: renders the row-free public bundle in ``demo/``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("AEROMAINTAIN_PUBLIC_BUNDLE", str(ROOT / "demo"))

from aeromaintain.app.main import main  # noqa: E402

main()
