from pathlib import Path

import pytest

from aeromaintain import __version__
from aeromaintain.runtime import runtime_fingerprint


def test_package_has_a_version() -> None:
    assert __version__ == "0.1.0"


def test_runtime_fingerprint_requires_tested_constraints(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="constraints are missing"):
        runtime_fingerprint(tmp_path)
