from __future__ import annotations

import pytest

from aeromaintain import synthetic


@pytest.fixture(scope="session")
def synthetic_archive() -> bytes:
    """The generated FD001-shaped archive, built once per test session."""
    return synthetic.synthetic_archive_bytes()


@pytest.fixture
def reuse_synthetic_archive(monkeypatch, synthetic_archive: bytes) -> bytes:
    """Make ``build_synthetic_project`` reuse the session archive."""
    monkeypatch.setattr(
        synthetic, "synthetic_archive_bytes", lambda seed=7: synthetic_archive
    )
    return synthetic_archive
