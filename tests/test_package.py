import json
from pathlib import Path

import pytest

from aeromaintain import __version__
from aeromaintain.runtime import runtime_fingerprint


def test_package_has_a_version() -> None:
    assert __version__ == "0.1.0"


def test_runtime_fingerprint_requires_tested_constraints(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="constraints are missing"):
        runtime_fingerprint(tmp_path)


def test_public_reference_evidence_is_sanitized_and_self_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads(
        (root / "docs" / "reference_evidence.json").read_text(encoding="utf-8")
    )

    assert evidence["schema_version"] == 2
    assert evidence["run"]["status"] == "pipeline_complete"
    assert evidence["run"]["stages"] == [
        "prepare",
        "train_and_lock",
        "evaluate_locked",
        "report",
    ]
    assert evidence["development_model_comparison"]["champion"] == "ridge"
    assert evidence["official_test"]["point_threshold_critical"]["recall"] == 0.48
    assert evidence["calibration"]["risk_band_source"] == "interval_low"
    assert evidence["active_scope"] == "rul_evaluation_only"
    serialized = json.dumps(evidence).casefold()
    assert "maintenance_policy_comparison" not in serialized
    assert "capacity_sensitivity" not in serialized
    assert "total_synthetic_cost" not in serialized
