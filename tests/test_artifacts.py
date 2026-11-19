"""Contracts and hash checks for public bundles and verified runs."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest
from run_fixture import build_verified_run, write_json

from aeromaintain import artifacts as app_artifacts
from aeromaintain.artifacts import (
    ArtifactValidationError,
    export_public_bundle,
    load_public_bundle,
    load_verified_run,
)
from aeromaintain.data.pipeline import sha256_file

ROOT = Path(__file__).resolve().parents[1]
FAKE_HASH = "0" * 64


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    target = tmp_path / "bundle"
    shutil.copytree(ROOT / "demo", target)
    return target


def _manifest(bundle: Path) -> dict:
    return json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))


def _set_artifacts(bundle: Path, artifacts) -> None:
    write_json(bundle / "manifest.json", {"artifacts": artifacts})


def _relist(bundle: Path) -> None:
    """Rehash the listed files, as someone tampering with a bundle would."""
    _set_artifacts(
        bundle,
        {name: sha256_file(bundle / name) for name in ("run.json", "risk_ranking.csv")},
    )


def test_tracked_demo_bundle_loads_unchanged(bundle: Path) -> None:
    before = {path.name: sha256_file(path) for path in (ROOT / "demo").iterdir()}
    loaded = load_public_bundle(ROOT / "demo")
    assert loaded.run_id == "fd004-v3-clean"
    assert list(loaded.risk_ranking.columns) == list(
        app_artifacts.SAFE_PREDICTION_COLUMNS
    )
    assert loaded.sensor_history.empty
    after = {path.name: sha256_file(path) for path in (ROOT / "demo").iterdir()}
    assert before == after


@pytest.mark.parametrize("missing", ["run.json", "risk_ranking.csv"])
def test_each_required_entry_must_be_listed(bundle: Path, missing: str) -> None:
    artifacts = _manifest(bundle)["artifacts"]
    del artifacts[missing]
    _set_artifacts(bundle, artifacts)
    with pytest.raises(
        ArtifactValidationError, match=re.escape(f"required artefacts: {missing}")
    ):
        load_public_bundle(bundle)


def test_changed_or_deleted_listed_file_is_rejected(bundle: Path) -> None:
    ranking = bundle / "risk_ranking.csv"
    ranking.write_text(ranking.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="SHA-256 mismatch"):
        load_public_bundle(bundle)
    ranking.unlink()
    with pytest.raises(ArtifactValidationError, match="Required artefact is missing"):
        load_public_bundle(bundle)


def test_ranking_with_true_rul_is_rejected(bundle: Path) -> None:
    ranking = pd.read_csv(bundle / "risk_ranking.csv")
    ranking["rul_true"] = 1.0
    ranking.to_csv(bundle / "risk_ranking.csv", index=False)
    _relist(bundle)
    with pytest.raises(ArtifactValidationError, match="exactly the review columns"):
        load_public_bundle(bundle)


def test_run_payload_with_true_rul_is_rejected(bundle: Path) -> None:
    payload = json.loads((bundle / "run.json").read_text(encoding="utf-8"))
    payload["metrics"]["rows"] = [{"unit_id": 1, "rul_true": 12}]
    write_json(bundle / "run.json", payload)
    _relist(bundle)
    with pytest.raises(ArtifactValidationError, match=r"metrics\.rows\[0\]\.rul_true"):
        load_public_bundle(bundle)


def test_export_refuses_forbidden_fields_and_round_trips(tmp_path: Path) -> None:
    build_verified_run(tmp_path)
    out = tmp_path / "out"
    export_public_bundle(tmp_path, "fixture", out)
    assert load_public_bundle(out).run_id == "fixture"

    second = tmp_path / "second"
    run_dir = build_verified_run(second)
    explanation = run_dir / "explanation.json"
    write_json(explanation, {"global_importance": [], "rul_true": [10]})
    _relock(run_dir, {"explanation.json": sha256_file(explanation)})
    with pytest.raises(ArtifactValidationError, match="must not be published"):
        export_public_bundle(second, "fixture", tmp_path / "refused")
    assert not (tmp_path / "refused" / "manifest.json").exists()


def _relock(run_dir: Path, artifact_updates: dict | None = None, **changes) -> None:
    """Edit the model lock, then rehash it in both manifests that pin it."""
    lock_path = run_dir / "model_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock.update(changes)
    if artifact_updates is not None:
        lock["artifacts"] = {**lock["artifacts"], **artifact_updates}
    write_json(lock_path, lock)
    for path in (
        run_dir / "manifest.json",
        run_dir / "official_test" / "evaluation_manifest.json",
    ):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["model_lock_sha256"] = sha256_file(lock_path)
        write_json(path, manifest)


def test_only_known_lock_schemas_load(tmp_path: Path) -> None:
    run_dir = build_verified_run(tmp_path)
    _relock(run_dir, schema_version=2)
    assert load_verified_run(tmp_path, "fixture").model_lock["schema_version"] == 2
    _relock(run_dir, schema_version=4)
    with pytest.raises(ArtifactValidationError, match="schema_version"):
        load_verified_run(tmp_path, "fixture")


def test_lock_must_list_the_files_it_uses(tmp_path: Path) -> None:
    run_dir = build_verified_run(tmp_path)
    lock = json.loads((run_dir / "model_lock.json").read_text(encoding="utf-8"))
    _relock(
        run_dir, artifacts={"explanation.json": lock["artifacts"]["explanation.json"]}
    )
    with pytest.raises(
        ArtifactValidationError, match=r"required artefacts: model.joblib"
    ):
        load_verified_run(tmp_path, "fixture")

    _relock(run_dir, artifacts={})
    with pytest.raises(ArtifactValidationError, match="required artefacts"):
        load_verified_run(tmp_path, "fixture")


def test_lock_artifact_outside_run_is_rejected(tmp_path: Path) -> None:
    run_dir = build_verified_run(tmp_path)
    outside = tmp_path / "runs" / "outside.json"
    write_json(outside, {})
    _relock(run_dir, {"../outside.json": sha256_file(outside)})
    with pytest.raises(ArtifactValidationError, match="external path"):
        load_verified_run(tmp_path, "fixture")


def _complete(run_dir: Path, artifacts: dict) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "pipeline_complete"
    manifest["pipeline"] = {
        "schema_version": 3,
        "status": "complete",
        "artifacts": artifacts,
    }
    write_json(run_dir / "manifest.json", manifest)


def test_pipeline_record_must_list_reports(tmp_path: Path) -> None:
    run_dir = build_verified_run(tmp_path)
    (run_dir / "report.json").write_text("{}\n", encoding="utf-8")
    listed = {"report.json": sha256_file(run_dir / "report.json")}
    _complete(run_dir, listed)
    with pytest.raises(ArtifactValidationError, match=r"report.html"):
        load_verified_run(tmp_path, "fixture")

    _complete(run_dir, {})
    with pytest.raises(ArtifactValidationError, match="required artefacts"):
        load_verified_run(tmp_path, "fixture")


def test_real_runs_still_load_when_present() -> None:
    runs = sorted((ROOT / "runs").glob("*/manifest.json"))
    complete = [
        path.parent.name
        for path in runs
        if json.loads(path.read_text(encoding="utf-8"))["status"] == "pipeline_complete"
    ]
    if not complete:
        pytest.skip("runs/ holds NASA-derived files and is not committed")
    for run_id in complete:
        assert load_verified_run(ROOT, run_id).run_id == run_id
