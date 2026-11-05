"""Verified artefact loading for the Streamlit RUL review application."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from aeromaintain.data.pipeline import SENSOR_COLUMNS, sha256_file

SAFE_PREDICTION_COLUMNS = (
    "unit_id",
    "cycle",
    "prediction",
    "interval_low",
    "interval_high",
    "risk_band",
)

PUBLIC_RUN_FILE = "run.json"
PUBLIC_RANKING_FILE = "risk_ranking.csv"
PUBLIC_MANIFEST_FILE = "manifest.json"


class ArtifactValidationError(RuntimeError):
    """Raised when a run cannot be trusted by the review application."""


@dataclass(frozen=True)
class AppArtifacts:
    """Validated model and evaluation data exposed to the presentation layer."""

    run_id: str
    run_dir: Path
    run_manifest: dict[str, Any]
    model_lock: dict[str, Any]
    metrics: dict[str, Any]
    explanation: dict[str, Any]
    risk_ranking: pd.DataFrame
    sensor_history: pd.DataFrame
    downloads: dict[str, bytes]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactValidationError(f"Required artefact is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            f"Artefact is unreadable or invalid: {path}"
        ) from exc


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ArtifactValidationError(f"Required artefact is missing: {path}")
    if sha256_file(path) != expected:
        raise ArtifactValidationError(f"{label} SHA-256 mismatch: {path}")


def _resolve_within(base: Path, relative_path: str, label: str) -> Path:
    path = (base / relative_path).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as exc:
        raise ArtifactValidationError(
            f"{label} references an external path: {relative_path}"
        ) from exc
    return path


def _verify_artifact_map(base: Path, artifacts: dict[str, str], label: str) -> None:
    for relative_path, expected in artifacts.items():
        path = _resolve_within(base, relative_path, label)
        _require_hash(path, expected, label)


def _safe_run_dir(project_root: Path, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id:
        raise ArtifactValidationError("run_id must be one safe path component")
    runs_root = (project_root / "runs").resolve()
    run_dir = (runs_root / run_id).resolve()
    try:
        run_dir.relative_to(runs_root)
    except ValueError as exc:
        raise ArtifactValidationError("run_id resolves outside runs/") from exc
    if not run_dir.is_dir():
        raise ArtifactValidationError(f"Run directory does not exist: {run_dir}")
    return run_dir


def _load_verified_run(project_root: Path, run_id: str) -> AppArtifacts:
    """Load one named model evaluation after verifying its complete hash chain."""
    root = project_root.resolve()
    run_dir = _safe_run_dir(root, run_id)
    run_manifest = _read_json(run_dir / "manifest.json")
    if run_manifest.get("run_id") != run_id:
        raise ArtifactValidationError("Run manifest identity mismatch")
    if run_manifest.get("status") not in {"model_locked", "pipeline_complete"}:
        raise ArtifactValidationError("Run is not model-locked or pipeline-complete")
    if run_manifest.get("status") == "pipeline_complete":
        pipeline = run_manifest.get("pipeline", {})
        if pipeline.get("status") != "complete":
            raise ArtifactValidationError("Pipeline completion record is invalid")
        _verify_artifact_map(run_dir, pipeline.get("artifacts", {}), "Pipeline")

    lock_path = run_dir / "model_lock.json"
    _require_hash(lock_path, run_manifest["model_lock_sha256"], "Model lock")
    lock = _read_json(lock_path)
    if lock.get("run_id") != run_id or lock.get("schema_version") not in {2, 3}:
        raise ArtifactValidationError("Model lock identity or schema mismatch")
    _verify_artifact_map(run_dir, lock.get("artifacts", {}), "Model artefact")
    _require_hash(
        _resolve_within(root, lock["config"]["path"], "Model config"),
        lock["config"]["sha256"],
        "Model config",
    )

    processed_dir = root / "data" / "processed" / str(lock["dataset"]).lower()
    for name, key in (
        ("manifest.json", "manifest_sha256"),
        ("split_manifest.json", "split_manifest_sha256"),
        ("train.parquet", "train_sha256"),
        ("test.parquet", "test_sha256"),
        ("evaluation/test_rul.parquet", "official_test_rul_sha256"),
    ):
        _require_hash(processed_dir / name, lock["data"][key], "Locked data")

    official_dir = run_dir / "official_test"
    evaluation_manifest = _read_json(official_dir / "evaluation_manifest.json")
    if evaluation_manifest.get("run_id") != run_id:
        raise ArtifactValidationError("Evaluation manifest identity mismatch")
    _require_hash(
        lock_path,
        evaluation_manifest["model_lock_sha256"],
        "Evaluation model lock",
    )
    for name, key in (
        ("predictions.parquet", "predictions_sha256"),
        ("metrics.json", "metrics_sha256"),
        ("error_analysis.json", "error_analysis_sha256"),
    ):
        _require_hash(official_dir / name, evaluation_manifest[key], "Evaluation")

    predictions = pd.read_parquet(official_dir / "predictions.parquet")
    missing = set(SAFE_PREDICTION_COLUMNS) - set(predictions.columns)
    if missing:
        raise ArtifactValidationError(
            "Predictions are missing review columns: " + ", ".join(sorted(missing))
        )
    risk_ranking = (
        predictions.loc[:, SAFE_PREDICTION_COLUMNS]
        .sort_values(["interval_low", "unit_id"])
        .reset_index(drop=True)
    )

    processed_test = processed_dir / "test.parquet"
    _require_hash(processed_test, lock["data"]["test_sha256"], "Processed test data")
    test = pd.read_parquet(processed_test)
    sensor_history = test.loc[:, ["unit_id", "cycle", *SENSOR_COLUMNS]].copy()
    downloads = {
        "risk_ranking.csv": risk_ranking.to_csv(index=False).encode("utf-8"),
    }
    return AppArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        run_manifest=run_manifest,
        model_lock=lock,
        metrics=_read_json(official_dir / "metrics.json"),
        explanation=_read_json(run_dir / "explanation.json"),
        risk_ranking=risk_ranking,
        sensor_history=sensor_history,
        downloads=downloads,
    )


def export_public_bundle(
    project_root: Path,
    run_id: str,
    output_dir: Path,
) -> Path:
    """Write a hash-listed demo bundle holding no NASA rows or true RUL values."""
    artifacts = load_verified_run(project_root, run_id)
    lock = artifacts.model_lock
    run_payload = {
        "run_id": run_id,
        "model_lock_sha256": artifacts.run_manifest["model_lock_sha256"],
        "model_lock": {
            "dataset": lock["dataset"],
            "protocol": lock["protocol"],
            "seed": lock["seed"],
            "champion": {"kind": lock["champion"]["kind"]},
            "calibration": lock["calibration"],
        },
        "metrics": artifacts.metrics,
        "explanation": artifacts.explanation,
    }
    files = {
        PUBLIC_RUN_FILE: (json.dumps(run_payload, indent=2, sort_keys=True) + "\n"),
        PUBLIC_RANKING_FILE: artifacts.risk_ranking.to_csv(
            index=False, lineterminator="\n"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, text in files.items():
        path = output_dir / name
        path.write_bytes(text.encode("utf-8"))
        hashes[name] = sha256_file(path)
    manifest = output_dir / PUBLIC_MANIFEST_FILE
    manifest.write_bytes(
        (json.dumps({"artifacts": hashes}, indent=2, sort_keys=True) + "\n").encode()
    )
    return manifest


def load_public_bundle(bundle_dir: Path) -> AppArtifacts:
    """Load a demo bundle after checking its listed hashes and row-free contract."""
    manifest = _read_json(bundle_dir / PUBLIC_MANIFEST_FILE)
    try:
        _verify_artifact_map(bundle_dir, manifest["artifacts"], "Public bundle")
        payload = _read_json(bundle_dir / PUBLIC_RUN_FILE)
        ranking = pd.read_csv(bundle_dir / PUBLIC_RANKING_FILE)
        if set(ranking.columns) != set(SAFE_PREDICTION_COLUMNS):
            raise ArtifactValidationError(
                "Public bundle ranking must hold review columns only"
            )
        return AppArtifacts(
            run_id=payload["run_id"],
            run_dir=bundle_dir,
            run_manifest={"model_lock_sha256": payload["model_lock_sha256"]},
            model_lock=payload["model_lock"],
            metrics=payload["metrics"],
            explanation=payload["explanation"],
            risk_ranking=ranking,
            sensor_history=pd.DataFrame(),
            downloads={
                "risk_ranking.csv": ranking.to_csv(index=False).encode("utf-8"),
            },
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"Public bundle structure is invalid: {exc}"
        ) from exc


def load_verified_run(project_root: Path, run_id: str) -> AppArtifacts:
    """Return a validated RUL run with structural failures translated for the UI."""
    try:
        return _load_verified_run(project_root, run_id)
    except ArtifactValidationError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"Verified run artefact structure is invalid: {exc}"
        ) from exc
