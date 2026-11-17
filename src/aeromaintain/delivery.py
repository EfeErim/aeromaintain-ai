"""End-to-end RUL evaluation orchestration and run reporting.

The run manifest is written last and is the only commit point: a run is complete
only when its manifest says so and every file it lists verifies. Run states and how
``finish`` resumes are described in ``docs/architecture.md`` (Run states and resuming).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import IO, Any

from aeromaintain.artifacts import (
    EVALUATION_MANIFEST,
    REPORT_HTML,
    REPORT_JSON,
    STATUS_MODEL_LOCKED,
    STATUS_PIPELINE_COMPLETE,
    RunArtifacts,
    load_verified_run,
    write_replacing,
)
from aeromaintain.data import PrepareResult, prepare_dataset
from aeromaintain.data.pipeline import json_bytes, sha256_file
from aeromaintain.models import (
    EvaluationResult,
    TrainResult,
    evaluate_locked,
    train_and_lock,
)

OFFICIAL_RESULT_FILES = (
    "predictions.parquet",
    "metrics.json",
    "error_analysis.json",
    "evaluation_manifest.json",
)


class DeliveryError(RuntimeError):
    """Raised when the RUL delivery contract cannot be completed."""


@dataclass(frozen=True)
class PipelineResult:
    """Small summary returned by a completed end-to-end run."""

    run_id: str
    run_dir: Path
    report_path: Path
    manifest_path: Path
    already_complete: bool = False


PrepareStep = Callable[..., PrepareResult]
TrainStep = Callable[..., TrainResult]
EvaluationStep = Callable[..., EvaluationResult]
LoadStep = Callable[[Path, str], RunArtifacts]


def _write_report_file(path: Path, payload: bytes) -> None:
    """Write a report, or accept an identical one left by an interrupted finish."""
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise DeliveryError(
            f"Existing {path.name} differs from the report this run produces; "
            f"refusing to replace it: {path}"
        )
    write_replacing(path, payload)


def _lock_file(handle: IO[bytes]) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: IO[bytes]) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def run_lock(project_root: Path, run_id: str) -> Iterator[None]:
    """Hold an OS lock so two finish or pipeline calls cannot write one run.

    The operating system releases the lock when the process exits, so a crash does
    not leave the run blocked. The empty lock file under ``runs/`` may remain.
    """
    if not run_id or Path(run_id).name != run_id:
        raise DeliveryError("run_id must be one safe path component")
    runs = project_root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    with (runs / f".{run_id}.lock").open("a+b") as handle:
        try:
            _lock_file(handle)
        except OSError as exc:
            raise DeliveryError(
                f"Another finish or pipeline call is working on run {run_id}"
            ) from exc
        try:
            yield
        finally:
            _unlock_file(handle)


def _report_payload(artifacts: RunArtifacts) -> dict[str, Any]:
    metrics = artifacts.metrics
    dataset = artifacts.model_lock["dataset"]
    return {
        "schema_version": 3,
        "run_id": artifacts.run_id,
        "dataset": dataset,
        "protocol": artifacts.model_lock["protocol"],
        "status": STATUS_PIPELINE_COMPLETE,
        "scope": (
            f"Educational evaluation on NASA's simulated C-MAPSS {dataset} "
            "benchmark; not an airworthiness or production-fleet system."
        ),
        "model": {
            "champion": artifacts.model_lock["champion"]["kind"],
            "seed": artifacts.model_lock["seed"],
            "model_lock_sha256": artifacts.run_manifest["model_lock_sha256"],
            "interval": artifacts.model_lock["calibration"]["label"],
        },
        "official_test": {
            "engines": metrics["engines"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "nasa_score_motor_normalized": metrics["nasa_score_motor_normalized"],
            "critical_rul": metrics["critical_rul"],
            "alert_rules": metrics["alert_rules"],
            "nominal_empirical_interval": metrics["nominal_empirical_interval"],
            "bootstrap": metrics["bootstrap"],
        },
        "label_boundary": (
            "Official test RUL is opened only after model lock and is excluded "
            "from the application review table."
        ),
    }


def _report_html(report: dict[str, Any]) -> bytes:
    official = report["official_test"]
    point_rule = official["alert_rules"]["point_prediction"]
    risk_rule = official["alert_rules"]["interval_low"]
    interval = official["nominal_empirical_interval"]
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AeroMaintain AI RUL report</title>
  <style>
    body {{ font: 16px/1.5 system-ui; margin: 2rem auto; max-width: 760px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #cbd5e1; padding: .55rem; text-align: left; }}
    .warning {{ background: #fff7ed; border-left: 5px solid #f97316; padding: 1rem; }}
  </style>
</head>
<body>
  <h1>AeroMaintain AI — {escape(report["run_id"])}</h1>
  <p class="warning">{escape(report["scope"])}</p>
  <p>{escape(report["protocol"]["label"])}</p>
  <h2>Locked official-test result</h2>
  <table>
    <tbody>
      <tr><th>Engines</th><td>{int(official["engines"])}</td></tr>
      <tr><th>MAE</th><td>{official["mae"]:.6f}</td></tr>
      <tr><th>RMSE</th><td>{official["rmse"]:.6f}</td></tr>
      <tr><th>Motor-normalized NASA score</th>
        <td>{official["nasa_score_motor_normalized"]:.6f}</td></tr>
      <tr><th>Point rule recall / precision</th>
        <td>{point_rule["recall"]:.2%} / {point_rule["precision"]:.2%}</td></tr>
      <tr><th>Interval-low rule recall / precision</th>
        <td>{risk_rule["recall"]:.2%} / {risk_rule["precision"]:.2%}</td></tr>
      <tr><th>Observed interval coverage</th>
        <td>{interval["observed_official_test_coverage"]:.2%}</td></tr>
    </tbody>
  </table>
  <p>{escape(report["label_boundary"])}</p>
</body>
</html>
"""
    return html.encode("utf-8")


def report_bytes(artifacts: RunArtifacts) -> tuple[bytes, bytes]:
    """Return the JSON and HTML reports for a verified run, byte for byte."""
    report = _report_payload(artifacts)
    return json_bytes(report), _report_html(report)


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DeliveryError(f"Run manifest is missing: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeliveryError(f"Run manifest is unreadable: {path}") from exc
    if not isinstance(manifest, dict):
        raise DeliveryError(f"Run manifest must hold a JSON object: {path}")
    return manifest


def official_evaluation_state(run_dir: Path) -> str:
    """Return ``absent``, ``recorded`` or ``partial`` for ``official_test/``.

    ``recorded`` means the evaluation manifest exists; whether its files still
    verify is checked when the run is loaded.
    """
    official = run_dir / "official_test"
    present = [name for name in OFFICIAL_RESULT_FILES if (official / name).exists()]
    if not present:
        return "absent"
    if "evaluation_manifest.json" in present:
        return "recorded"
    return "partial"


def _result(run_dir: Path, *, already_complete: bool) -> PipelineResult:
    return PipelineResult(
        run_id=run_dir.name,
        run_dir=run_dir,
        report_path=run_dir / REPORT_HTML,
        manifest_path=run_dir / "manifest.json",
        already_complete=already_complete,
    )


def _finalize_run(
    project_root: Path,
    run_id: str,
    *,
    load_step: LoadStep,
) -> PipelineResult:
    run_dir = project_root / "runs" / run_id
    manifest = _read_manifest(run_dir)
    if manifest.get("status") != STATUS_MODEL_LOCKED:
        raise DeliveryError("Pipeline finalization requires a model_locked manifest")
    artifacts = load_step(project_root, run_id)
    report_json, report_html = report_bytes(artifacts)
    _write_report_file(run_dir / REPORT_JSON, report_json)
    _write_report_file(run_dir / REPORT_HTML, report_html)

    manifest["status"] = STATUS_PIPELINE_COMPLETE
    manifest["pipeline"] = {
        "schema_version": 3,
        "status": "complete",
        "stages": ["prepare", "train_and_lock", "evaluate_locked", "report"],
        "artifacts": {
            REPORT_JSON: hashlib.sha256(report_json).hexdigest(),
            REPORT_HTML: hashlib.sha256(report_html).hexdigest(),
            EVALUATION_MANIFEST: sha256_file(run_dir / EVALUATION_MANIFEST),
        },
    }
    write_replacing(run_dir / "manifest.json", json_bytes(manifest))
    return _result(run_dir, already_complete=False)


def _finish(
    root: Path,
    run_id: str,
    evaluation_step: EvaluationStep,
    load_step: LoadStep,
) -> PipelineResult:
    run_dir = root / "runs" / run_id
    status = _read_manifest(run_dir).get("status")
    if status == STATUS_PIPELINE_COMPLETE:
        load_step(root, run_id)
        return _result(run_dir, already_complete=True)
    if status != STATUS_MODEL_LOCKED:
        raise DeliveryError(f"Run {run_id} has unexpected status: {status!r}")
    state = official_evaluation_state(run_dir)
    if state == "partial":
        raise DeliveryError(
            f"Official evaluation of {run_id} stopped part-way: result files exist "
            "without evaluation_manifest.json. Labels may already have been read "
            "and the holdout claimed, so it is not rerun automatically."
        )
    if state == "absent":
        evaluation_step(root, run_id=run_id)
    return _finalize_run(root, run_id, load_step=load_step)


def finish_locked_run(
    project_root: Path,
    *,
    run_id: str,
    evaluation_step: EvaluationStep = evaluate_locked,
    load_step: LoadStep = load_verified_run,
) -> PipelineResult:
    """Evaluate a locked run if needed, then write its reports and completion.

    Calling it again on a completed run verifies the run and changes nothing.
    """
    root = project_root.resolve()
    with run_lock(root, run_id):
        try:
            return _finish(root, run_id, evaluation_step, load_step)
        except DeliveryError:
            raise
        except Exception as exc:
            raise DeliveryError(f"Finish stopped before completion: {exc}") from exc


def run_pipeline(
    project_root: Path,
    *,
    run_id: str,
    dataset: str = "FD001",
    archive_path: Path | None = None,
    prepare_step: PrepareStep = prepare_dataset,
    train_step: TrainStep = train_and_lock,
    evaluation_step: EvaluationStep = evaluate_locked,
    load_step: LoadStep = load_verified_run,
) -> PipelineResult:
    """Run prepare through a locked evaluation report without overwriting a run."""
    root = project_root.resolve()
    run_dir = root / "runs" / run_id
    with run_lock(root, run_id):
        if run_dir.exists():
            raise DeliveryError(f"Run directory already exists: {run_dir}")
        try:
            prepare_step(root, dataset=dataset, archive_path=archive_path)
            train_step(root, run_id=run_id, dataset=dataset)
            evaluation_step(root, run_id=run_id)
            return _finalize_run(root, run_id, load_step=load_step)
        except DeliveryError:
            raise
        except Exception as exc:
            raise DeliveryError(f"Pipeline stopped before completion: {exc}") from exc
