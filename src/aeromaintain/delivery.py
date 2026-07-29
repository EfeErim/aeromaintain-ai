"""End-to-end delivery orchestration and immutable run reporting."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from aeromaintain.app import AppArtifacts, load_verified_run
from aeromaintain.data import PrepareResult, prepare_fd001
from aeromaintain.data.pipeline import sha256_file
from aeromaintain.models import evaluate_locked, train_and_lock
from aeromaintain.models.rul import EvaluationResult, TrainResult
from aeromaintain.optimization import OptimizationResult, optimize_run


class DeliveryError(RuntimeError):
    """Raised when an M4 delivery contract cannot be completed."""


@dataclass(frozen=True)
class PipelineResult:
    """Small summary returned by a completed end-to-end run."""

    run_id: str
    run_dir: Path
    report_path: Path
    manifest_path: Path


PrepareStep = Callable[..., PrepareResult]
TrainStep = Callable[..., TrainResult]
EvaluationStep = Callable[..., EvaluationResult]
OptimizationStep = Callable[..., OptimizationResult]
LoadStep = Callable[[Path, str], AppArtifacts]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise DeliveryError(f"Refusing to overwrite existing output: {path}")
    path.write_bytes(payload)


def _atomic_replace(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _report_payload(artifacts: AppArtifacts) -> dict[str, Any]:
    metrics = artifacts.metrics
    return {
        "schema_version": 1,
        "run_id": artifacts.run_id,
        "status": "pipeline_complete",
        "scope": (
            "Educational NASA C-MAPSS FD001 prototype with synthetic operational "
            "and cost assumptions; not an airworthiness or production-fleet system."
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
            "nominal_empirical_interval": metrics["nominal_empirical_interval"],
        },
        "policy_comparison": artifacts.policy_comparison.to_dict(orient="records"),
        "capacity_comparison": artifacts.capacity_comparison.to_dict(orient="records"),
        "truth_boundary": (
            "Official true RUL is excluded from scenario generation, policies, "
            "solver inputs, and application decision views."
        ),
    }


def _report_html(report: dict[str, Any]) -> bytes:
    official = report["official_test"]
    policy_rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['policy']))}</td>"
        f"<td>{escape(str(row['solver_status']))}</td>"
        f"<td>{int(row['scheduled_maintenance'])}</td>"
        f"<td>{int(row['due_deferrals'])}</td>"
        f"<td>{int(row['total_synthetic_cost_units'])}</td>"
        "</tr>"
        for row in report["policy_comparison"]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AeroMaintain AI run report</title>
  <style>
    body {{ font: 16px/1.5 system-ui; margin: 2rem auto; max-width: 960px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #cbd5e1; padding: .55rem; text-align: left; }}
    .warning {{ background: #fff7ed; border-left: 5px solid #f97316; padding: 1rem; }}
  </style>
</head>
<body>
  <h1>AeroMaintain AI — {escape(report["run_id"])}</h1>
  <p class="warning">{escape(report["scope"])}</p>
  <h2>Locked official-test result</h2>
  <ul>
    <li>MAE: {official["mae"]:.6f}</li>
    <li>RMSE: {official["rmse"]:.6f}</li>
    <li>Motor-normalized NASA score:
      {official["nasa_score_motor_normalized"]:.6f}</li>
  </ul>
  <h2>Decision comparison</h2>
  <table>
    <thead><tr><th>Policy</th><th>Status</th><th>Scheduled</th>
      <th>Due deferrals</th><th>Synthetic total cost</th></tr></thead>
    <tbody>{policy_rows}</tbody>
  </table>
  <p>{escape(report["truth_boundary"])}</p>
</body>
</html>
"""
    return html.encode("utf-8")


def _finalize_run(
    project_root: Path,
    run_id: str,
    *,
    load_step: LoadStep,
) -> PipelineResult:
    run_dir = project_root / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "model_locked":
        raise DeliveryError("Pipeline finalization requires a model_locked manifest")
    artifacts = load_step(project_root, run_id)
    report = _report_payload(artifacts)
    report_json = _json_bytes(report)
    report_html = _report_html(report)
    report_json_path = run_dir / "report.json"
    report_html_path = run_dir / "report.html"
    _write_new(report_json_path, report_json)
    _write_new(report_html_path, report_html)

    manifest["status"] = "pipeline_complete"
    manifest["pipeline"] = {
        "schema_version": 1,
        "status": "complete",
        "stages": [
            "prepare",
            "train_and_lock",
            "evaluate_locked",
            "optimize",
            "report",
        ],
        "artifacts": {
            "report.json": hashlib.sha256(report_json).hexdigest(),
            "report.html": hashlib.sha256(report_html).hexdigest(),
            "official_test/evaluation_manifest.json": sha256_file(
                run_dir / "official_test" / "evaluation_manifest.json"
            ),
            "optimization/manifest.json": sha256_file(
                run_dir / "optimization" / "manifest.json"
            ),
        },
    }
    _atomic_replace(manifest_path, _json_bytes(manifest))
    return PipelineResult(
        run_id=run_id,
        run_dir=run_dir,
        report_path=report_html_path,
        manifest_path=manifest_path,
    )


def run_pipeline(
    project_root: Path,
    *,
    run_id: str,
    archive_path: Path | None = None,
    prepare_step: PrepareStep = prepare_fd001,
    train_step: TrainStep = train_and_lock,
    evaluation_step: EvaluationStep = evaluate_locked,
    optimization_step: OptimizationStep = optimize_run,
    load_step: LoadStep = load_verified_run,
) -> PipelineResult:
    """Run prepare through report without overwriting an existing run."""
    root = project_root.resolve()
    if not run_id or Path(run_id).name != run_id:
        raise DeliveryError("run_id must be one safe path component")
    run_dir = root / "runs" / run_id
    if run_dir.exists():
        raise DeliveryError(f"Run directory already exists: {run_dir}")
    try:
        prepare_step(root, archive_path=archive_path)
        train_step(root, run_id=run_id)
        evaluation_step(root, run_id=run_id)
        optimization_step(root, run_id=run_id)
        return _finalize_run(root, run_id, load_step=load_step)
    except DeliveryError:
        raise
    except Exception as exc:
        raise DeliveryError(f"Pipeline stopped before completion: {exc}") from exc
