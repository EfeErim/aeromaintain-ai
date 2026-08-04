"""Command-line interface for AeroMaintain AI."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from aeromaintain.config import (
    REQUIRED_CONFIG_FILES,
    REQUIRED_LOCAL_DATA_DIRS,
    resolve_project_root,
)
from aeromaintain.data import DataPipelineError, prepare_fd001
from aeromaintain.delivery import DeliveryError, run_pipeline
from aeromaintain.models import ModelingError, evaluate_locked, train_and_lock
from aeromaintain.optimization import OptimizationError, optimize_run

app = typer.Typer(
    add_completion=False,
    help="AeroMaintain AI educational decision-support prototype.",
    no_args_is_help=True,
)


@dataclass(frozen=True)
class DoctorCheck:
    """One deterministic environment health check."""

    name: str
    passed: bool
    detail: str


def collect_doctor_checks(
    project_root: Path,
    version_info: tuple[int, int, int] | None = None,
) -> tuple[DoctorCheck, ...]:
    """Collect environment checks without printing or changing project files."""
    version = version_info or (
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
    )
    supported_python = version[:2] == (3, 11)

    missing_configs = [
        name
        for name in REQUIRED_CONFIG_FILES
        if not (project_root / "configs" / name).is_file()
    ]
    missing_directories = [
        name for name in REQUIRED_LOCAL_DATA_DIRS if not (project_root / name).is_dir()
    ]
    package_available = importlib.util.find_spec("aeromaintain") is not None
    writable_root = project_root.is_dir() and os.access(project_root, os.W_OK)
    local_directories_ready = not missing_directories or writable_root

    return (
        DoctorCheck(
            "python",
            supported_python,
            f"{version[0]}.{version[1]}.{version[2]} (required >=3.11,<3.12)",
        ),
        DoctorCheck(
            "package",
            package_available,
            "aeromaintain import is available"
            if package_available
            else "aeromaintain import is unavailable",
        ),
        DoctorCheck(
            "configs",
            not missing_configs,
            "project.yaml and scenario.yaml found"
            if not missing_configs
            else f"missing: {', '.join(missing_configs)}",
        ),
        DoctorCheck(
            "local directories",
            local_directories_ready,
            "raw, processed, and artifacts directories found"
            if not missing_directories
            else (
                "created on first run: " + ", ".join(missing_directories)
                if writable_root
                else "missing and project root is not writable: "
                + ", ".join(missing_directories)
            ),
        ),
        DoctorCheck(
            "project root",
            writable_root,
            f"writable: {project_root}"
            if writable_root
            else f"not writable: {project_root}",
        ),
    )


@app.callback()
def main() -> None:
    """Run AeroMaintain AI commands."""


@app.command()
def doctor(
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            help="Project root to inspect. Defaults to the current directory.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Report whether the local Python and project environment are usable."""
    root = resolve_project_root(project_root)
    checks = collect_doctor_checks(root)

    typer.echo("AeroMaintain AI environment doctor")
    for check in checks:
        marker = "PASS" if check.passed else "FAIL"
        typer.echo(f"[{marker}] {check.name}: {check.detail}")

    passed = sum(check.passed for check in checks)
    typer.echo(f"Summary: {passed}/{len(checks)} checks passed")
    if passed != len(checks):
        raise typer.Exit(code=1)


@app.command()
def prepare(
    archive: Annotated[
        Path | None,
        typer.Option(
            "--archive",
            help="Verified local CMAPSSData.zip; otherwise download from NASA.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            help="Project root. Defaults to the current directory.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Prepare the frozen NASA C-MAPSS FD001 dataset."""
    root = resolve_project_root(project_root)
    try:
        result = prepare_fd001(root, archive_path=archive)
    except DataPipelineError as exc:
        typer.echo(f"Preparation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("FD001 preparation complete")
    typer.echo(
        f"Rows: train={result.train_rows}, test={result.test_rows}; "
        f"engines: development={result.development_engines}, "
        f"calibration={result.calibration_engines}"
    )
    typer.echo(f"Processed data: {result.output_dir}")
    typer.echo(f"Verified artifacts: {len(result.artifact_hashes)}")


@app.command()
def train(
    run_id: Annotated[
        str,
        typer.Option(
            "--run-id",
            help="New immutable run directory name under runs/.",
        ),
    ],
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            help="Project root. Defaults to the current directory.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Select, fit, calibrate, explain, persist, and lock an FD001 RUL model."""
    root = resolve_project_root(project_root)
    try:
        result = train_and_lock(root, run_id=run_id)
    except ModelingError as exc:
        typer.echo(f"Training failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("FD001 model training and lock complete")
    typer.echo(f"Run: {result.run_id}")
    typer.echo(f"Champion: {result.champion}")
    typer.echo(f"Calibration q: {result.calibration_q:.6f}")
    typer.echo(f"Model lock: {result.model_lock}")


@app.command()
def evaluate(
    run_id: Annotated[
        str,
        typer.Option(
            "--run-id",
            help="Existing run with a verified model_lock.json.",
        ),
    ],
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            help="Project root. Defaults to the current directory.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Evaluate one verified model lock against isolated official test labels."""
    root = resolve_project_root(project_root)
    try:
        result = evaluate_locked(root, run_id=run_id)
    except ModelingError as exc:
        typer.echo(f"Evaluation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    metrics = result.metrics
    typer.echo("Locked FD001 official evaluation complete")
    typer.echo(f"Run: {result.run_id}")
    typer.echo(
        f"MAE={metrics['mae']:.6f}; RMSE={metrics['rmse']:.6f}; "
        f"NASA={metrics['nasa_score_motor_normalized']:.6f}"
    )
    typer.echo(f"Official results: {result.output_dir}")


@app.command()
def optimize(
    run_id: Annotated[
        str,
        typer.Option(
            "--run-id",
            help="Existing run with verified locked official predictions.",
        ),
    ],
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            help="Project root. Defaults to the current directory.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Generate, solve, and compare the synthetic maintenance scenario."""
    root = resolve_project_root(project_root)
    try:
        result = optimize_run(root, run_id=run_id)
    except OptimizationError as exc:
        typer.echo(f"Optimization failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    cp_sat = next(row for row in result.policy_comparison if row["policy"] == "cp_sat")
    typer.echo("Synthetic FD001 maintenance optimization complete")
    typer.echo(f"Run: {result.run_id}")
    typer.echo(
        f"CP-SAT status={cp_sat['solver_status']}; "
        f"due deferrals={cp_sat.get('due_deferrals', 'n/a')}; "
        f"late days={cp_sat.get('late_days', 'n/a')}"
    )
    typer.echo(f"Optimization artifacts: {result.output_dir}")


@app.command()
def pipeline(
    run_id: Annotated[
        str,
        typer.Option(
            "--run-id",
            help="New immutable end-to-end run directory name under runs/.",
        ),
    ],
    archive: Annotated[
        Path | None,
        typer.Option(
            "--archive",
            help="Verified local CMAPSSData.zip; otherwise download from NASA.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            help="Project root. Defaults to the current directory.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Run prepare, train/lock, evaluate, optimize, and report."""
    root = resolve_project_root(project_root)
    try:
        result = run_pipeline(root, run_id=run_id, archive_path=archive)
    except DeliveryError as exc:
        typer.echo(f"Pipeline failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("AeroMaintain end-to-end pipeline complete")
    typer.echo(f"Run: {result.run_id}")
    typer.echo(f"Manifest: {result.manifest_path}")
    typer.echo(f"Report: {result.report_path}")


@app.command("app")
def app_command(
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="Explicit verified run to display."),
    ],
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            help="Project root. Defaults to the current directory.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    smoke: Annotated[
        bool,
        typer.Option("--smoke", help="Render once with Streamlit's test runtime."),
    ] = False,
) -> None:
    """Open the local Streamlit application for one verified run."""
    from aeromaintain.app import ArtifactValidationError, load_verified_run

    root = resolve_project_root(project_root)
    try:
        load_verified_run(root, run_id)
    except ArtifactValidationError as exc:
        typer.echo(f"Application failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    script = Path(__file__).parent / "app" / "main.py"
    environment = os.environ.copy()
    environment["AEROMAINTAIN_RUN_ID"] = run_id
    environment["AEROMAINTAIN_PROJECT_ROOT"] = str(root)
    if smoke:
        previous_run = os.environ.get("AEROMAINTAIN_RUN_ID")
        previous_root = os.environ.get("AEROMAINTAIN_PROJECT_ROOT")
        os.environ.update(
            {
                "AEROMAINTAIN_RUN_ID": run_id,
                "AEROMAINTAIN_PROJECT_ROOT": str(root),
            }
        )
        try:
            from streamlit.testing.v1 import AppTest

            tested = AppTest.from_file(str(script), default_timeout=30).run()
        finally:
            if previous_run is None:
                os.environ.pop("AEROMAINTAIN_RUN_ID", None)
            else:
                os.environ["AEROMAINTAIN_RUN_ID"] = previous_run
            if previous_root is None:
                os.environ.pop("AEROMAINTAIN_PROJECT_ROOT", None)
            else:
                os.environ["AEROMAINTAIN_PROJECT_ROOT"] = previous_root
        if tested.exception:
            typer.echo(f"Application smoke failed: {tested.exception}", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Streamlit smoke passed for verified run: {run_id}")
        return

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(script),
            "--server.headless=true",
        ],
        check=False,
        env=environment,
    )
    if completed.returncode:
        raise typer.Exit(code=completed.returncode)
