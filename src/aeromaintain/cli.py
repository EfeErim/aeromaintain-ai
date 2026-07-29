"""Command-line interface for AeroMaintain AI."""

from __future__ import annotations

import importlib.util
import os
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
            not missing_directories,
            "raw, processed, and artifacts directories found"
            if not missing_directories
            else f"missing: {', '.join(missing_directories)}",
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
