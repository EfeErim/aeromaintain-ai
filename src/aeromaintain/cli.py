"""Command-line interface for AeroMaintain AI."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from aeromaintain.config import resolve_project_root
from aeromaintain.data import DataPipelineError, prepare_dataset
from aeromaintain.delivery import DeliveryError, finish_locked_run, run_pipeline
from aeromaintain.models import (
    ModelingError,
    compare_runs,
    evaluate_locked,
    run_seed_sweep,
    train_and_lock,
    write_report,
)

app = typer.Typer(
    add_completion=False,
    help="AeroMaintain AI educational RUL evaluation prototype.",
    no_args_is_help=True,
)

ProjectRoot = Annotated[
    Path | None,
    typer.Option(
        "--project-root",
        help="Project root. Defaults to the current directory.",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
]
Archive = Annotated[
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
]
Dataset = Annotated[
    str,
    typer.Option("--dataset", help="C-MAPSS subset: FD001 through FD004."),
]


@app.callback()
def main() -> None:
    """Run AeroMaintain AI commands."""


@app.command()
def prepare(
    archive: Archive = None,
    project_root: ProjectRoot = None,
    dataset: Dataset = "FD001",
) -> None:
    """Prepare one frozen NASA C-MAPSS subset."""
    root = resolve_project_root(project_root)
    try:
        result = prepare_dataset(root, dataset=dataset, archive_path=archive)
    except DataPipelineError as exc:
        typer.echo(f"Preparation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"{dataset} preparation complete")
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
    project_root: ProjectRoot = None,
    dataset: Dataset = "FD001",
) -> None:
    """Select, fit, calibrate, explain, persist, and lock an RUL model."""
    root = resolve_project_root(project_root)
    try:
        result = train_and_lock(root, run_id=run_id, dataset=dataset)
    except ModelingError as exc:
        typer.echo(f"Training failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"{dataset} model training and lock complete")
    typer.echo(f"Run: {result.run_id}")
    typer.echo(f"Champion: {result.champion}")
    typer.echo(f"Largest interval half-width: {result.interval_half_width:.6f}")
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
    project_root: ProjectRoot = None,
) -> None:
    """Evaluate one verified model lock against isolated official test labels."""
    root = resolve_project_root(project_root)
    try:
        result = evaluate_locked(root, run_id=run_id)
    except ModelingError as exc:
        typer.echo(f"Evaluation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    metrics = result.metrics
    typer.echo("Locked official evaluation complete")
    typer.echo(f"Run: {result.run_id}")
    typer.echo(
        f"MAE={metrics['mae']:.6f}; RMSE={metrics['rmse']:.6f}; "
        f"NASA={metrics['nasa_score_motor_normalized']:.6f}"
    )
    typer.echo(f"Official results: {result.output_dir}")


@app.command()
def robustness(
    base_run_id: Annotated[
        str,
        typer.Option("--base-run-id", help="Verified run whose settings are refit."),
    ],
    seeds: Annotated[
        int,
        typer.Option("--seeds", min=1, help="Number of consecutive seeds to run."),
    ] = 10,
    first_seed: Annotated[
        int,
        typer.Option("--first-seed", help="First seed in the sweep."),
    ] = 42,
    project_root: ProjectRoot = None,
) -> None:
    """Repeat a locked specification across seeds on already seen test labels."""
    root = resolve_project_root(project_root)
    try:
        report = run_seed_sweep(
            root,
            base_run_id=base_run_id,
            seeds=tuple(range(first_seed, first_seed + seeds)),
        )
        output = root / "runs" / base_run_id / "robustness" / "seed_sweep.json"
        write_report(output, report)
    except ModelingError as exc:
        typer.echo(f"Robustness failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    rmse = report["summary"]["rmse"]
    typer.echo(f"Seeds: {len(report['seeds'])}")
    typer.echo(f"RMSE mean={rmse['mean']:.6f}; std={rmse['std']:.6f}")
    typer.echo(f"Report: {output}")


@app.command("escalation-study")
def escalation_study(
    base_run_id: Annotated[
        str,
        typer.Option("--base-run-id", help="Verified run to challenge."),
    ],
    method: Annotated[
        str,
        typer.Option("--method", help="Challenger: lstm or cqr."),
    ],
    project_root: ProjectRoot = None,
) -> None:
    """Score a protocol amendment challenger without opening test labels."""
    from aeromaintain.models.escalation import run_cqr_study, run_lstm_study

    studies = {"lstm": run_lstm_study, "cqr": run_cqr_study}
    root = resolve_project_root(project_root)
    try:
        if method not in studies:
            raise ModelingError(f"Unknown challenger: {method}")
        report = studies[method](root, base_run_id=base_run_id)
        output = root / "runs" / base_run_id / "robustness" / f"{method}_study.json"
        write_report(output, report)
    except ModelingError as exc:
        typer.echo(f"Study failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Report: {output}")


@app.command("compare-runs")
def compare_runs_command(
    first: Annotated[str, typer.Option("--first", help="First run ID.")],
    second: Annotated[str, typer.Option("--second", help="Second run ID.")],
    project_root: ProjectRoot = None,
) -> None:
    """Report which files of two runs repeat byte for byte."""
    root = resolve_project_root(project_root)
    try:
        report = compare_runs(root, first_run_id=first, second_run_id=second)
        output = root / "runs" / first / "robustness" / f"compare_{second}.json"
        write_report(output, report)
    except ModelingError as exc:
        typer.echo(f"Comparison failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Identical files: {len(report['identical_files'])}")
    typer.echo(f"Different files: {report['different_files']}")
    typer.echo(f"Report: {output}")


@app.command()
def pipeline(
    run_id: Annotated[
        str,
        typer.Option(
            "--run-id",
            help="New immutable end-to-end run directory name under runs/.",
        ),
    ],
    archive: Archive = None,
    project_root: ProjectRoot = None,
    dataset: Dataset = "FD001",
    synthetic: Annotated[
        bool,
        typer.Option(
            "--synthetic",
            help=(
                "Run on generated FD001-shaped data instead of the NASA archive. "
                "Needs --project-root: a new directory outside the repository."
            ),
        ),
    ] = False,
) -> None:
    """Run prepare, train/lock, evaluate, and report."""
    if synthetic:
        archive, root = _synthetic_project(project_root, archive, dataset)
    else:
        root = resolve_project_root(project_root)
    try:
        result = run_pipeline(
            root,
            run_id=run_id,
            dataset=dataset,
            archive_path=archive,
        )
    except DeliveryError as exc:
        typer.echo(f"Pipeline failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("AeroMaintain end-to-end pipeline complete")
    typer.echo(f"Run: {result.run_id}")
    typer.echo(f"Manifest: {result.manifest_path}")
    typer.echo(f"Report: {result.report_path}")
    if synthetic:
        typer.echo("Data: SYNTHETIC. These numbers describe no real or NASA engine.")
        typer.echo(
            f"View it: aeromaintain app --run-id {result.run_id} --project-root {root}"
        )


def _synthetic_project(
    project_root: Path | None, archive: Path | None, dataset: str
) -> tuple[Path, Path]:
    from aeromaintain.synthetic import SyntheticProjectError, build_synthetic_project

    if archive is not None or dataset != "FD001":
        typer.echo(
            "--synthetic generates FD001 data; drop --archive/--dataset", err=True
        )
        raise typer.Exit(code=2)
    if project_root is None:
        typer.echo(
            "--synthetic needs --project-root: a directory outside the repository",
            err=True,
        )
        raise typer.Exit(code=2)
    repo_root = resolve_project_root(None)
    try:
        archive_path = build_synthetic_project(repo_root, project_root)
    except SyntheticProjectError as exc:
        typer.echo(f"Pipeline failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Synthetic project: {project_root}")
    return archive_path, project_root.resolve()


V5Output = Annotated[
    Path,
    typer.Option("--output", help="Output directory for this v5 stage."),
]


@app.command("v5")
def v5_command(
    stage: Annotated[
        str,
        typer.Argument(help="Stage of protocol v5: hybrid, prepare or evaluate."),
    ],
    output: V5Output,
    project_root: ProjectRoot = None,
) -> None:
    """Run a protocol v5 stage (frozen code; evaluate opens the reserved labels)."""
    import json

    from aeromaintain.models import v5

    stages = {"hybrid": v5.run_hybrid, "prepare": v5.prepare, "evaluate": v5.evaluate}
    if stage not in stages:
        typer.echo(f"Unknown v5 stage {stage!r}; use {', '.join(stages)}", err=True)
        raise typer.Exit(code=2)
    root = resolve_project_root(project_root)
    try:
        config = json.loads((root / v5.CONFIG).read_text(encoding="utf-8"))
        stages[stage](root, output.resolve(), config)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        # v5 signals refusals with these built-in errors (FileExistsError is an
        # OSError); its code is frozen, so they are translated here.
        typer.echo(f"v5 {stage} failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except subprocess.CalledProcessError as exc:
        typer.echo(f"v5 {stage} failed: git {exc.args[1]!r} did not succeed", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"v5 {stage} complete: {output.resolve()}")


@app.command()
def finish(
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="Existing locked run to evaluate and report."),
    ],
    project_root: ProjectRoot = None,
) -> None:
    """Open official labels for an existing lock and write its report."""
    root = resolve_project_root(project_root)
    try:
        result = finish_locked_run(root, run_id=run_id)
    except DeliveryError as exc:
        typer.echo(f"Finish failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if result.already_complete:
        typer.echo(f"Run {result.run_id} was already complete; nothing changed")
    else:
        typer.echo(f"Run complete: {result.run_id}")
    typer.echo(f"Report: {result.report_path}")


@app.command("export-demo")
def export_demo(
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="Verified run to export."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for the row-free demo bundle."),
    ] = Path("demo"),
    project_root: ProjectRoot = None,
) -> None:
    """Export aggregate metrics and predictions, without NASA rows or true RUL."""
    from aeromaintain.artifacts import ArtifactValidationError, export_public_bundle

    root = resolve_project_root(project_root)
    try:
        manifest = export_public_bundle(root, run_id, output_dir)
    except ArtifactValidationError as exc:
        typer.echo(f"Export failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Demo bundle manifest: {manifest}")


@app.command("app")
def app_command(
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="Explicit verified run to display."),
    ],
    project_root: ProjectRoot = None,
) -> None:
    """Open the local Streamlit application for one verified run."""
    from aeromaintain.artifacts import ArtifactValidationError, load_verified_run

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
