"""Run and demo-bundle artefacts: JSON contracts, hash checks and loading.

This layer has no UI dependency. The CLI, ``delivery`` and the Streamlit app all
load runs through it.

What the checks guarantee: every file a run or bundle relies on is listed in a
manifest with a SHA-256, sits inside its run or bundle directory, and still has
the listed bytes. What they do not guarantee: that a pickled model is safe to load.
A hash only shows that a file is unchanged since it was listed; a run directory
from someone else can list a malicious ``.joblib`` file with a correct hash.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationError,
    model_validator,
)

from aeromaintain.data.pipeline import SENSOR_COLUMNS, sha256_file

SAFE_PREDICTION_COLUMNS = (
    "unit_id",
    "cycle",
    "prediction",
    "interval_low",
    "interval_high",
    "risk_band",
)
RISK_BANDS = frozenset({"critical", "elevated", "routine"})
# Keys that would carry labels or NASA rows into a public bundle.
FORBIDDEN_PUBLIC_KEYS = frozenset({"rul_true", "sensor_history"})

PUBLIC_RUN_FILE = "run.json"
PUBLIC_RANKING_FILE = "risk_ranking.csv"
PUBLIC_MANIFEST_FILE = "manifest.json"
PUBLIC_REQUIRED_FILES = (PUBLIC_RUN_FILE, PUBLIC_RANKING_FILE)

REPORT_JSON = "report.json"
REPORT_HTML = "report.html"
EVALUATION_MANIFEST = "official_test/evaluation_manifest.json"
PIPELINE_REQUIRED_FILES = (REPORT_JSON, REPORT_HTML, EVALUATION_MANIFEST)
LOCK_REQUIRED_FILES = ("explanation.json",)

STATUS_MODEL_LOCKED = "model_locked"
STATUS_PIPELINE_COMPLETE = "pipeline_complete"


class ArtifactValidationError(RuntimeError):
    """Raised when a run or bundle cannot be trusted."""


@dataclass(frozen=True)
class RunArtifacts:
    """Validated model and evaluation data for reports and the review app."""

    run_id: str
    run_manifest: dict[str, Any]
    model_lock: dict[str, Any]
    metrics: dict[str, Any]
    explanation: dict[str, Any]
    risk_ranking: pd.DataFrame
    sensor_history: pd.DataFrame


# Older name, kept because callers and tests import it.
AppArtifacts = RunArtifacts


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ArtifactMap = dict[str, Sha256]


class _Open(BaseModel):
    """Run files written by earlier versions carry extra fields; keep accepting them."""

    model_config = ConfigDict(extra="allow")


class _Closed(BaseModel):
    """Public bundle files have one writer, so unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class PublicManifest(_Closed):
    artifacts: ArtifactMap


class _ProtocolSummary(_Open):
    label: str


class _ChampionSummary(_Open):
    kind: str


class PublicLockSummary(_Open):
    dataset: str
    protocol: _ProtocolSummary
    seed: int
    champion: _ChampionSummary
    calibration: dict[str, Any]


class PublicRunPayload(_Closed):
    run_id: str
    model_lock_sha256: Sha256
    model_lock: PublicLockSummary
    metrics: dict[str, Any]
    explanation: dict[str, Any]


class PipelineRecord(_Open):
    schema_version: Literal[3]
    status: Literal["complete"]
    artifacts: ArtifactMap


class RunManifest(_Open):
    run_id: str
    status: Literal["model_locked", "pipeline_complete"]
    model_lock_sha256: Sha256
    pipeline: PipelineRecord | None = None

    @model_validator(mode="after")
    def _pipeline_matches_status(self) -> RunManifest:
        complete = self.status == STATUS_PIPELINE_COMPLETE
        if complete != (self.pipeline is not None):
            raise ValueError("pipeline record must exist exactly when complete")
        return self


class _ConfigReference(_Open):
    path: str
    sha256: Sha256


class _LockedData(_Open):
    manifest_sha256: Sha256
    split_manifest_sha256: Sha256
    train_sha256: Sha256
    test_sha256: Sha256
    official_test_rul_sha256: Sha256


class ModelLock(_Open):
    schema_version: Literal[2, 3]
    run_id: str
    dataset: str
    seed: int
    protocol: dict[str, Any]
    config: _ConfigReference
    data: _LockedData
    champion: dict[str, Any]
    calibration: dict[str, Any]
    artifacts: ArtifactMap

    def referenced_files(self) -> set[str]:
        """Files the evaluator or the app opens, which the lock must therefore list."""
        names = set(LOCK_REQUIRED_FILES)
        for key in ("path", "preprocessor_path", "normalizer_path"):
            if isinstance(self.champion.get(key), str):
                names.add(self.champion[key])
        names.update(self.champion.get("member_paths", []))
        for key in ("quantile_model_path", "quantile_preprocessor_path"):
            if isinstance(self.calibration.get(key), str):
                names.add(self.calibration[key])
        return names


class EvaluationManifest(_Open):
    run_id: str
    model_lock_sha256: Sha256
    predictions_sha256: Sha256
    metrics_sha256: Sha256
    error_analysis_sha256: Sha256


ContractT = TypeVar("ContractT", bound=BaseModel)


def parse_contract(model: type[ContractT], payload: Any, label: str) -> ContractT:
    """Validate one JSON payload, turning schema errors into a domain error."""
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'root'}: {error['msg']}"
            for error in exc.errors()[:5]
        )
        raise ArtifactValidationError(
            f"{label} does not match its schema: {problems}"
        ) from exc


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactValidationError(f"Required artefact is missing: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            f"Artefact is unreadable or invalid: {path}"
        ) from exc


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ArtifactValidationError(f"Required artefact is missing: {path}")
    if sha256_file(path) != expected:
        raise ArtifactValidationError(f"{label} SHA-256 mismatch: {path}")


def resolve_within(base: Path, relative_path: str, label: str) -> Path:
    path = (base / relative_path).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as exc:
        raise ArtifactValidationError(
            f"{label} references an external path: {relative_path}"
        ) from exc
    return path


def verify_artifact_map(
    base: Path,
    artifacts: Mapping[str, str],
    label: str,
    *,
    required: Iterable[str] = (),
) -> None:
    """Check a non-empty, already schema-valid map that names every required file."""
    if not artifacts:
        raise ArtifactValidationError(f"{label} lists no artefacts")
    missing = sorted(set(required) - set(artifacts))
    if missing:
        raise ArtifactValidationError(
            f"{label} does not list required artefacts: {', '.join(missing)}"
        )
    for relative_path, expected in artifacts.items():
        require_hash(resolve_within(base, relative_path, label), expected, label)


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


def _find_forbidden_keys(value: Any, path: str = "") -> list[str]:
    found = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            here = f"{path}.{key}" if path else str(key)
            if key in FORBIDDEN_PUBLIC_KEYS:
                found.append(here)
            found.extend(_find_forbidden_keys(item, here))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_forbidden_keys(item, f"{path}[{index}]"))
    return found


def _check_public_ranking(ranking: pd.DataFrame) -> None:
    if list(ranking.columns) != list(SAFE_PREDICTION_COLUMNS):
        raise ArtifactValidationError(
            "Public bundle ranking must hold exactly the review columns: "
            + ", ".join(SAFE_PREDICTION_COLUMNS)
        )
    if ranking.empty:
        raise ArtifactValidationError("Public bundle ranking has no engines")
    numeric = ["unit_id", "cycle", "prediction", "interval_low", "interval_high"]
    if not all(pd.api.types.is_numeric_dtype(ranking[name]) for name in numeric) or (
        ranking[numeric].isna().any().any()
    ):
        raise ArtifactValidationError(
            "Public bundle ranking has non-numeric or missing values"
        )
    if not set(ranking["risk_band"]).issubset(RISK_BANDS):
        raise ArtifactValidationError("Public bundle ranking has unknown risk bands")


def load_run_manifest(run_dir: Path, run_id: str) -> RunManifest:
    manifest = parse_contract(
        RunManifest, read_json(run_dir / "manifest.json"), "Run manifest"
    )
    if manifest.run_id != run_id:
        raise ArtifactValidationError("Run manifest identity mismatch")
    return manifest


def _load_verified_run(project_root: Path, run_id: str) -> RunArtifacts:
    root = project_root.resolve()
    run_dir = _safe_run_dir(root, run_id)
    raw_manifest = read_json(run_dir / "manifest.json")
    manifest = load_run_manifest(run_dir, run_id)
    if manifest.pipeline is not None:
        verify_artifact_map(
            run_dir,
            manifest.pipeline.artifacts,
            "Pipeline",
            required=PIPELINE_REQUIRED_FILES,
        )

    lock_path = run_dir / "model_lock.json"
    require_hash(lock_path, manifest.model_lock_sha256, "Model lock")
    raw_lock = read_json(lock_path)
    lock = parse_contract(ModelLock, raw_lock, "Model lock")
    if lock.run_id != run_id:
        raise ArtifactValidationError("Model lock identity mismatch")
    verify_artifact_map(
        run_dir,
        lock.artifacts,
        "Model artefact",
        required=lock.referenced_files(),
    )
    require_hash(
        resolve_within(root, lock.config.path, "Model config"),
        lock.config.sha256,
        "Model config",
    )

    processed_dir = root / "data" / "processed" / lock.dataset.lower()
    for name, expected in (
        ("manifest.json", lock.data.manifest_sha256),
        ("split_manifest.json", lock.data.split_manifest_sha256),
        ("train.parquet", lock.data.train_sha256),
        ("test.parquet", lock.data.test_sha256),
        ("evaluation/test_rul.parquet", lock.data.official_test_rul_sha256),
    ):
        require_hash(processed_dir / name, expected, "Locked data")

    official_dir = run_dir / "official_test"
    evaluation = parse_contract(
        EvaluationManifest,
        read_json(official_dir / "evaluation_manifest.json"),
        "Evaluation manifest",
    )
    if evaluation.run_id != run_id:
        raise ArtifactValidationError("Evaluation manifest identity mismatch")
    require_hash(lock_path, evaluation.model_lock_sha256, "Evaluation model lock")
    for name, expected in (
        ("predictions.parquet", evaluation.predictions_sha256),
        ("metrics.json", evaluation.metrics_sha256),
        ("error_analysis.json", evaluation.error_analysis_sha256),
    ):
        require_hash(official_dir / name, expected, "Evaluation")

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

    test = pd.read_parquet(processed_dir / "test.parquet")
    sensor_history = test.loc[:, ["unit_id", "cycle", *SENSOR_COLUMNS]].copy()
    return RunArtifacts(
        run_id=run_id,
        run_manifest=raw_manifest,
        model_lock=raw_lock,
        metrics=read_json(official_dir / "metrics.json"),
        explanation=read_json(run_dir / "explanation.json"),
        risk_ranking=risk_ranking,
        sensor_history=sensor_history,
    )


def load_verified_run(project_root: Path, run_id: str) -> RunArtifacts:
    """Load a run after checking its manifests, schemas and complete hash chain."""
    try:
        return _load_verified_run(project_root, run_id)
    except ArtifactValidationError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"Verified run artefact structure is invalid: {exc}"
        ) from exc


def _write_replacing(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}-", suffix=".tmp", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export_public_bundle(project_root: Path, run_id: str, output_dir: Path) -> Path:
    """Write a hash-listed demo bundle holding no NASA rows or true RUL values.

    The manifest is written last, and the finished bundle is loaded back through
    ``load_public_bundle`` before this returns.
    """
    artifacts = load_verified_run(project_root, run_id)
    lock = artifacts.model_lock
    payload = PublicRunPayload.model_validate(
        {
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
    ).model_dump(mode="json")
    forbidden = _find_forbidden_keys(payload)
    if forbidden:
        raise ArtifactValidationError(
            "Run holds fields that must not be published: " + ", ".join(forbidden)
        )
    _check_public_ranking(artifacts.risk_ranking)
    files = {
        PUBLIC_RUN_FILE: json.dumps(payload, indent=2, sort_keys=True) + "\n",
        PUBLIC_RANKING_FILE: artifacts.risk_ranking.to_csv(
            index=False, lineterminator="\n"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, text in files.items():
        path = output_dir / name
        _write_replacing(path, text.encode("utf-8"))
        hashes[name] = sha256_file(path)
    manifest = output_dir / PUBLIC_MANIFEST_FILE
    _write_replacing(
        manifest,
        (json.dumps({"artifacts": hashes}, indent=2, sort_keys=True) + "\n").encode(),
    )
    load_public_bundle(output_dir)
    return manifest


def load_public_bundle(bundle_dir: Path) -> RunArtifacts:
    """Load a demo bundle after checking its schema, hashes and row-free contract."""
    manifest = parse_contract(
        PublicManifest,
        read_json(bundle_dir / PUBLIC_MANIFEST_FILE),
        "Public bundle manifest",
    )
    unexpected = sorted(set(manifest.artifacts) - set(PUBLIC_REQUIRED_FILES))
    if unexpected:
        raise ArtifactValidationError(
            "Public bundle lists unexpected artefacts: " + ", ".join(unexpected)
        )
    verify_artifact_map(
        bundle_dir,
        manifest.artifacts,
        "Public bundle",
        required=PUBLIC_REQUIRED_FILES,
    )
    raw_payload = read_json(bundle_dir / PUBLIC_RUN_FILE)
    forbidden = _find_forbidden_keys(raw_payload)
    if forbidden:
        raise ArtifactValidationError(
            "Public bundle holds forbidden fields: " + ", ".join(forbidden)
        )
    payload = parse_contract(PublicRunPayload, raw_payload, "Public bundle run")
    try:
        ranking = pd.read_csv(bundle_dir / PUBLIC_RANKING_FILE)
    except (OSError, ValueError) as exc:
        raise ArtifactValidationError(
            f"Public bundle ranking is unreadable: {exc}"
        ) from exc
    _check_public_ranking(ranking)
    return RunArtifacts(
        run_id=payload.run_id,
        run_manifest={"model_lock_sha256": payload.model_lock_sha256},
        model_lock=raw_payload["model_lock"],
        metrics=payload.metrics,
        explanation=payload.explanation,
        risk_ranking=ranking,
        sensor_history=pd.DataFrame(),
    )
