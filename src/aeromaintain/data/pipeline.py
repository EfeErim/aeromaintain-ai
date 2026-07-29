"""Deterministic, fail-closed preparation of NASA C-MAPSS FD001."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import yaml
from sklearn.model_selection import train_test_split

ARCHIVE_NAME = "CMAPSSData.zip"
EXPECTED_MEMBERS = {
    "train": "train_FD001.txt",
    "test": "test_FD001.txt",
    "test_rul": "RUL_FD001.txt",
}
SCHEMA = (
    "unit_id",
    "cycle",
    *(f"setting_{index}" for index in range(1, 4)),
    *(f"sensor_{index}" for index in range(1, 22)),
)
SENSOR_COLUMNS = tuple(f"sensor_{index}" for index in range(1, 22))
SPLIT_ALGORITHM_VERSION = "lifetime-quartile-stratified-v1"


class DataPipelineError(RuntimeError):
    """Raised when an acquisition or data-contract check fails."""


@dataclass(frozen=True)
class DataContract:
    """Frozen acquisition and FD001 validation values."""

    source_url: str
    archive_size_bytes: int
    archive_sha256: str
    train_rows: int = 20_631
    test_rows: int = 13_096
    train_engines: int = 100
    test_engines: int = 100
    test_labels: int = 100
    development_engines: int = 80
    calibration_engines: int = 20
    rul_cap: int = 125
    seed: int = 42


@dataclass(frozen=True)
class PrepareResult:
    """Small, CLI-safe summary of a successful preparation."""

    output_dir: Path
    train_rows: int
    test_rows: int
    development_engines: int
    calibration_engines: int
    artifact_hashes: dict[str, str]


def sha256_file(path: Path) -> str:
    """Return a file's lowercase SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data_contract(project_root: Path) -> DataContract:
    """Load the fixed data contract from the project configuration."""
    config_path = project_root / "configs" / "project.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data = config["data"]
        project = config["project"]
        return DataContract(
            source_url=str(data["source_url"]),
            archive_size_bytes=int(data["archive_size_bytes"]),
            archive_sha256=str(data["archive_sha256"]).lower(),
            development_engines=int(data["development_engines"]),
            calibration_engines=int(data["calibration_engines"]),
            rul_cap=int(data["rul_cap"]),
            seed=int(project["seed"]),
        )
    except (KeyError, TypeError, ValueError, OSError, yaml.YAMLError) as exc:
        raise DataPipelineError(f"Invalid project data configuration: {exc}") from exc


def verify_archive(path: Path, contract: DataContract) -> None:
    """Reject an archive unless both byte size and SHA-256 match."""
    if not path.is_file():
        raise DataPipelineError(f"Archive does not exist: {path}")
    observed_size = path.stat().st_size
    if observed_size != contract.archive_size_bytes:
        raise DataPipelineError(
            "Archive size mismatch: "
            f"expected {contract.archive_size_bytes}, observed {observed_size}"
        )
    observed_hash = sha256_file(path)
    if observed_hash.lower() != contract.archive_sha256.lower():
        raise DataPipelineError(
            "Archive SHA-256 mismatch: "
            f"expected {contract.archive_sha256.lower()}, observed {observed_hash}"
        )


def _copy_verified_archive(
    source: Path,
    destination: Path,
    contract: DataContract,
) -> Path:
    verify_archive(source, contract)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_archive(destination, contract)
        return destination
    if source.resolve() == destination.resolve():
        return destination

    temporary = destination.with_suffix(f"{destination.suffix}.part")
    if temporary.exists():
        temporary.unlink()
    try:
        shutil.copyfile(source, temporary)
        verify_archive(temporary, contract)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def acquire_archive(
    raw_dir: Path,
    contract: DataContract,
    archive_path: Path | None = None,
) -> Path:
    """Use a verified local archive or download the frozen NASA archive."""
    destination = raw_dir / ARCHIVE_NAME
    if archive_path is not None:
        return _copy_verified_archive(archive_path.resolve(), destination, contract)
    if destination.exists():
        verify_archive(destination, contract)
        return destination

    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="cmapss-",
        suffix=".zip.part",
        dir=raw_dir,
        delete=False,
    ) as temporary_stream:
        temporary = Path(temporary_stream.name)
        try:
            with urllib.request.urlopen(contract.source_url, timeout=120) as response:
                shutil.copyfileobj(response, temporary_stream)
        except Exception as exc:
            raise DataPipelineError(
                f"Could not download NASA archive from {contract.source_url}: {exc}"
            ) from exc

    try:
        verify_archive(temporary, contract)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _validated_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    member_path = PurePosixPath(normalized)
    first_part = member_path.parts[0] if member_path.parts else ""
    if (
        not normalized
        or member_path.is_absolute()
        or ".." in member_path.parts
        or ":" in first_part
    ):
        raise DataPipelineError(f"Unsafe ZIP member path: {name!r}")
    return member_path


def _write_bytes_idempotently(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise DataPipelineError(
                f"Refusing to overwrite existing file with different content: {path}"
            )
        return
    path.write_bytes(payload)


def extract_fd001_members(archive_path: Path, destination: Path) -> dict[str, Path]:
    """Safely select and materialize only the three FD001 source members."""
    selected: dict[str, zipfile.ZipInfo] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                member_path = _validated_member_path(info.filename)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise DataPipelineError(
                        f"Symbolic-link ZIP member is not allowed: {info.filename!r}"
                    )
                basename = member_path.name
                for role, expected_basename in EXPECTED_MEMBERS.items():
                    if basename == expected_basename:
                        if role in selected:
                            raise DataPipelineError(
                                f"Duplicate FD001 ZIP member: {expected_basename}"
                            )
                        selected[role] = info

            missing = [
                basename
                for role, basename in EXPECTED_MEMBERS.items()
                if role not in selected
            ]
            if missing:
                raise DataPipelineError(
                    f"Archive is missing required FD001 members: {', '.join(missing)}"
                )

            extracted: dict[str, Path] = {}
            for role, info in selected.items():
                target = destination / EXPECTED_MEMBERS[role]
                _write_bytes_idempotently(target, archive.read(info))
                extracted[role] = target
            return extracted
    except (OSError, zipfile.BadZipFile) as exc:
        raise DataPipelineError(f"Invalid ZIP archive {archive_path}: {exc}") from exc


def parse_sensor_table(path: Path) -> pd.DataFrame:
    """Parse a whitespace-delimited C-MAPSS table into the fixed schema."""
    try:
        frame = pd.read_csv(path, sep=r"\s+", header=None)
    except (OSError, pd.errors.ParserError) as exc:
        raise DataPipelineError(f"Could not parse sensor table {path}: {exc}") from exc
    if frame.shape[1] != len(SCHEMA):
        raise DataPipelineError(
            f"Expected {len(SCHEMA)} columns in {path.name}, observed {frame.shape[1]}"
        )
    frame.columns = list(SCHEMA)
    try:
        frame = frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataPipelineError(f"Non-numeric value in {path.name}: {exc}") from exc

    for column in ("unit_id", "cycle"):
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.equal(values, np.floor(values)).all():
            raise DataPipelineError(f"{column} must contain integers in {path.name}")
        frame[column] = frame[column].astype("int64")
    for column in SCHEMA[2:]:
        frame[column] = frame[column].astype("float64")
    return frame


def parse_test_rul(path: Path, expected_labels: int) -> pd.DataFrame:
    """Parse the protected official test RUL vector."""
    try:
        frame = pd.read_csv(path, sep=r"\s+", header=None)
    except (OSError, pd.errors.ParserError) as exc:
        raise DataPipelineError(
            f"Could not parse test RUL table {path}: {exc}"
        ) from exc
    if frame.shape != (expected_labels, 1):
        raise DataPipelineError(
            "Official test RUL shape mismatch: "
            f"expected ({expected_labels}, 1), observed {frame.shape}"
        )
    values = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise DataPipelineError("Official test RUL contains invalid values")
    if not np.equal(values, np.floor(values)).all() or (values < 0).any():
        raise DataPipelineError("Official test RUL must contain non-negative integers")
    return pd.DataFrame(
        {
            "unit_id": np.arange(1, expected_labels + 1, dtype=np.int64),
            "rul_true": values.astype("int64"),
        }
    )


def validate_sensor_table(
    frame: pd.DataFrame,
    *,
    name: str,
    expected_rows: int,
    expected_engines: int,
) -> dict[str, Any]:
    """Enforce dimensions, keys, ordering, finite values, and numeric types."""
    if tuple(frame.columns) != SCHEMA:
        raise DataPipelineError(f"{name} does not match the fixed 26-column schema")
    if len(frame) != expected_rows:
        raise DataPipelineError(
            f"{name} row count mismatch: expected {expected_rows}, "
            f"observed {len(frame)}"
        )
    engine_count = int(frame["unit_id"].nunique())
    if engine_count != expected_engines:
        raise DataPipelineError(
            f"{name} engine count mismatch: "
            f"expected {expected_engines}, observed {engine_count}"
        )
    if (frame[["unit_id", "cycle"]] <= 0).any().any():
        raise DataPipelineError(f"{name} unit_id and cycle values must be positive")
    if frame.duplicated(["unit_id", "cycle"]).any():
        raise DataPipelineError(f"{name} contains duplicate (unit_id, cycle) keys")
    if frame.isna().any().any():
        raise DataPipelineError(f"{name} contains null values")
    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise DataPipelineError(f"{name} contains non-finite values")

    ordered = frame.sort_values(["unit_id", "cycle"], kind="stable").index
    if not ordered.equals(frame.index):
        raise DataPipelineError(f"{name} rows are not ordered by unit_id and cycle")
    cycle_steps = frame.groupby("unit_id", sort=False)["cycle"].diff().dropna()
    if not cycle_steps.eq(1).all():
        raise DataPipelineError(f"{name} cycles must increase one step at a time")

    return {
        "rows": len(frame),
        "engines": engine_count,
        "columns": int(frame.shape[1]),
        "null_values": int(frame.isna().sum().sum()),
        "duplicate_keys": int(frame.duplicated(["unit_id", "cycle"]).sum()),
        "finite_values": True,
        "ordered_cycles": True,
    }


def add_train_rul(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Add uncapped physical target and capped modeling target."""
    result = frame.copy()
    engine_max_cycle = result.groupby("unit_id")["cycle"].transform("max")
    result["rul_true"] = (engine_max_cycle - result["cycle"]).astype("int64")
    result["rul_target"] = result["rul_true"].clip(upper=cap).astype("int64")
    if (result[["rul_true", "rul_target"]] < 0).any().any():
        raise DataPipelineError("Generated train RUL contains negative values")
    return result


def create_engine_split(
    train: pd.DataFrame,
    *,
    development_engines: int,
    calibration_engines: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a deterministic engine-only split stratified by lifetime quartile."""
    lifetimes = train.groupby("unit_id", sort=True)["cycle"].max()
    total = development_engines + calibration_engines
    if len(lifetimes) != total:
        raise DataPipelineError(
            f"Split requires {total} engines, observed {len(lifetimes)}"
        )
    quartiles = pd.qcut(
        lifetimes.rank(method="first"),
        q=4,
        labels=False,
    ).astype("int64")
    unit_ids = lifetimes.index.to_numpy(dtype=np.int64)
    development, calibration = train_test_split(
        unit_ids,
        test_size=calibration_engines,
        random_state=seed,
        shuffle=True,
        stratify=quartiles.to_numpy(),
    )
    development_ids = sorted(int(value) for value in development)
    calibration_ids = sorted(int(value) for value in calibration)
    if set(development_ids) & set(calibration_ids):
        raise DataPipelineError("Development and calibration engine roles overlap")

    role_by_engine = {
        **dict.fromkeys(development_ids, "development"),
        **dict.fromkeys(calibration_ids, "calibration"),
    }
    result = train.copy()
    result["role"] = result["unit_id"].map(role_by_engine)
    if result["role"].isna().any():
        raise DataPipelineError("At least one train engine has no assigned role")

    manifest = {
        "algorithm": SPLIT_ALGORITHM_VERSION,
        "seed": seed,
        "stratification": "engine maximum cycle lifetime quartiles",
        "development_engine_ids": development_ids,
        "calibration_engine_ids": calibration_ids,
        "engine_lifetimes": {
            str(int(unit_id)): int(lifetime) for unit_id, lifetime in lifetimes.items()
        },
        "engine_lifetime_quartiles": {
            str(int(unit_id)): int(quartile) for unit_id, quartile in quartiles.items()
        },
    }
    return result, manifest


def build_eda_summary(development: pd.DataFrame) -> dict[str, Any]:
    """Summarize only development engines for leakage-safe EDA."""
    if development.empty or development["role"].ne("development").any():
        raise DataPipelineError("EDA input must contain development engines only")
    lifetimes = development.groupby("unit_id")["cycle"].max()
    sensor_variability: dict[str, Any] = {}
    outliers: dict[str, int] = {}
    for column in SENSOR_COLUMNS:
        series = development[column]
        first_quartile = float(series.quantile(0.25))
        third_quartile = float(series.quantile(0.75))
        iqr = third_quartile - first_quartile
        lower = first_quartile - 1.5 * iqr
        upper = third_quartile + 1.5 * iqr
        sensor_variability[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "unique_values": int(series.nunique()),
        }
        outliers[column] = int(((series < lower) | (series > upper)).sum())

    selected_sensors = sorted(
        SENSOR_COLUMNS,
        key=lambda name: (-sensor_variability[name]["std"], name),
    )[:3]
    return {
        "scope": "development engines only",
        "rows": len(development),
        "engines": int(development["unit_id"].nunique()),
        "null_counts": {
            column: int(count) for column, count in development.isna().sum().items()
        },
        "engine_lifetime": {
            "min": int(lifetimes.min()),
            "max": int(lifetimes.max()),
            "mean": float(lifetimes.mean()),
            "median": float(lifetimes.median()),
        },
        "sensor_variability": sensor_variability,
        "iqr_outlier_counts": outliers,
        "selected_trend_sensors": selected_sensors,
    }


def build_eda_html(
    development: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    """Create a deterministic interactive sensor-trend report."""
    sample_engine_ids = sorted(development["unit_id"].unique())[:5]
    selected_sensors = summary["selected_trend_sensors"]
    sample = development.loc[
        development["unit_id"].isin(sample_engine_ids),
        ["unit_id", "cycle", *selected_sensors],
    ]
    long_sample = sample.melt(
        id_vars=["unit_id", "cycle"],
        value_vars=selected_sensors,
        var_name="sensor",
        value_name="value",
    )
    long_sample["unit_id"] = long_sample["unit_id"].astype(str)
    figure = px.line(
        long_sample,
        x="cycle",
        y="value",
        color="unit_id",
        facet_row="sensor",
        title="FD001 development-only selected sensor trends",
        labels={"unit_id": "Engine", "cycle": "Cycle", "value": "Sensor value"},
    )
    figure.update_yaxes(matches=None)
    chart = figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        div_id="fd001-development-sensor-trends",
    )
    summary_json = json.dumps(summary, indent=2, sort_keys=True)
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        "<title>FD001 EDA report</title></head><body>"
        "<h1>FD001 EDA report</h1>"
        "<p>This report uses development engines only. The source is simulated "
        "NASA C-MAPSS data and is not operational fleet telemetry.</p>"
        f"{chart}<h2>Machine-readable summary</h2><pre>{summary_json}</pre>"
        "</body></html>\n"
    )


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, engine="pyarrow")
    return buffer.getvalue()


def _write_outputs(
    output_dir: Path,
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    test_rul: pd.DataFrame,
    split_manifest: dict[str, Any],
    quality_report: dict[str, Any],
    eda_summary: dict[str, Any],
    eda_html: str,
    contract: DataContract,
    archive_hash: str,
    config_hash: str,
) -> dict[str, str]:
    payloads = {
        "train.parquet": _parquet_bytes(train),
        "test.parquet": _parquet_bytes(test),
        "evaluation/test_rul.parquet": _parquet_bytes(test_rul),
        "split_manifest.json": _json_bytes(split_manifest),
        "data_quality_report.json": _json_bytes(quality_report),
        "eda_summary.json": _json_bytes(eda_summary),
        "eda_report.html": eda_html.encode(),
    }
    artifact_hashes: dict[str, str] = {}
    for relative_path, payload in payloads.items():
        _write_bytes_idempotently(output_dir / relative_path, payload)
        artifact_hashes[relative_path] = hashlib.sha256(payload).hexdigest()

    manifest = {
        "dataset": "FD001",
        "archive": {
            "filename": ARCHIVE_NAME,
            "sha256": archive_hash,
            "size_bytes": contract.archive_size_bytes,
        },
        "config_sha256": config_hash,
        "data_contract": asdict(contract),
        "split_algorithm": SPLIT_ALGORITHM_VERSION,
        "official_test_labels": {
            "path": "evaluation/test_rul.parquet",
            "usage": "locked evaluation only; excluded from training paths",
        },
        "artifacts": artifact_hashes,
    }
    _write_bytes_idempotently(output_dir / "manifest.json", _json_bytes(manifest))
    return artifact_hashes


def prepare_fd001(
    project_root: Path,
    *,
    archive_path: Path | None = None,
    contract: DataContract | None = None,
) -> PrepareResult:
    """Run acquisition, parsing, validation, splitting, EDA, and persistence."""
    project_root = project_root.resolve()
    active_contract = contract or load_data_contract(project_root)
    config_path = project_root / "configs" / "project.yaml"
    config_hash = sha256_file(config_path) if config_path.is_file() else "test-contract"

    raw_dir = project_root / "data" / "raw"
    archive = acquire_archive(raw_dir, active_contract, archive_path)
    extracted = extract_fd001_members(archive, raw_dir / "fd001")

    train = parse_sensor_table(extracted["train"])
    test = parse_sensor_table(extracted["test"])
    train_quality = validate_sensor_table(
        train,
        name="train",
        expected_rows=active_contract.train_rows,
        expected_engines=active_contract.train_engines,
    )
    test_quality = validate_sensor_table(
        test,
        name="test",
        expected_rows=active_contract.test_rows,
        expected_engines=active_contract.test_engines,
    )
    test_rul = parse_test_rul(extracted["test_rul"], active_contract.test_labels)
    if set(test["unit_id"].unique()) != set(test_rul["unit_id"]):
        raise DataPipelineError("Official test RUL engine IDs do not match test data")

    train = add_train_rul(train, active_contract.rul_cap)
    train, split_manifest = create_engine_split(
        train,
        development_engines=active_contract.development_engines,
        calibration_engines=active_contract.calibration_engines,
        seed=active_contract.seed,
    )
    development = train.loc[train["role"].eq("development")].copy()
    eda_summary = build_eda_summary(development)
    eda_html = build_eda_html(development, eda_summary)
    quality_report = {
        "dataset": "FD001",
        "train": train_quality,
        "test": test_quality,
        "test_rul": {
            "rows": len(test_rul),
            "engines": int(test_rul["unit_id"].nunique()),
            "negative_values": int(test_rul["rul_true"].lt(0).sum()),
        },
        "train_targets": {
            "rul_true_min": int(train["rul_true"].min()),
            "rul_true_max": int(train["rul_true"].max()),
            "rul_target_min": int(train["rul_target"].min()),
            "rul_target_max": int(train["rul_target"].max()),
            "rul_cap": active_contract.rul_cap,
            "negative_values": int(train[["rul_true", "rul_target"]].lt(0).sum().sum()),
        },
    }
    output_dir = project_root / "data" / "processed" / "fd001"
    artifact_hashes = _write_outputs(
        output_dir,
        train=train,
        test=test,
        test_rul=test_rul,
        split_manifest=split_manifest,
        quality_report=quality_report,
        eda_summary=eda_summary,
        eda_html=eda_html,
        contract=active_contract,
        archive_hash=sha256_file(archive),
        config_hash=config_hash,
    )
    return PrepareResult(
        output_dir=output_dir,
        train_rows=len(train),
        test_rows=len(test),
        development_engines=len(split_manifest["development_engine_ids"]),
        calibration_engines=len(split_manifest["calibration_engine_ids"]),
        artifact_hashes=artifact_hashes,
    )
