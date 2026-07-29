from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aeromaintain.data.pipeline import (
    SCHEMA,
    DataContract,
    DataPipelineError,
    add_train_rul,
    build_eda_summary,
    create_engine_split,
    extract_fd001_members,
    load_data_contract,
    parse_sensor_table,
    prepare_fd001,
    validate_sensor_table,
    verify_archive,
)


def _sensor_rows(engine_lifetimes: dict[int, int]) -> list[list[float]]:
    rows: list[list[float]] = []
    for unit_id, lifetime in engine_lifetimes.items():
        for cycle in range(1, lifetime + 1):
            settings = [cycle / 10, unit_id / 100, 1.0]
            sensors = [
                unit_id * 0.5 + cycle * (sensor_index + 1) / 100
                for sensor_index in range(21)
            ]
            rows.append([unit_id, cycle, *settings, *sensors])
    return rows


def _table_bytes(rows: list[list[float]]) -> bytes:
    return (
        "\n".join(" ".join(str(value) for value in row) for row in rows) + "\n"
    ).encode()


def _write_archive(
    path: Path,
    *,
    train_rows: list[list[float]],
    test_rows: list[list[float]],
    labels: list[int],
    malicious_member: str | None = None,
) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("CMAPSSData/train_FD001.txt", _table_bytes(train_rows))
        archive.writestr("CMAPSSData/test_FD001.txt", _table_bytes(test_rows))
        archive.writestr(
            "CMAPSSData/RUL_FD001.txt",
            ("\n".join(str(value) for value in labels) + "\n").encode(),
        )
        archive.writestr("CMAPSSData/train_FD002.txt", b"not selected\n")
        if malicious_member is not None:
            archive.writestr(malicious_member, b"malicious\n")


def _contract_for(
    archive: Path,
    *,
    train_rows: int,
    test_rows: int,
    engines: int,
    development: int,
    calibration: int,
) -> DataContract:
    return DataContract(
        source_url="https://example.invalid/CMAPSSData.zip",
        archive_size_bytes=archive.stat().st_size,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        train_rows=train_rows,
        test_rows=test_rows,
        train_engines=engines,
        test_engines=engines,
        test_labels=engines,
        development_engines=development,
        calibration_engines=calibration,
        rul_cap=4,
        seed=42,
    )


def test_project_configuration_loads_the_frozen_data_contract(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "project.yaml").write_text(
        """
project:
  seed: 42
data:
  source_url: https://example.invalid/CMAPSSData.zip
  archive_size_bytes: 123
  archive_sha256: ABCDEF
  development_engines: 80
  calibration_engines: 20
  rul_cap: 125
""".strip(),
        encoding="utf-8",
    )

    contract = load_data_contract(tmp_path)

    assert contract.archive_sha256 == "abcdef"
    assert contract.seed == 42
    assert contract.rul_cap == 125

    (config_dir / "project.yaml").write_text("data: {}", encoding="utf-8")
    with pytest.raises(DataPipelineError, match="Invalid project data configuration"):
        load_data_contract(tmp_path)


def test_archive_hash_and_path_traversal_are_rejected(tmp_path: Path) -> None:
    altered = tmp_path / "altered.zip"
    altered.write_bytes(b"altered")
    contract = DataContract(
        source_url="https://example.invalid/archive.zip",
        archive_size_bytes=len(b"altered"),
        archive_sha256="0" * 64,
    )
    with pytest.raises(DataPipelineError, match="SHA-256 mismatch"):
        verify_archive(altered, contract)

    malicious = tmp_path / "malicious.zip"
    rows = _sensor_rows({1: 2})
    _write_archive(
        malicious,
        train_rows=rows,
        test_rows=rows,
        labels=[1],
        malicious_member="../escaped.txt",
    )
    with pytest.raises(DataPipelineError, match="Unsafe ZIP member path"):
        extract_fd001_members(malicious, tmp_path / "raw")
    assert not (tmp_path / "escaped.txt").exists()


def test_extraction_selects_only_fd001_and_refuses_changed_raw_file(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    rows = _sensor_rows({1: 2})
    _write_archive(archive, train_rows=rows, test_rows=rows, labels=[1])
    destination = tmp_path / "raw"

    extracted = extract_fd001_members(archive, destination)
    assert set(extracted) == {"train", "test", "test_rul"}
    assert sorted(path.name for path in destination.iterdir()) == [
        "RUL_FD001.txt",
        "test_FD001.txt",
        "train_FD001.txt",
    ]
    assert extract_fd001_members(archive, destination) == extracted

    extracted["train"].write_text("changed", encoding="utf-8")
    with pytest.raises(DataPipelineError, match="Refusing to overwrite"):
        extract_fd001_members(archive, destination)


def test_parser_and_contract_reject_bad_schema_duplicate_and_nonfinite(
    tmp_path: Path,
) -> None:
    valid_path = tmp_path / "valid.txt"
    rows = _sensor_rows({1: 2, 2: 3})
    valid_path.write_bytes(_table_bytes(rows))
    frame = parse_sensor_table(valid_path)
    summary = validate_sensor_table(
        frame,
        name="fixture",
        expected_rows=5,
        expected_engines=2,
    )
    assert tuple(frame.columns) == SCHEMA
    assert summary["finite_values"]
    assert summary["ordered_cycles"]

    bad_width = tmp_path / "bad-width.txt"
    bad_width.write_text("1 2 3\n", encoding="utf-8")
    with pytest.raises(DataPipelineError, match="Expected 26 columns"):
        parse_sensor_table(bad_width)

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataPipelineError, match="duplicate"):
        validate_sensor_table(
            duplicate,
            name="duplicate",
            expected_rows=6,
            expected_engines=2,
        )

    nonfinite = frame.copy()
    nonfinite.loc[0, "sensor_1"] = np.inf
    with pytest.raises(DataPipelineError, match="non-finite"):
        validate_sensor_table(
            nonfinite,
            name="nonfinite",
            expected_rows=5,
            expected_engines=2,
        )


def test_rul_split_and_eda_are_engine_disjoint_and_deterministic() -> None:
    lifetimes = {unit_id: 4 + unit_id % 5 for unit_id in range(1, 21)}
    frame = pd.DataFrame(_sensor_rows(lifetimes), columns=SCHEMA)
    frame[["unit_id", "cycle"]] = frame[["unit_id", "cycle"]].astype("int64")
    frame[list(SCHEMA[2:])] = frame[list(SCHEMA[2:])].astype("float64")
    targeted = add_train_rul(frame, cap=4)
    assert targeted["rul_true"].min() == 0
    assert targeted["rul_target"].max() == 4

    first, first_manifest = create_engine_split(
        targeted,
        development_engines=16,
        calibration_engines=4,
        seed=42,
    )
    second, second_manifest = create_engine_split(
        targeted,
        development_engines=16,
        calibration_engines=4,
        seed=42,
    )
    assert first_manifest == second_manifest
    assert first.equals(second)
    development = set(first_manifest["development_engine_ids"])
    calibration = set(first_manifest["calibration_engine_ids"])
    assert len(development) == 16
    assert len(calibration) == 4
    assert development.isdisjoint(calibration)

    eda = build_eda_summary(first.loc[first["role"].eq("development")])
    assert eda["scope"] == "development engines only"
    assert eda["engines"] == 16
    assert len(eda["selected_trend_sensors"]) == 3


def test_prepare_writes_isolated_labels_reports_and_stable_hashes(
    tmp_path: Path,
) -> None:
    train_lifetimes = {unit_id: 4 + unit_id % 5 for unit_id in range(1, 21)}
    test_lifetimes = {unit_id: 2 + unit_id % 3 for unit_id in range(1, 21)}
    train_rows = _sensor_rows(train_lifetimes)
    test_rows = _sensor_rows(test_lifetimes)
    archive = tmp_path / "fixture.zip"
    _write_archive(
        archive,
        train_rows=train_rows,
        test_rows=test_rows,
        labels=[unit_id + 5 for unit_id in range(1, 21)],
    )
    contract = _contract_for(
        archive,
        train_rows=len(train_rows),
        test_rows=len(test_rows),
        engines=20,
        development=16,
        calibration=4,
    )
    project_root = tmp_path / "project"

    first = prepare_fd001(project_root, archive_path=archive, contract=contract)
    second = prepare_fd001(project_root, archive_path=archive, contract=contract)
    assert first.artifact_hashes == second.artifact_hashes
    assert first.development_engines == 16
    assert first.calibration_engines == 4

    output = project_root / "data" / "processed" / "fd001"
    assert (output / "evaluation" / "test_rul.parquet").is_file()
    train = pd.read_parquet(output / "train.parquet")
    assert "rul_true" in train
    assert "rul_target" in train
    assert "role" in train
    test = pd.read_parquet(output / "test.parquet")
    assert "rul_true" not in test
    assert "rul_target" not in test

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["official_test_labels"]["usage"].startswith("locked evaluation")
    assert manifest["artifacts"] == first.artifact_hashes
    eda = json.loads((output / "eda_summary.json").read_text(encoding="utf-8"))
    assert eda["scope"] == "development engines only"
    assert "<h1>FD001 EDA report</h1>" in (output / "eda_report.html").read_text(
        encoding="utf-8"
    )
