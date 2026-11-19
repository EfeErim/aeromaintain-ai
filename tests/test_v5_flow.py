"""Protocol v5 prepare, lock and evaluate, end to end on synthetic HDF5 files."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("h5py")

from ncmapss_fixture import REPO, TEST_SPLITS, build_project, commit_all

from aeromaintain.models import v5

OUTPUT = "runs/v5/final"
MEASURED = [
    f"{kind}_{index}_{stat}"
    for kind, count in (("sensor", 14), ("setting", 4))
    for index in range(count)
    for stat in ("mean", "std", "min", "max")
]


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> tuple[Path, dict]:
    """A committed project whose v5 lock has been written and committed."""
    root = tmp_path_factory.mktemp("v5") / "project"
    root.mkdir()
    config = build_project(root)
    v5.prepare(root, root / OUTPUT, config)
    commit_all(root, "publish v5 lock")
    return root, config


@pytest.fixture
def project(prepared, tmp_path) -> tuple[Path, dict]:
    """A private copy, so a failing evaluation cannot affect other tests."""
    root, config = prepared
    copy = tmp_path / "project"
    shutil.copytree(root, copy)
    return copy, config


@pytest.fixture(scope="module")
def evaluated(prepared, tmp_path_factory) -> tuple[Path, dict, dict]:
    root, config = prepared
    copy = tmp_path_factory.mktemp("v5-evaluated") / "project"
    shutil.copytree(root, copy)
    report = v5.evaluate(copy, copy / OUTPUT, config)
    return copy, config, report


def _labels_must_stay_closed(monkeypatch) -> list:
    calls = []

    def guarded(*args, **kwargs):
        calls.append(args)
        raise AssertionError("labels were read before every check passed")

    monkeypatch.setattr(v5, "read_multiple_targets", guarded)
    return calls


def test_lock_pins_sources_features_and_engines(prepared) -> None:
    root, _ = prepared
    lock = json.loads((root / v5.PUBLISHED_LOCK).read_text(encoding="utf-8"))
    output_lock = (root / OUTPUT / "model_lock.json").read_bytes()
    assert (root / v5.PUBLISHED_LOCK).read_bytes() == output_lock
    # The copied sources hash exactly like the frozen files in this repository.
    assert lock["source_sha256"] == v5.source_hashes(REPO)
    assert lock["features"] == MEASURED
    assert len(lock["training_engines"]) == v5.TRAINING_ENGINES
    # Training engines are namespaced by file offset 0-5, apart from the test files.
    assert {e // 1000 for e in lock["training_engines"]} == set(range(6))
    expected_tests = {
        name: list(
            range(
                1000 * (v5.TEST_NAMESPACE + i) + dev + 1,
                1000 * (v5.TEST_NAMESPACE + i) + dev + test + 1,
            )
        )
        for i, (name, (dev, test)) in enumerate(TEST_SPLITS.items())
    }
    assert {
        name: units["test"] for name, units in lock["test_engines"].items()
    } == expected_tests
    assert lock["chosen"] in v5.CANDIDATES
    assert lock["development_sha256"] == v5.sha256_file(root / v5.DEVELOPMENT)


def test_evaluate_writes_expected_outputs(evaluated) -> None:
    root, config, report = evaluated
    output = root / OUTPUT
    assert json.loads(
        (output / "evidence.json").read_text(encoding="utf-8")
    ) == json.loads(json.dumps(report))
    assert set(report) >= {
        "point",
        "baseline",
        "per_file",
        "intervals",
        "replay",
        "bootstrap_95",
        "gates",
    }
    assert set(report["per_file"]) == set(config["test_files"])
    assert report["point"]["engines"] == v5.TEST_ENGINES
    assert report["gates"] == v5.final_gates(
        report["point"],
        report["baseline"],
        report["intervals"]["mondrian"]["overall"]["coverage"],
        report["replay"]["point_timely"],
        config["gates"],
    )

    frame = pd.read_parquet(output / "evaluation.parquet")
    assert list(frame.columns) == [
        "source_file",
        "archive_unit_id",
        "unit_id",
        "cycle",
        "rul_true",
        "prediction",
        "mondrian_low",
        "mondrian_high",
        "split_conformal_low",
        "split_conformal_high",
    ]
    assert frame.unit_id.nunique() == v5.TEST_ENGINES
    assert (frame.unit_id // 1000 - v5.TEST_NAMESPACE).between(0, 2).all()
    assert (frame.prediction >= 0).all()
    assert (frame.mondrian_low <= frame.prediction).all()
    assert (frame.prediction <= frame.mondrian_high).all()

    ledger = json.loads((root / v5.LEDGER).read_text(encoding="utf-8"))
    assert set(ledger) == {v5.LEDGER_KEY}


def test_evaluate_refuses_a_second_run(evaluated) -> None:
    root, config, _ = evaluated
    with pytest.raises(FileExistsError, match="already exists"):
        v5.evaluate(root, root / OUTPUT, config)


def test_changed_test_file_stops_before_labels(project, monkeypatch) -> None:
    root, config = project
    calls = _labels_must_stay_closed(monkeypatch)
    test_file = root / config["raw_dir"] / config["test_files"][0]
    with test_file.open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(RuntimeError, match="differs from the reserved file"):
        v5.evaluate(root, root / OUTPUT, config)
    assert calls == []
    assert not (root / v5.LEDGER).exists()


def test_changed_source_stops_before_labels(project, monkeypatch) -> None:
    root, config = project
    calls = _labels_must_stay_closed(monkeypatch)
    source = root / "src/aeromaintain/evaluation/v4.py"
    source.write_bytes(source.read_bytes() + b"# edited\n")
    commit_all(root, "edit a frozen source")
    with pytest.raises(RuntimeError, match="Source code differs"):
        v5.evaluate(root, root / OUTPUT, config)
    assert calls == []
    assert not (root / v5.LEDGER).exists()


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (v5.CONFIG, "config hash differs"),
        (v5.PROTOCOL, "protocol hash differs"),
    ],
)
def test_changed_frozen_document_stops_before_labels(
    project, monkeypatch, path, message
) -> None:
    root, config = project
    calls = _labels_must_stay_closed(monkeypatch)
    target = root / path
    target.write_bytes(target.read_bytes() + b"\n")
    commit_all(root, "edit a frozen document")
    with pytest.raises(RuntimeError, match=message):
        v5.evaluate(root, root / OUTPUT, config)
    assert calls == []
    assert not (root / v5.LEDGER).exists()


def test_uncommitted_change_stops_evaluation(project, monkeypatch) -> None:
    root, config = project
    calls = _labels_must_stay_closed(monkeypatch)
    (root / "notes.txt").write_text("draft\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean, committed"):
        v5.evaluate(root, root / OUTPUT, config)
    assert calls == []


def test_existing_ledger_claim_blocks_another_lock(project, monkeypatch) -> None:
    root, config = project
    calls = _labels_must_stay_closed(monkeypatch)
    ledger = root / v5.LEDGER
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps({v5.LEDGER_KEY: {"model_lock_sha256": "0" * 64}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="already opened"):
        v5.evaluate(root, root / OUTPUT, config)
    assert calls == []
