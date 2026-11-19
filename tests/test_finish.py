"""Resuming ``finish`` after interruptions, and what it refuses to do."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from run_fixture import build_verified_run
from typer.testing import CliRunner

from aeromaintain import delivery
from aeromaintain.artifacts import load_verified_run
from aeromaintain.cli import app
from aeromaintain.delivery import DeliveryError, finish_locked_run, run_lock

RUN = "fixture"


def _never_evaluate(root, *, run_id):
    raise AssertionError("a recorded evaluation must not be rerun")


def _status(run_dir: Path) -> str:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["status"]


def _snapshot(run_dir: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(run_dir)): path.read_bytes()
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    }


def _finish(root: Path, **kwargs):
    kwargs.setdefault("evaluation_step", _never_evaluate)
    return finish_locked_run(root, run_id=RUN, **kwargs)


def _assert_completed(root: Path, run_dir: Path) -> None:
    assert _status(run_dir) == "pipeline_complete"
    load_verified_run(root, RUN)
    assert not list(run_dir.glob(".*.tmp"))


@pytest.fixture
def run(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path, build_verified_run(tmp_path, RUN)


@pytest.fixture
def reference(tmp_path_factory) -> dict[str, bytes]:
    """The files of an uninterrupted finish, to compare resumed runs against."""
    root = tmp_path_factory.mktemp("reference")
    run_dir = build_verified_run(root, RUN)
    _finish(root)
    return _snapshot(run_dir)


def _fail_on_call(monkeypatch, name: str, failing_call: int) -> None:
    original = getattr(delivery, name)
    calls = []

    def flaky(*args, **kwargs):
        calls.append(args)
        if len(calls) == failing_call:
            raise OSError(f"injected failure in {name} call {failing_call}")
        return original(*args, **kwargs)

    monkeypatch.setattr(delivery, name, flaky)


@pytest.mark.parametrize(
    ("label", "function", "failing_call", "reports_left"),
    [
        ("before the first report", "write_once", 1, []),
        ("after report.json", "write_once", 2, ["report.json"]),
        ("after report.html", "write_replacing", 1, ["report.html", "report.json"]),
    ],
)
def test_finish_resumes_after_interruption(
    run, reference, monkeypatch, label, function, failing_call, reports_left
) -> None:
    root, run_dir = run
    _fail_on_call(monkeypatch, function, failing_call)
    with pytest.raises(DeliveryError, match="injected failure"):
        _finish(root)
    assert _status(run_dir) == "model_locked", label
    present = sorted(p.name for p in run_dir.glob("report.*"))
    assert present == reports_left, label

    monkeypatch.undo()
    result = _finish(root)
    assert not result.already_complete
    _assert_completed(root, run_dir)
    assert _snapshot(run_dir) == reference


def test_finish_resumes_when_manifest_replace_fails(run, reference, monkeypatch):
    root, run_dir = run
    manifest = run_dir / "manifest.json"
    before = manifest.read_bytes()
    original_replace = Path.replace

    def failing_replace(self, target):
        if Path(target).name == "manifest.json":
            raise OSError("injected failure while replacing the manifest")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(DeliveryError, match="replacing the manifest"):
        _finish(root)
    assert manifest.read_bytes() == before
    assert not list(run_dir.glob(".manifest.json-*"))

    monkeypatch.undo()
    _finish(root)
    _assert_completed(root, run_dir)
    assert _snapshot(run_dir) == reference


def test_conflicting_report_is_not_overwritten(run) -> None:
    root, run_dir = run
    (run_dir / "report.json").write_text('{"edited": true}\n', encoding="utf-8")
    with pytest.raises(DeliveryError, match="Refusing to overwrite"):
        _finish(root)
    assert (run_dir / "report.json").read_text(encoding="utf-8") == '{"edited": true}\n'
    assert _status(run_dir) == "model_locked"


def test_finish_on_completed_run_changes_nothing(run) -> None:
    root, run_dir = run
    _finish(root)
    before = _snapshot(run_dir)
    result = _finish(root)
    assert result.already_complete
    assert _snapshot(run_dir) == before


def test_partial_evaluation_stops_without_rerunning(run) -> None:
    root, run_dir = run
    (run_dir / "official_test" / "evaluation_manifest.json").unlink()
    with pytest.raises(DeliveryError, match="stopped part-way"):
        _finish(root)
    assert _status(run_dir) == "model_locked"


def test_missing_evaluation_runs_once_then_is_reused(run, tmp_path) -> None:
    root, run_dir = run
    saved = tmp_path / "saved_official"
    shutil.move(run_dir / "official_test", saved)
    calls = []

    def evaluate(root, *, run_id):
        calls.append(run_id)
        shutil.copytree(saved, run_dir / "official_test")

    def fail_load(root, run_id):
        raise OSError("injected failure after evaluation")

    with pytest.raises(DeliveryError, match="after evaluation"):
        _finish(root, evaluation_step=evaluate, load_step=fail_load)
    assert calls == [RUN]

    _finish(root, evaluation_step=evaluate)
    assert calls == [RUN]
    _assert_completed(root, run_dir)


def test_second_finish_is_refused_while_one_holds_the_run(run) -> None:
    root, run_dir = run
    with run_lock(root, RUN), pytest.raises(DeliveryError, match="Another finish"):
        _finish(root)
    assert _status(run_dir) == "model_locked"
    _finish(root)
    _assert_completed(root, run_dir)


def test_lock_held_by_another_process_blocks_finish(run, tmp_path) -> None:
    root, run_dir = run
    ready = tmp_path / "holder-ready"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, sys; from pathlib import Path\n"
            "from aeromaintain.delivery import run_lock\n"
            "with run_lock(Path(sys.argv[1]), sys.argv[2]):\n"
            "    Path(sys.argv[3]).write_text('ready')\n"
            "    sys.stdin.buffer.readline()\n"
            "    os._exit(0)\n",
            str(root),
            RUN,
            str(ready),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 20
        while not ready.exists():
            assert holder.poll() is None, "lock holder exited early"
            assert time.monotonic() < deadline, "lock holder did not start"
            time.sleep(0.05)
        with pytest.raises(DeliveryError, match="Another finish"):
            _finish(root)
    finally:
        # Exit inside the lock holder: Windows venv launchers can have a child
        # interpreter that survives killing the launcher process.
        holder.communicate(input=b"exit\n", timeout=20)
    # The OS released the dead holder's lock; no manual cleanup is needed.
    _finish(root)
    _assert_completed(root, run_dir)


def test_cli_finish_exit_codes(run) -> None:
    root, _ = run
    runner = CliRunner()
    args = ["finish", "--run-id", RUN, "--project-root", str(root)]

    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "Run complete: fixture" in first.output

    again = runner.invoke(app, args)
    assert again.exit_code == 0, again.output
    assert "already complete" in again.output

    missing = runner.invoke(
        app, ["finish", "--run-id", "absent", "--project-root", str(root)]
    )
    assert missing.exit_code == 1
    assert "manifest is missing" in missing.output
