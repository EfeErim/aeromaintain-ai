from __future__ import annotations

import json
import multiprocessing

import pytest

from aeromaintain.evaluation.ledger import claim_holdout


def _claim_worker(path, name, ready, start, results):
    ready.put(name)
    start.wait(15)
    try:
        claim_holdout(path, "new_holdout", {"model": name})
        results.put("won")
    except ValueError:
        results.put("refused")


def test_two_processes_cannot_claim_same_holdout(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready, results, start = context.Queue(), context.Queue(), context.Event()
    path = tmp_path / "ledger.json"
    workers = [
        context.Process(target=_claim_worker, args=(path, name, ready, start, results))
        for name in ("a", "b")
    ]
    for worker in workers:
        worker.start()
    try:
        ready.get(timeout=30)
        ready.get(timeout=30)
        start.set()
        assert sorted([results.get(timeout=30), results.get(timeout=30)]) == [
            "refused",
            "won",
        ]
        for worker in workers:
            worker.join(30)
            assert worker.exitcode == 0
        assert json.loads(path.read_text())["new_holdout"]["model"] in {"a", "b"}
        assert not path.with_suffix(".json.lock").exists()
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()


def test_ledger_resume_corruption_and_atomic_failure(tmp_path, monkeypatch):
    path = tmp_path / "ledger.json"
    claim_holdout(path, "one", {"model": "a"})
    claim_holdout(path, "one", {"model": "a"})
    with pytest.raises(ValueError, match="already opened"):
        claim_holdout(path, "one", {"model": "b"})
    original = path.read_bytes()

    def fail(*args):
        raise OSError("replacement failed")

    monkeypatch.setattr("aeromaintain.evaluation.ledger.os.replace", fail)
    with pytest.raises(OSError, match="replacement"):
        claim_holdout(path, "two", {"model": "b"})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    path.write_text("[]")
    with pytest.raises(ValueError, match="object"):
        claim_holdout(path, "two", {})
