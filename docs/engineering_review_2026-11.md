# Engineering review, November 2026

Scope: artefact verification, resuming `finish`, integration tests for v5 and the
CLI, and the line between frozen and maintained code. No model was retrained, no
published result was regenerated, and no holdout label was opened.

## Starting point

Measured on 2026-11-13 before any change: 96 tests passed, 1 skipped (no PyTorch);
total coverage 81%, `models/v5.py` 32%, `cli.py` 46%, `delivery.py` 79%; Ruff clean.

Two defects were reproduced:

1. A demo bundle whose manifest listed `"artifacts": {}` loaded, because the loader
   only checked the files that happened to be listed.
2. If `finish` stopped after writing `report.json` and `report.html` but before
   replacing the run manifest, every later `finish` failed with "Refusing to
   overwrite existing output", so the run could not be completed. Each retry also
   ran the official evaluation again, reading the test labels a second time.

The same gap as (1) existed for verified runs: an empty or partial model-lock map or
pipeline record was accepted, and `explanation.json` was read without being listed.

## Changes

| Area | Change |
|---|---|
| `artifacts.py` (new) | Pydantic contracts for the public manifest and run payload, run manifest, pipeline record, model lock (schema 2 and 3) and evaluation manifest; required-file lists derived from what each run actually opens; path containment; forbidden-key and ranking checks; export writes the manifest last and reloads the bundle |
| `app/artifacts.py` | Re-exports only; old imports keep working |
| `delivery.py` | Same-bytes reports accepted, different ones refused; recorded evaluation reused; partial evaluation stops; completed run verified and left untouched; OS file lock per run; manifest stays the last, atomic write |
| `models/inference.py`, `models/run_files.py` (new) | Locked-model loading, prediction, intervals and write-once files, taken out of `models/evaluation.py` and away from private `rul.py` helpers |
| `models/evaluation.py` | Validation and official evaluation only; `validate_lock` public, `_validate_lock` kept as alias |
| `cli.py`, `models/__init__.py` | Use `artifacts` and the `models` package API instead of `app` and `rul` internals; `finish` reports an already-complete run |

The split between frozen and maintained code is recorded in
[decision 0001](decisions/0001-frozen-experiment-code.md).

## Evidence

| Check | Result |
|---|---|
| `pip check`, `ruff check .`, `ruff format --check .` | Clean |
| `pytest --cov=src/aeromaintain --cov-fail-under=80` | 164 passed, 1 skipped (PyTorch); total 88.3% |
| Coverage of critical modules | `models/v5.py` 32% → 86%, `cli.py` 46% → 74%, `delivery.py` 79% → 90%, `models/evaluation.py` 89% → 94%, `artifacts.py` 93% |
| Lock tests (`tests/test_protocol_v5.py -k "lock or gate"`) | 9 passed; every frozen file still matches the v5 lock |
| Stored C-MAPSS runs (local only) | All 7 completed runs and the tracked demo bundle load under the new contracts |
| Report determinism (local only) | Reports regenerated from the 7 completed runs equal the stored `report.json` and `report.html` byte for byte |
| Refactor before/after (local only) | Predictions and intervals of all 8 stored C-MAPSS locks on their test features are bit-identical (no labels read) |
| Writes outside temp dirs | `data/`, `runs/` and `demo/` file lists, sizes and timestamps unchanged after the full suite |

What the new tests now exercise:

- `tests/test_artifacts.py`: empty and non-object maps, each missing required entry,
  wrong hash formats, unknown fields and entries, changed and deleted files, paths
  outside the run, forbidden columns and keys, unknown risk bands and missing values,
  lock schemas 2, 3 and 4, required lock and pipeline entries.
- `tests/test_finish.py`: interruption before the first report, after `report.json`,
  after `report.html`, and while the manifest is replaced, each followed by a
  successful retry that matches an uninterrupted run byte for byte; conflicting
  report; repeated `finish` on a complete run; corrupt and partial evaluations; one
  evaluation call across retries; a second `finish` in the same process and while
  another process holds the lock; CLI exit codes.
- `tests/test_v5_flow.py`: v5 `prepare` and `evaluate` end to end on synthetic HDF5
  files in a temporary git repository, including lock contents, feature order, engine
  namespacing, output schemas and gate decisions; a changed test file, source file,
  config, protocol or reservation, an uncommitted change or an existing ledger claim
  each stop evaluation before any label is read.
- `tests/test_model_boundary.py`: the repeated helpers match their frozen copies;
  pickled classes keep their module paths; stored `.joblib` files load.
- `tests/test_bootstrap.py`: the corrected bootstrap draws from all 35 engines; the
  frozen one still draws from two.

## Not done, and why

- `models/rul.py` was not split. It is pinned by the v5 lock, and a rewrite of
  training would need its own proof of equal behaviour. Its 300-line
  `train_and_lock` remains the main piece of technical debt.
- `models/escalation.py` stays at 49%: its LSTM path needs PyTorch, which is an
  optional extra and not installed here. The CQR path is covered.
- The v5 module entry point (`python -m aeromaintain.models.v5`) is tested through a
  subprocess, which coverage does not count.
- `cli.py` commands `robustness`, `escalation-study` and `compare-runs` still lack
  CLI-level tests; their underlying functions are tested.
- Runs, data and the stored-model checks above are local: `runs/` and `data/` hold
  NASA-derived files and are not committed, so CI skips those tests.
- Tests ran on Windows 11 with Python 3.11.9. The POSIX branch of the file lock
  (`fcntl`) runs only in CI on Linux.
- The demo bundle's `run.json` is checked for forbidden keys by name only; a label
  stored under another key would not be caught.
