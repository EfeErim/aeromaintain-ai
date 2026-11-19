# Architecture

## Frozen and maintained code

Modules that ran published experiments are pinned by SHA-256 in
`docs/ncmapss_locks/final_v5.json` and never edited; `tests/test_protocol_v5.py` fails
if one changes, and also checks the lock against the `v1.0.0` tag. Defects in pinned code
are corrected outside it, as with `evaluation/bootstrap.py` and the v5 bootstrap erratum.
A few small serialization and inference helpers are copied into maintained modules, and
`tests/test_model_boundary.py` keeps the copies equivalent.

| Frozen (lock-pinned) | Role |
|---|---|
| `models/rul.py` | C-MAPSS selection, training, calibration and model locks (v2, v3) |
| `models/v4.py`, `models/v5.py` | N-CMAPSS protocol runs |
| `evaluation/protocol.py`, `evaluation/v4.py` | Truncated scoring, intervals, alarm replay |
| `evaluation/ledger.py` | Single-open holdout ledger |
| `data/ncmapss.py` | N-CMAPSS HDF5 reader |

Everything else is maintained and can change as long as existing runs keep loading.
`.joblib` files store `models.rul.RidgeBundle`, `features.causal.FoldPreprocessor` and
`ConditionNormalizer` by module path, so those classes do not move, and the frozen
modules import `sha256_file` from `data.pipeline`, which keeps re-exporting it from
`files.py`.

Three things outside the lock are fixed in practice:

- `configs/project.yaml`: its SHA-256 is recorded in the v2 model locks and in every
  prepared C-MAPSS data manifest, which the v2 and v3 locks hash. Editing it, even to
  drop the keys no code reads, stops those runs from verifying.
  `tests/test_protocol_v3.py` pins its hash.
- The bytes written by `prepare`, `evaluate` and the reports: locks and manifests hold
  their hashes, so a format change breaks existing runs.
- `files.py` holds hashing and write-once helpers with no modeling imports, so the
  review app and the hosted demo do not load scikit-learn or SciPy.

## What "opened once" does and does not prove

The tooling refuses an accidental second opening. It cannot prove that a test set was
opened only once; that rests on the author's word, backed by the public record.

What the code enforces:

- labels are read only after the lock and every file it names match their hashes;
- v5 evaluation runs only from a clean working tree whose `HEAD` contains the lock;
- the ledger refuses a second, different claim on the same holdout.

What it does not enforce:

- **The lock was public before opening.** The code checks local `HEAD`, not that the
  commit was pushed.
- **The ledger survives.** `runs/` is not committed, so deleting the ledger file
  re-arms a holdout, and nothing stops a notebook from reading the label files directly.
- **Commit hashes are stable.** History was rewritten twice after the v5 test (see
  [results v5](results_v5.md)), so commit hashes recorded in locks no longer exist.
  Locks rely on file hashes, which did not change, and the `v1.0.0` tag holds the exact
  files the v5 lock names. One file, `protocol_v5_development.json`, is stored at the tag
  with LF line endings while the lock hashed its CRLF bytes; the test checks that only
  line endings differ.

For any future holdout, anchor the lock outside this repository before opening (a signed
tag recorded with a third party) and never rewrite history after that.

## Run verification

`artifacts.verify_lock` is the one check of a run's hash chain: the run manifest pins
the model lock, and the lock pins the prepared data, `configs/project.yaml` and every
model file. Official evaluation, seed sweeps, challenger studies and the review app all
start from it. `load_verified_run` adds the evaluation results, and `load_public_bundle`
checks the demo bundle. Each JSON file is checked against its schema, the artefact maps
must list every file the run needs, every listed path must stay inside its directory
with its listed SHA-256, and the public bundle may hold only the review columns and no
`rul_true` or `sensor_history` key.

This shows that files are unchanged since they were listed. It does not make a `.joblib`
file safe to load: a run directory from someone else can list a malicious pickle with a
correct hash. Load only runs this project produced.

## Run states and resuming

| State | Meaning |
|---|---|
| `model_locked` | Lock written; official labels not opened |
| `model_locked` + `official_test/evaluation_manifest.json` | Evaluation recorded |
| `pipeline_complete` | Reports written and listed; the manifest is the commit point |

`aeromaintain finish --run-id ID` moves a run forward from wherever it stopped:

- it evaluates only when `official_test/` is empty, so labels are not read twice;
- if result files exist without an evaluation manifest, it stops for a person to look;
- a report left by an interrupted call is accepted only if its bytes match;
- the manifest is replaced last and atomically, so a run never reads as complete early;
- an OS file lock under `runs/` refuses a second concurrent `finish` or `pipeline`.

Each file is replaced atomically, but the set of files is not a transaction; the next
`finish` verifies what it finds instead of trusting it.
