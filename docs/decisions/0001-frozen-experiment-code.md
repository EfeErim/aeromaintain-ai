# 0001: Frozen experiment code next to maintained code

Date: 2026-11-13

## Context

The published results rest on model locks that hash the exact source files they
ran. The v5 lock pins seven files, including `models/rul.py` (1,600 lines), which
holds C-MAPSS selection, training, calibration, explanation, persistence and the
`train_and_lock` orchestration in one module. Editing any pinned file breaks
`tests/test_protocol_v5.py` and, more importantly, the claim that the published code
is the code that produced the numbers.

Other code still changes: run loading, reports, the CLI and the review app. Before
this decision, `models/evaluation.py` imported private helpers from `rul.py`, and
`delivery.py` imported its run type from the Streamlit package.

## Decision

1. Pinned files stay byte-identical. A defect found in one is corrected outside it,
   as with the v5 bootstrap (`evaluation/bootstrap.py`, erratum in
   `docs/results_v5.md`), and a test pins the old behaviour so the erratum stays
   accurate.
2. `rul.py` is not split or copied. Its training path is called by `aeromaintain
   train` and by the seed and challenger studies, which exist to repeat that exact
   code. A rewrite would need its own evidence of equal behaviour across every
   selection rule, and nothing currently needs one.
3. What active code needed from `rul.py` moves to maintained modules, grouped by why
   it changes:
   - `models/run_files.py`: write-once run files (JSON, Parquet, refuse-different);
   - `models/inference.py`: loading a locked model, building its features, predicting,
     applying its interval;
   - `artifacts.py`: JSON contracts and loading for runs and the public bundle, with
     no UI dependency.
4. Where a maintained module repeats a frozen helper (`json_bytes`, `parquet_bytes`,
   `write_new_bytes`, `processed_dir`, `predict_model`, `predict_quantiles`), a test
   in `tests/test_model_boundary.py` checks that both produce identical output.
5. Classes stored in `.joblib` files (`RidgeBundle`, `FoldPreprocessor`,
   `ConditionNormalizer`) keep their module paths.

## Consequences

- Official evaluation, reports and the app can change without touching a lock.
- Six small helpers exist twice. Each pair is short and covered by an equivalence
  test.
- Changing how a C-MAPSS model is trained means a new protocol version with new
  code, not an edit to `rul.py`.
- Before and after this change, the eight stored C-MAPSS locks gave bit-identical
  predictions and intervals on their test features (no labels read).

## Amendment, 2026-11-16: provenance from the release tag

An outside review pointed out that pinning files in the working tree is both the
proof and the cost: seven files can never change on `main`, and the proof itself
depends on `main` staying untouched.

- `tests/test_protocol_v5.py` now also checks the v5 lock against the files stored at
  the `v1.0.0` tag. That tag is the provenance record; CI fetches tags so the check
  never skips there.
- The working-tree check stays for now, because the seed and challenger studies
  reuse `rul.py` and are meant to repeat the published code exactly.
- If a pinned file ever needs to change, the change can be made on `main` with the
  working-tree check retired for that file, as long as the tag check keeps passing
  and the studies are rerun against the tag. `evaluation/ledger.py` and
  `evaluation/protocol.py` are the likeliest candidates. Copying them into a
  maintained module now would only add more duplicated code with no current need, so
  that is deferred until a change is actually needed.
