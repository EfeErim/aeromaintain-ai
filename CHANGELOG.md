# Changelog

## Unreleased

Protocol v4 tests whether uncapped targets fix long-RUL underprediction. They do not pass
the frozen rules, so the capped v3 model stays. The demo remains on its v3 bundle.

### Results

- C-MAPSS development engines, not a clean test: every uncapped candidate improved RMSE
  above 150 cycles on FD002 and FD004, and every one failed the critical-miss gate on both.
- Intervals on v4 evaluation engines: Mondrian coverage 91.7% and 92.4%, split conformal
  88.1% and 88.6%. Split conformal covered only 50.6% and 63.6% above 150 cycles.
- N-CMAPSS DS03-DS07, 20 evaluation engines, simulation only: Ridge RMSE 15.51 against a
  23.56 baseline and coverage 92.8%. Both gates passed; the critical miss rate is 48.3%.
- DS08 was dropped before any label was read: the archive did not match the 54-unit
  assumption, and DS08d was truncated.

### Added

- `aeromaintain.models.v4`, `aeromaintain.evaluation.v4` and `aeromaintain.data.ncmapss`,
  with an optional `ncmapss` dependency group.
- `scripts/fetch_ncmapss.py` and `scripts/publish_v4_evidence.py`.

### Changed

- Holdout claims now use an interprocess lock and atomic ledger writes.

## 0.3.0

Protocol v3 tests the approach under six operating conditions, on the two subsets whose
labels were still clean. The clean results are weak, and are published as they came.

### Results

- `fd002-v3-clean`, clean single-open test: RMSE 27.36 (95% CI 23.77–30.82), MAE 18.46,
  coverage 83% against a nominal 90%, risk rule 60/61 recall with 22 false alarms.
- `fd004-v3-clean`, clean single-open test: RMSE 27.40 (95% CI 24.53–30.23), MAE 19.65,
  coverage 83% against a nominal 90%, risk rule 51/53 recall with 16 false alarms.
- A diagnosis made after opening traces the gap to the RUL cap: predictions stop near 127
  cycles, while true RUL reaches 195. The development window of 150 hid this.
- Condition normalization met its target on training engines: RMSE 10.6% and 10.1% below
  raw features.
- `fd001-v3` and `fd003-v3`, test previously seen: RMSE 15.80 and 15.45. Across ten seeds,
  RMSE averages 16.03 and 15.41. Development RMSE and interval-width targets stay missed
  on both.

### Added

- Operating-condition normalization, fitted inside each training partition.
- `window_focus` training emphasis, five-seed ensembles, and a CQR trial with its own
  tree counts. Only the ensemble passed its rule, on FD003 alone.
- FD002 and FD004 support, with stage rules that also check NASA score.
- Published locks and development evidence before either clean test opened.

### Changed

- The hosted demo and screenshots now show `fd004-v3-clean`.
- New model locks use schema version 3. Version 2 locks still load.

### Unchanged

- The v2 runs stay published as v2, and `fd003-v2-clean` keeps its ledger entry.

## 0.2.0

Protocol v2 replaces all-rows model selection with test-like scoring, and adds a clean
FD003 check.

### Results

- `fd003-v2-clean`, clean single-open test: RMSE 15.07 (95% CI 12.61–17.50), MAE 11.23,
  coverage 92%, risk rule 20/20 recall with 8 false alarms.
- `fd001-v2`, test previously seen: RMSE 15.80, MAE 11.15, coverage 94%, risk rule
  24/25 recall with 4 false alarms. Across ten seeds, RMSE averages 16.03.
- Missed targets: development RMSE of about 16.6 (FD001 18.57, FD003 17.62) and mean
  interval width below 45 (55.4 and 53.3 on test).

### Added

- Development scoring on one random row per engine, over 200 seeded draws.
- Early stopping on an inner engine split.
- Staged search over models, feature sets, and RUL caps.
- Mondrian interval cross-fitted on all training engines, with results by RUL band.
- Risk threshold selection, with miss and false-alarm costs side by side.
- Wilson and bootstrap intervals on official metrics.
- FD003 support and a ledger that allows one model lock to open a clean holdout.
- `robustness`, `compare-runs`, `escalation-study`, and `export-demo` commands.
- Row-free demo bundle and Streamlit entrypoint.

### Changed

- The model lock schema is now version 2. Runs from 0.1.0 need commit `143f272`.
- Malformed manifests and locks now produce a clear evaluation error.

### Unchanged

- `m4-fd001-seed42-20260729` stays published as the v1 reference.
