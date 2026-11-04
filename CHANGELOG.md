# Changelog

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
