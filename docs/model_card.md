# Model and data card

An educational benchmark prototype for RUL prediction on simulated NASA engines. It must
not be used for dispatch, safety, certification, airworthiness or real maintenance
approval. Test results are in [results](results.md) (C-MAPSS) and
[results v5](results_v5.md) (N-CMAPSS).

## Models

C-MAPSS protocols v2 and v3 lock one XGBoost specification per subset, trained with
engine-equal sample weights, plus a Mondrian empirical interval and a risk threshold.

| Run | Dataset | Test status | Features | Cap | Trees | Risk threshold |
|---|---|---|---:|---:|---:|---:|
| `fd002-v3-clean` | FD002 | clean | `compact`, condition-normalized | 125 | 166 | `interval_low <= 25` |
| `fd004-v3-clean` | FD004 | clean | `compact`, condition-normalized | 125 | 321 | `interval_low <= 20` |
| `fd003-v2-clean` | FD003 | clean | 64 (`compact`) | 130 | 347 | `interval_low <= 25` |
| `fd001-v2`, `fd001-v3` | FD001 | previously seen | 403 (`all`) | 125 | 123 | `interval_low <= 20` |
| `fd003-v3` | FD003 | previously seen | 403 (`all`) | 125 | 184, five seeds | `interval_low <= 25` |

Features use only current and earlier rows: operating settings, current sensors, engine
age, and 5/10/20-cycle sensor statistics; `compact` keeps current sensors, engine age and
20-cycle means and slopes. On FD002 and FD004, six operating-condition clusters are fitted
inside each training partition and sensors are standardized within each condition. The
hosted demo shows `fd004-v3-clean`.

The final N-CMAPSS model (protocol v5) is separate:

| Item | Value |
|---|---|
| Data | N-CMAPSS DS02–DS07, 64 simulated engines, all cycles |
| Features | 72 cycle summaries: mean, std, min and max of 14 `X_s` sensors and 4 `W` operating channels |
| Excluded inputs | Cycle index, unit ID, flight class, `X_v`, `T`, health parameters |
| Target | Uncapped RUL, engine-equal weights |
| Selection | XGBoost (869 trees) over Ridge, by out-of-fold RMSE 12.87 against 15.01 |
| Lock | [`final_v5.json`](ncmapss_locks/final_v5.json) |

## Data

**C-MAPSS** (NASA Prognostics Center of Excellence;
[dataset page](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data),
[generation reference](https://ntrs.nasa.gov/citations/20090029214)): simulated
run-to-failure trajectories, not fleet measurements. FD001 and FD003 run under one
operating condition, FD002 and FD004 under six; FD003 and FD004 add fan degradation to
compressor degradation. The archive's size and SHA-256 are in `configs/project.yaml` and
are checked before extraction. Shapes come from the archive and are enforced by
`DATASET_SHAPES` in `data/pipeline.py`:

| Subset | Train engines | Train rows | Test engines | Test rows | Largest test RUL |
|---|---:|---:|---:|---:|---:|
| FD001 | 100 | 20,631 | 100 | 13,096 | 145 |
| FD002 | 260 | 53,759 | 259 | 33,991 | 194 |
| FD003 | 100 | 24,720 | 100 | 16,596 | 145 |
| FD004 | 249 | 61,249 | 248 | 41,214 | 195 |

NASA's page lists FD004 as 248 train and 249 test trajectories, the reverse of the
archive; the pipeline follows the archive.

**N-CMAPSS** (Arias Chao et al., Data 6(1), 2021): run-to-failure trajectories driven by
real recorded flight conditions, sampled at 1 Hz, still simulation. Only measured sensors
`X_s` and operating conditions `W` are inputs; `Y` is the target. Engine IDs are
namespaced by file, and per-file SHA-256 hashes are in the published locks.

| Files | Engines | Role |
|---|---:|---|
| DS02-006 | 9 | v4 descriptive pilot; v5 training |
| DS03-012, DS04, DS05, DS06, DS07 | 55 | v4 benchmark; v5 training |
| DS01-005, DS08a-009, DS08c-008 | 35 | v5 final clean test, opened once |
| DS08d-010 | – | Excluded; the archive member is truncated |

The 35 final test engines were reserved in
[`final_holdout_reservation.json`](ncmapss_locks/final_holdout_reservation.json) before
protocol v5 existed; until the lock was committed only their unit IDs were read.

## Splits and targets

All splits and folds are by whole engine. A seed-42 split by lifetime quartile assigns
80% of training engines to development and 20% to calibration: 80/20 on FD001 and FD003,
208/52 on FD002, 199/50 on FD004. Selection uses five engine-grouped folds of the
development engines; the final model and the cross-fit interval use all training engines.
Official test labels are stored apart from the test features and read only after
`model_lock.json` validates.

Training targets are `min(engine_max_cycle - cycle, 125)`. The cap is a modeling
assumption, not a physical truth; official metrics use NASA's uncapped labels, so the cap
costs accuracy on engines with far more life left.

## Limitations

- Results do not establish performance on real engines or under real distribution shift.
- The cap limits predictions to about 127 cycles; above 150 true cycles on FD002 and
  FD004, RMSE is 54–61 and coverage 25–31% (computed after opening).
- Test interval coverage on FD002 and FD004 fell short of the nominal 90%; coverage has
  no finite-sample guarantee and the interval is not a safety guarantee.
- The 150-cycle development window was chosen after FD001 labels were seen; with no
  window the champion rule picks Ridge on all four subsets.
- SHAP values and Ridge coefficients describe the model, not physical causes.
- Per-file failure modes for DS01 and DS08 were not verified from a primary source, so
  the v5 test's shift is defined only by file identity.

## Redistribution

NASA's dataset page displayed `License not specified` when rechecked on 2026-11-04, so no
source rows from either dataset are committed, attached to releases or used as test
fixtures. The `demo/` bundle holds aggregate metrics, explanations and per-engine model
outputs, plus each test engine's observed cycle count; no sensor rows, settings or true
RUL. The MIT license covers project code and documentation only.
