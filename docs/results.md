# Reference Run Results

Model and data-derived files are Git-ignored. Sanitized aggregate evidence is stored
in [`reference_evidence.json`](reference_evidence.json).

## Protocol v2 runs

Both runs were produced on 2026-11-04 under the frozen [protocol v2](protocol_v2.md).

| Run | Dataset | Label | What it can support |
|---|---|---|---|
| `fd003-v2-clean` | FD003 | clean single-open test | Independent check of the frozen protocol |
| `fd001-v2` | FD001 | test previously seen | Comparison with v1 only |

The FD003 ledger records one opening, by model lock
`bcd2acda16f5a60497725543879d033eca0d83932a7f62525581e44172bfba33`.

### Official test

Ranges are 95% engine-level bootstrap intervals over 2,000 resamples.

| Metric | FD003 clean | FD001 seen |
|---|---:|---:|
| Engines | 100 | 100 |
| RMSE | 15.066 (12.607–17.503) | 15.796 (12.867–18.781) |
| MAE | 11.227 (9.339–13.276) | 11.154 (9.100–13.431) |
| Motor-normalized NASA score | 398.56 (239.62–619.97) | 398.31 (232.98–624.67) |
| Signed bias | 0.384 (-2.547–3.284) | -1.593 (-4.608–1.280) |
| Interval coverage, nominal 90% | 0.92 (0.86–0.97) | 0.94 (0.89–0.98) |
| Mean interval width | 53.285 (49.279–57.278) | 55.422 (51.190–59.443) |

### Critical-engine alert rules

A truly critical engine has `rul_true <= 30`: 20 engines on FD003 and 25 on FD001.
Ranges are two-sided Wilson 95% intervals.

| Dataset | Rule | Recall | Precision | False-alarm rate | Missed | False alarms |
|---|---|---:|---:|---:|---:|---:|
| FD003 | `prediction <= 30` | 80.0% (58.4–91.9%) | 94.1% (73.0–99.0%) | 1.25% (0.2–6.7%) | 4 | 1 |
| FD003 | `interval_low <= 25` | 100% (83.9–100%) | 71.4% (52.9–84.7%) | 10.0% (5.2–18.5%) | 0 | 8 |
| FD001 | `prediction <= 30` | 80.0% (60.9–91.1%) | 100% (83.9–100%) | 0.0% (0.0–4.9%) | 5 | 0 |
| FD001 | `interval_low <= 20` | 96.0% (80.5–99.3%) | 85.7% (68.5–94.3%) | 5.3% (2.1–12.9%) | 1 | 4 |

### Interval by RUL band

Each cell gives test coverage and mean width. Small bands make single-band coverage
noisy; FD001's `31-60` predicted band holds 17 engines.

| Band | FD003 | FD001 |
|---|---:|---:|
| Predicted `<=30` | 0.88, 16.3 | 1.00, 19.5 |
| Predicted `31-60` | 0.95, 34.9 | 0.76, 40.1 |
| Predicted `61-100` | 1.00, 69.3 | 0.97, 74.8 |
| Predicted `>100` | 0.86, 68.0 | 0.97, 67.5 |
| True `0-30` | 0.90, 20.0 | 0.96, 23.6 |
| True `31-60` | 1.00, 43.8 | 1.00, 52.5 |
| True `61-125` | 0.89, 67.9 | 0.92, 68.9 |
| True `>125` | 0.93, 68.2 | 0.91, 70.1 |

## Development selection

Metrics are medians over 200 seeded draws of one row per engine, on 80 development
engines with `rul_true <= 150`.

| Model | Dataset | RMSE | MAE | NASA | Point-rule recall |
|---|---|---:|---:|---:|---:|
| Ridge, alpha 1 | FD001 | 21.01 | 16.75 | 626 | 0.70 |
| **XGBoost, all features, cap 125** | FD001 | **18.57** | 14.40 | 501 | 0.85 |
| Ridge, alpha 100, no age | FD003 | 21.58 | 17.11 | 702 | 0.70 |
| **XGBoost, compact features, cap 130** | FD003 | **17.62** | 12.60 | 515 | 0.89 |

XGBoost lowered RMSE 11.6% on FD001 and 18.4% on FD003, and won 99% of paired draws
on each. Neither run reached the RMSE target of about 16.6.

### Window sensitivity

The window of 150 was an analysis choice. Each cell gives median RMSE for Ridge and
XGBoost, then the rule's champion.

| Window | FD001 | FD003 |
|---|---|---|
| 130 | 18.35 / 16.35, XGBoost | 18.33 / 15.59, XGBoost |
| **150 (used)** | 21.01 / 18.57, XGBoost | 21.58 / 17.62, XGBoost |
| 175 | 26.32 / 24.15, XGBoost | 27.36 / 22.25, XGBoost |
| None | 44.86 / 43.78, **Ridge** | 72.34 / 69.27, **Ridge** |

With no window, XGBoost still has the lower RMSE but a worse NASA score, so Ridge
wins. By rule, this disagreement is published and does not change the decision.

### Challengers on FD001

[Amendment 1](protocol_v2.md#amendment-1) wrote these rules after the first FD001
result and before any challenger result. Neither challenger earned a place, so FD003
did not use them.

| Challenger | Result | Rule outcome |
|---|---|---|
| LSTM, 64 units, one layer | Median RMSE 18.39 (0.95% lower), NASA 546 against 501 | XGBoost stays |
| CQR | Coverage 0.91, mean width 98.6, 77% wider than Mondrian | Mondrian stays |

The CQR trial reused a tree count tuned for squared error, so its quantile models may
underfit. It shows only that this bounded trial lost.

## Risk threshold

Leave-engines-out medians on all 100 training engines. The selected threshold is
the smallest with at least 95% recall. Missing a critical engine costs far more than
a false alarm in this framing, but no real cost data exists, so both counts are shown
rather than combined.

| `interval_low <= T` | FD001 recall | FD001 missed / false alarms | FD003 recall | FD003 missed / false alarms |
|---:|---:|---:|---:|---:|
| 10 | 0.65 | 7 / 0 | 0.58 | 8.5 / 0 |
| 15 | 0.83 | 4 / 2 | 0.77 | 5 / 1 |
| **20 (FD001)** | **0.96** | **1 / 5** | 0.94 | 1 / 4 |
| **25 (FD003)** | 1.00 | 0 / 7 | **1.00** | **0 / 7** |
| 30 | 1.00 | 0 / 12 | 1.00 | 0 / 10 |
| 40 | 1.00 | 0 / 23 | 1.00 | 0 / 22 |

## Interval calibration

The Mondrian interval uses all 100 training engines in place of v1's 20. Leave-
engines-out coverage is 0.91 on both datasets, with mean width 55.7 on FD001 and 52.0
on FD003. That beats v1's 60.6 but misses the target of 45.

| Predicted band | FD001 coverage, width | FD003 coverage, width |
|---|---:|---:|
| `<=30` | 0.92, 18.8 | 0.92, 15.7 |
| `31-60` | 0.91, 40.6 | 0.90, 35.0 |
| `61-100` | 0.91, 75.0 | 0.89, 69.3 |
| `>100` | 0.92, 67.5 | 0.93, 68.6 |

## Seed sensitivity

Ten seeds (42–51) refit the locked FD001 settings. They vary cross-fit folds, model
randomness, draws, and the final fit. Selection is not rerun, so this spread excludes
selection variance. FD003 is excluded because its labels open only once.

| Metric | Mean | SD | Range |
|---|---:|---:|---:|
| RMSE | 16.034 | 0.135 | 15.796–16.205 |
| MAE | 11.623 | 0.205 | 11.154–11.902 |
| NASA score | 419.4 | 13.3 | 398.3–442.4 |
| Interval coverage | 0.942 | 0.009 | 0.93–0.95 |
| Mean interval width | 56.31 | 0.65 | 55.42–57.43 |
| Risk-rule recall | 0.960 | 0.027 | 0.92–1.00 |
| Risk-rule false alarms | 3.2 | 0.8 | 2–4 |

The published seed 42 has the lowest RMSE, MAE, and NASA score of the ten, so its
FD001 point values are slightly optimistic. The risk threshold was 20 for every seed.

## Reproducibility

`fd001-v2` and `fd001-v2-repeat` ran with identical inputs on one machine.

- **Byte-identical:** model, preprocessor, feature manifest, comparison, decision,
  explanation, calibration analysis, out-of-fold and cross-fit predictions, official
  predictions, official metrics, and error analysis.
- **Different:** `manifest.json`, `model_lock.json`, `evaluation_manifest.json`, and
  both reports. Each carries the run ID or a hash of a file that does; the lock has no
  other changed field.

This does not establish byte-identical output across machines or library versions.

**v1 lock hash, resolved.** The clean-clone `fd001-repro` lock hash differed from the
v1 reference because that lock adds the `environment` field from commit `5f69de1`.
Removing that field and restoring the reference `run_id` reproduces the reference
hash `32e5e7a4…f829` exactly, so every hash the lock records matched.

## Protocol v1 reference

The immutable run `m4-fd001-seed42-20260729` remains published as v1, reproducible
from commit `143f272`. Ridge was its champion under v1's all-rows selection.

| Metric | Result |
|---|---:|
| Engines | 100 |
| MAE | 15.369728 |
| RMSE | 19.622062 |
| Motor-normalized NASA score | 625.326953 |
| Signed bias | -1.777402 |
| Overprediction rate | 0.480000 |
| Nominal / observed coverage | 0.90 / 0.89 |
| Mean interval width | 60.623140 |

| Rule (25 critical engines) | Recall | Precision | False-alarm rate | Missed | False alarms |
|---|---:|---:|---:|---:|---:|
| Point, `prediction <= 30` | 48.0% (30.0–66.5%) | 100% (75.8–100%) | 0.0% (0.0–4.9%) | 13 | 0 |
| Risk, `interval_low <= 30` | 100% (86.7–100%) | 73.5% (56.9–85.4%) | 12.0% (6.4–21.3%) | 0 | 9 |

## Screenshots

- [Overview](screenshots/overview.png)
- [Engine risk](screenshots/engine-health.png)

Both show `fd003-v2-clean`, loaded through the verified artefact loader.
