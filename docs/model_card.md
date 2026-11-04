# AeroMaintain AI RUL Model Card

## Model summary

AeroMaintain AI predicts remaining useful life (RUL) for simulated NASA C-MAPSS
engines. Protocol v2 locks one XGBoost regressor per dataset, trained with
engine-equal sample weights, plus a Mondrian empirical interval and a risk threshold.

| Run | Dataset | Features | Cap | Trees | Risk threshold |
|---|---|---:|---:|---:|---:|
| `fd003-v2-clean` | FD003 | 64 (`compact`) | 130 | 347 | `interval_low <= 25` |
| `fd001-v2` | FD001 | 403 (`all`) | 125 | 123 | `interval_low <= 20` |

This is an educational benchmark prototype, not an airworthiness,
maintenance-approval, production-readiness, or real-fleet system.

## Intended use

- compare a target mean, Ridge, and a bounded XGBoost search under one test-like
  protocol;
- produce a nominal empirical prediction interval and risk band;
- explain model behavior with SHAP or Ridge coefficients; and
- review one locked official-test evaluation.

It must not be used for dispatch, safety, certification, or real maintenance
approval.

## Data and split

C-MAPSS is simulated run-to-failure turbofan data. A seed-42 split assigns 80 training
engines to development and 20 to calibration. Selection uses five engine-grouped folds
of the development engines only. The final model and the cross-fit interval then use
all 100 training engines. See [`data_card.md`](data_card.md).

## Features and selection

Features use only the current and earlier rows: operating settings, current sensors,
engine age, and 5/10/20-cycle sensor statistics. The `compact` set keeps current
sensors, engine age, and 20-cycle means and slopes. Fold-local preprocessing and an
inner engine split for early stopping prevent leakage into scored engines.

[Protocol v2](protocol_v2.md) has the complete rule. XGBoost replaced Ridge on both
datasets, by lowering median development RMSE at least 5% without worsening NASA score.

| Development model | Dataset | MAE | RMSE | NASA | Point-rule recall |
|---|---|---:|---:|---:|---:|
| Ridge | FD001 | 16.75 | 21.01 | 626 | 0.70 |
| **XGBoost** | FD001 | 14.40 | **18.57** | 501 | 0.85 |
| Ridge | FD003 | 17.11 | 21.58 | 702 | 0.70 |
| **XGBoost** | FD003 | 12.60 | **17.62** | 515 | 0.89 |

Both miss the development RMSE target of about 16.6.

## Locked official-test evaluation

FD003 is a clean single-open test of the frozen protocol. FD001 labels were seen before
v2, so its column is not independent evidence. Ranges are 95% bootstrap intervals.

| Metric | FD003 clean | FD001 seen |
|---|---:|---:|
| RMSE | 15.07 (12.61–17.50) | 15.80 (12.87–18.78) |
| MAE | 11.23 (9.34–13.28) | 11.15 (9.10–13.43) |
| Motor-normalized NASA score | 399 (240–620) | 398 (233–625) |
| Interval coverage, nominal 90% | 0.92 (0.86–0.97) | 0.94 (0.89–0.98) |
| Mean interval width | 53.3 (49.3–57.3) | 55.4 (51.2–59.4) |

| Alert rule | Dataset | Recall | Precision | False-alarm rate |
|---|---|---:|---:|---:|
| `prediction <= 30` | FD003 | 0.80 (0.58–0.92) | 0.94 (0.73–0.99) | 0.01 (0.00–0.07) |
| `interval_low <= 25` (interface) | FD003 | 1.00 (0.84–1.00) | 0.71 (0.53–0.85) | 0.10 (0.05–0.19) |
| `prediction <= 30` | FD001 | 0.80 (0.61–0.91) | 1.00 (0.84–1.00) | 0.00 (0.00–0.05) |
| `interval_low <= 20` (interface) | FD001 | 0.96 (0.80–0.99) | 0.86 (0.69–0.94) | 0.05 (0.02–0.13) |

Alert ranges are two-sided Wilson 95% intervals. Across ten seeds of the FD001
settings, RMSE averages 16.03 (SD 0.14); seed 42 is the lowest of the ten.

## Explanations and limitations

SHAP values describe model behavior, not physical causality or component diagnosis.

- Each dataset has one simulated operating condition, and results do not establish
  performance under real distribution shift.
- Critical-engine rates rest on 20–25 engines.
- The development RMSE target and interval width target of 45 were both missed.
- The eligibility window of 150 was chosen after FD001 labels were seen. Without any
  window, the champion rule picks Ridge on both datasets.
- Interval coverage comes from cross-fit scores with no formal finite-sample guarantee,
  and the interval is not a safety guarantee.
- Band-specific test coverage can be low: 0.76 on FD001's predicted `31-60` band.
- A predicted-band boundary can make `interval_low` jump, so ranking by it is not
  strictly monotonic in point RUL.

## Reproducibility and integrity

| Run | Model-lock SHA-256 |
|---|---|
| `fd003-v2-clean` | `bcd2acda16f5a60497725543879d033eca0d83932a7f62525581e44172bfba33` |
| `fd001-v2` | `9c5af0ded502c396df454f2fbc468d839b5bdd680da8f0abc9a416859c05e5da` |
| `m4-fd001-seed42-20260729` (v1, Ridge) | `32e5e7a421ad8ccb81fff4e8b8c17957f49fa855954b27e4e4b16219634af829` |

Each v2 lock records data, split, config, protocol document, feature order, model,
calibration, and artefact hashes. The evaluator refuses a missing or changed lock, and
a ledger refuses a second lock on a clean holdout. The app checks the lock, evaluation,
and report chain before rendering. See [`results.md`](results.md) for v1 and for what
repeats byte for byte.
