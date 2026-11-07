# AeroMaintain AI RUL Model Card

## Model summary

AeroMaintain AI predicts remaining useful life (RUL) for simulated NASA C-MAPSS
engines. Protocols v2 and v3 lock one XGBoost specification per dataset, trained with
engine-equal sample weights, plus a Mondrian empirical interval and a risk threshold.

| Run | Dataset | Test status | Features | Cap | Trees | Risk threshold |
|---|---|---|---:|---:|---:|---:|
| `fd002-v3-clean` | FD002 | clean | `compact`, condition-normalized | 125 | 166 | `interval_low <= 25` |
| `fd004-v3-clean` | FD004 | clean | `compact`, condition-normalized | 125 | 321 | `interval_low <= 20` |
| `fd003-v2-clean` | FD003 | clean | 64 (`compact`) | 130 | 347 | `interval_low <= 25` |
| `fd001-v2` | FD001 | previously seen | 403 (`all`) | 125 | 123 | `interval_low <= 20` |
| `fd003-v3` | FD003 | previously seen | 403 (`all`) | 125 | 184, five seeds | `interval_low <= 25` |
| `fd001-v3` | FD001 | previously seen | 403 (`all`) | 125 | 123 | `interval_low <= 20` |

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

C-MAPSS is simulated run-to-failure turbofan data. A seed-42 split assigns 80% of
training engines to development and 20% to calibration, by lifetime quartile: 80/20 on
FD001 and FD003, 208/52 on FD002, and 199/50 on FD004. Selection uses five engine-grouped
folds of the development engines only. The final model and the cross-fit interval then
use all training engines. See [`data_card.md`](data_card.md).

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

On FD002 and FD004, [protocol v3](protocol_v3.md) also fits six operating-condition
clusters inside each training partition and standardizes sensors within each condition.

| Development model | Dataset | MAE | RMSE | NASA | Point-rule recall |
|---|---|---:|---:|---:|---:|
| Ridge | FD002 | 16.72 | 21.41 | 1,924 | 0.73 |
| XGBoost, raw features | FD002 | n/a | 21.19 | 2,110 | n/a |
| **XGBoost, normalized** | FD002 | 14.28 | **18.93** | 1,470 | 0.84 |
| Ridge | FD004 | 17.93 | 22.62 | 2,386 | 0.68 |
| XGBoost, raw features | FD004 | n/a | 21.82 | 2,345 | n/a |
| **XGBoost, normalized** | FD004 | 14.60 | **19.61** | 1,764 | 0.84 |

[Protocol v4](protocol_v4.md) compared these capped models with uncapped alternatives on
new development partitions. Uncapped models reduced long-RUL error but missed more
critical cycles than its frozen limit allowed, so the capped specification and the
weights below are unchanged. See the [v4 results](results_v4.md).

## Locked official-test evaluation: six conditions

FD002 and FD004 are clean single-open tests of protocol v3. Ranges are 95% bootstrap
intervals for metrics and two-sided Wilson 95% intervals for alert rates.

| Metric | FD002 clean | FD004 clean |
|---|---:|---:|
| RMSE | 27.36 (23.77–30.82) | 27.40 (24.53–30.23) |
| MAE | 18.46 (15.96–21.01) | 19.65 (17.30–21.99) |
| Signed bias | -8.98 (-12.12 to -5.71) | -9.34 (-12.62 to -6.31) |
| Motor-normalized NASA score | 10,388 (5,193–16,823) | 5,126 (3,627–6,816) |
| Interval coverage, nominal 90% | 0.83 (0.78–0.87) | 0.83 (0.78–0.87) |
| Mean interval width | 55.8 (52.9–58.7) | 58.5 (55.6–61.3) |

| Alert rule | Dataset | Recall | Precision | False-alarm rate |
|---|---|---:|---:|---:|
| `prediction <= 30` | FD002 | 0.93 (0.84–0.97) | 0.93 (0.84–0.97) | 0.02 (0.01–0.05) |
| `interval_low <= 25` (interface) | FD002 | 0.98 (0.91–1.00) | 0.73 (0.63–0.82) | 0.11 (0.07–0.16) |
| `prediction <= 30` | FD004 | 0.85 (0.73–0.92) | 0.90 (0.79–0.96) | 0.03 (0.01–0.06) |
| `interval_low <= 20` (interface) | FD004 | 0.96 (0.87–0.99) | 0.76 (0.65–0.85) | 0.08 (0.05–0.13) |

Predictions stop near 127 cycles because targets are capped at 125, while true RUL
reaches 195. See [the diagnosis](results.md#diagnosis-made-after-opening).

## Locked official-test evaluation: one condition

FD003 is a clean single-open test of the frozen v2 protocol. FD001 labels were seen
before v2, so its column is not independent evidence. Ranges are 95% bootstrap intervals.

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

- Results do not establish performance under real distribution shift.
- The RUL cap limits predictions to about 127 cycles. Above 150 true cycles on FD002
  and FD004, RMSE is 54–61 and interval coverage 25–31%. These figures were computed
  after opening and are diagnostic only.
- Test interval coverage on FD002 and FD004 fell short of its nominal 90%.
- Critical-engine rates rest on 20–61 engines.
- On FD001 and FD003, the development RMSE target and interval width target of 45 were
  both missed.
- The eligibility window of 150 was chosen after FD001 labels were seen. Without any
  window, the champion rule picks Ridge on all four datasets.
- FD002 and FD004 used 4 XGBoost candidates in place of 12 and no CQR trial, under
  [Amendment 1](protocol_v3.md#amendment-1).
- Interval coverage comes from cross-fit scores with no formal finite-sample guarantee,
  and the interval is not a safety guarantee.
- Band-specific test coverage can be low: 0.76 on FD001's predicted `31-60` band.
- A predicted-band boundary can make `interval_low` jump, so ranking by it is not
  strictly monotonic in point RUL.

## Reproducibility and integrity

| Run | Model-lock SHA-256 |
|---|---|
| `fd002-v3-clean` | `fa50040abf9731cf3fd4a3b47aeb82222792dfc7304dc137b10166615ba4cb37` |
| `fd004-v3-clean` | `f296bab85503ef145919402059fc63fc735764f2019b9e24893396bcabca41f7` |
| `fd001-v3` | `256ea39507936cf250f1acbcde578ad78a8ed68401bdcf345d438f90441cb135` |
| `fd003-v3` | `60d768b494406f72fc683b00e27494f7f46e950e5ecd5cbb6c987c8d5aea7798` |
| `fd003-v2-clean` | `bcd2acda16f5a60497725543879d033eca0d83932a7f62525581e44172bfba33` |
| `fd001-v2` | `9c5af0ded502c396df454f2fbc468d839b5bdd680da8f0abc9a416859c05e5da` |
| `m4-fd001-seed42-20260729` (v1, Ridge) | `32e5e7a421ad8ccb81fff4e8b8c17957f49fa855954b27e4e4b16219634af829` |

Each v2 and v3 lock records data, split, config, protocol document, feature order, model,
calibration, and artefact hashes. The evaluator refuses a missing or changed lock, and
a ledger refuses a second lock on a clean holdout. The app checks the lock, evaluation,
and report chain before rendering. See [`results.md`](results.md) for v1 and for what
repeats byte for byte.
