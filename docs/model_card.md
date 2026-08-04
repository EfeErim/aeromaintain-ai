# AeroMaintain AI RUL Model Card

## Model summary

AeroMaintain AI predicts remaining useful life (RUL) for simulated NASA
C-MAPSS FD001 test engines. The locked reference champion is a Ridge regression
pipeline trained with engine-equal sample weights and fold-local preprocessing
over 403 causal features.

This is an educational benchmark prototype, not an airworthiness,
maintenance-approval, production-readiness, or real-fleet system.

## Intended use

- compare a development target mean, Ridge, and bounded XGBoost search under one
  engine-grouped protocol;
- produce a nominal empirical prediction interval and risk band;
- explain model behavior with standardized Ridge coefficients; and
- review one locked official-test evaluation.

It must not be used for dispatch, safety, certification, or real maintenance
approval.

## Data and split

FD001 is simulated run-to-failure turbofan data. A deterministic seed-42 split
assigns 80 training engines to development and 20 to calibration. Model
comparison uses five engine-grouped folds. Official test labels remain isolated
until model lock. See [`data_card.md`](data_card.md).

## Features and selection

Features use only the current and earlier rows: operating settings, current
sensors, engine age, and 5/10/20-cycle sensor statistics. Imputation,
constant-column removal, and Ridge scaling are fitted within each training fold.

XGBoost can replace Ridge only when development RMSE improves at least 5%
without worsening motor-normalized NASA score. The reference candidate improved
RMSE by `1.76396%` and worsened the NASA score, so Ridge remained champion.

| Development model | MAE | RMSE | Motor-normalized NASA score | Critical recall |
|---|---:|---:|---:|---:|
| Target mean | 59.0568 | 74.0016 | 88,366,212.6027 | 0.0000 |
| **Ridge, alpha 100** | **34.3931** | 50.8934 | **3,248,762.7027** | 0.7044 |
| XGBoost candidate 5 | 32.3805 | **49.9957** | 3,893,066.5299 | **0.8464** |

## Locked official-test evaluation

Run: `m4-fd001-seed42-20260729`

| Metric | Result |
|---|---:|
| Engines | 100 |
| MAE | 15.369728 |
| RMSE | 19.622062 |
| Motor-normalized NASA score | 625.326953 |
| Signed bias | -1.777402 |
| Point-threshold critical precision | 1.000000 |
| Point-threshold critical recall | 0.480000 |
| Point-threshold critical F1 | 0.648649 |

The nominal 90% empirical interval observed 0.89 coverage with mean width
60.623140. It is not a safety guarantee. Interface risk bands are assigned from
`interval_low`, not the point-threshold classification.

## Explanations and limitations

Standardized Ridge coefficients describe model behavior, not physical
causality or component diagnosis. FD001 contains one simulated operating
condition and one simulated fault mode. Critical recall is 0.48, observed
interval coverage is below target, and neither result establishes performance
under real distribution shift.

## Reproducibility and integrity

Reference model-lock SHA-256:
`32e5e7a421ad8ccb81fff4e8b8c17957f49fa855954b27e4e4b16219634af829`.
The lock records data, split, config, feature order, model, calibration, and
artefact hashes. The evaluator refuses a missing or changed lock. The app checks
the lock, evaluation, and completed-pipeline report chain before rendering.
