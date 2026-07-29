# AeroMaintain AI RUL Model Card

## Model summary

AeroMaintain AI predicts remaining useful life (RUL) for the simulated NASA
C-MAPSS FD001 test engines. The locked release-candidate champion is a Ridge
regression pipeline trained with engine-equal sample weights and fold-local
preprocessing over 403 causal features.

This model is an educational prototype. It is not an airworthiness,
maintenance-approval, production-readiness, or real-fleet system.

## Intended use

The model supports a local demonstration of a leakage-resistant
prediction-to-decision workflow:

- compare a development target mean, Ridge, and bounded XGBoost search under one
  engine-grouped protocol;
- produce a nominal empirical prediction interval and risk band;
- explain model behavior with standardized Ridge coefficients; and
- feed locked predictions, not true test RUL, into a synthetic maintenance
  scheduling problem.

Prohibited uses include operational dispatch, safety decisions, component
certification, real maintenance approval, or claims that FD001 performance
generalizes to other conditions, fault modes, engines, or fleets.

## Data and split

FD001 is simulated run-to-failure turbofan data, not operational fleet
telemetry. Training contains 100 engines. A deterministic seed-42,
lifetime-quartile-stratified split assigns 80 engines to development and 20 to
calibration. Model comparison uses five engine-grouped folds within development.
Official test labels are isolated until after model lock.

Raw NASA files are not committed or redistributed. See
[`data_card.md`](data_card.md) for provenance, schema, governance, and license
boundaries.

## Features and model selection

Features use the current row and earlier rows only:

- three operating settings, 21 current sensors, and engine age;
- 5-, 10-, and 20-cycle sensor mean, standard deviation, minimum, maximum,
  linear slope, and last-minus-mean.

Imputation, constant-column removal, and Ridge scaling are fitted inside the
active training fold. XGBoost can replace Ridge only when development RMSE
improves by at least 5% without worsening the motor-normalized NASA score. The
release candidate did not meet that fixed rule, so Ridge remained champion.

## Release-candidate evaluation

Run: `m4-fd001-seed42-20260729`

| Metric | Official test result |
|---|---:|
| Engines | 100 |
| MAE | 15.369728 |
| RMSE | 19.622062 |
| Motor-normalized NASA score | 625.326953 |
| Signed bias | -1.777402 |
| Critical-RUL precision | 1.000000 |
| Critical-RUL recall | 0.480000 |
| Critical-RUL F1 | 0.648649 |

The nominal 90% empirical interval observed 0.89 coverage on the official test,
with mean width 60.623140. This is an empirical result, not a safety guarantee.

## Explanations

Global and local explanations use standardized Ridge coefficients. They
describe model behavior and must not be interpreted as physical causality,
component diagnosis, or proof that a sensor causes degradation.

## Limitations and risks

- FD001 contains one simulated operating condition and one simulated fault mode.
- The capped training target is a modeling assumption; official metrics use
  original uncapped test RUL.
- Critical-RUL recall is 0.48, so more than half of official critical engines
  were not identified by the point-prediction threshold.
- The nominal interval missed its nominal 0.90 coverage by 0.01 on this fixed
  test set.
- Planning fields, resource limits, duration, demand, and costs are synthetic.
- The CP-SAT release-candidate schedules are feasible but not proven optimal.
- The model does not estimate uncertainty under real distribution shift.

## Reproducibility and integrity

The release-candidate model lock SHA-256 is
`32e5e7a421ad8ccb81fff4e8b8c17957f49fa855954b27e4e4b16219634af829`.
`model_lock.json` records data, split, config, feature order, model, calibration,
and artefact hashes. The official evaluator refuses a missing or changed lock.
The Streamlit application verifies the lock, evaluation, optimization, and
pipeline report hash chain before rendering.
