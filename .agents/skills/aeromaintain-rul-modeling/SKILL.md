---
name: aeromaintain-rul-modeling
description: Develop, compare, calibrate, explain, persist, lock, and evaluate AeroMaintain AI RUL models. Use for causal rolling features, engine-grouped cross-validation, mean/Ridge/XGBoost models, NASA score, nominal prediction intervals, SHAP or coefficient explanations, model_lock.json, or official test evaluation.
---

# AeroMaintain RUL Modeling

Build a leakage-safe model whose selection is complete before official test
labels are opened.

## Load the contract

Read `AGENTS.md`, `PROJECT_STATE.md`, the modeling sections of
`PROJECT_PLAN.md`, and
[references/evaluation-contract.md](references/evaluation-contract.md).
Require the Phase 1 gate and inspect existing split/data manifests.

## Build causal features

- Group by engine and sort by cycle.
- Use current and earlier rows only; never use centered or future-shifted
  windows.
- Produce current settings/sensors, motor age and 5/10/20-cycle rolling mean,
  standard deviation and slope features.
- Fit imputers, scalers and constant-column filters inside each training fold.
- Persist the final feature names and order.

## Compare models

1. Use only development engines for selection.
2. Reuse the same five GroupKFold engine folds for the development target mean,
   Ridge candidates and XGBoost candidates.
3. Evaluate MAE, RMSE, motor-normalized NASA score, critical-RUL metrics and RUL
   band errors.
4. Use at most 12 XGBoost candidates with `tree_method="hist"`, seed 42, at
   most 1,500 trees and 75-round early stopping.
5. Derive the final tree count from the median fold `best_iteration + 1`; do
   not use calibration engines for early stopping.
6. Select XGBoost only when it improves development RMSE over Ridge by at least
   5% without worsening NASA score. Otherwise select Ridge.

## Calibrate and lock

- Sort calibration engine IDs and assign target RUL cutoffs cyclically:
  20, 60, 100, 126.
- For each engine, select the row whose uncapped true RUL is nearest its
  assigned cutoff; break ties toward the later cycle.
- Produce one absolute residual score per calibration engine.
- For nominal coverage 0.90 and `n` scores, use the ascending order statistic
  at `min(n, ceil((n + 1) * 0.90))`.
- Form `point ± q`, clamp the lower bound to zero and label coverage empirical,
  not guaranteed.
- Produce SHAP TreeExplainer output for XGBoost or standardized coefficients
  for Ridge.
- Save XGBoost in native JSON or the local Ridge pipeline with `joblib`.
- Create `model_lock.json` with data/split/config hashes, seed, feature order,
  champion rule evidence, model hash and calibration `q`.

## Run official evaluation

- Require a valid lock and matching model/data hashes.
- Load official test RUL only inside the evaluation command.
- Write predictions and metrics without altering the champion.
- Reject any attempt to feed test labels back into training, calibration,
  threshold choice or feature selection.

Run focused feature, grouping, metric, persistence, interval and lock tests.
Record completion evidence through `$aeromaintain-phase-gates`.
