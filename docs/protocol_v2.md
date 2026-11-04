# Protocol v2

Written on 2026-11-04, before any model result produced by this protocol was
inspected. `configs/project.yaml` holds its parameters and every v2 model lock
records the SHA-256 of this file and of that config.

## Why v2 exists

Protocol v1 selected models by scoring every row of every development engine.
Almost all of its NASA score came from rows with true RUL above 125, which the
capped target cannot predict, so selection was not measuring what the test
measures: one prediction per engine at an unknown point of its life.

## Fixed rules

1. FD001 test labels have already been seen. Every model, feature, cap,
   interval, and threshold decision uses training engines only.
2. FD001 v2 results are published as a separate locked run labeled
   `protocol v2, test previously seen`.
3. FD003 test labels are opened once, after this protocol is frozen, and the
   result is published whatever it is.
4. The v1 reference run `m4-fd001-seed42-20260729` stays published as v1.

## Scoring definition

Development and test metrics are computed by the same function on one row per
engine. Development repeats that measurement:

- For each of 200 seeded draws, pick one uniformly random eligible cycle per
  engine and score only that row against uncapped `rul_true`.
- A cycle is eligible when its true RUL is at most 150. **This window is an
  analysis choice, made after FD001 test labels were seen; it is not independent
  evidence.** Without a window, early-life rows would dominate every metric and
  reward higher caps. Each selection run therefore also reports the decision
  under windows of 130, 175, and no limit. A disagreement is published, and does
  not change the decision.
- All candidates are scored on the same draws. Metrics are out-of-fold medians
  across draws.

## Early stopping

XGBoost early stopping uses an inner engine split: 20% of each outer training
fold's engines, seeded and grouped. The outer fold is never seen during fitting,
so it scores the model without the v1 stopping bias.

## Selection stages

Each stage runs automatically, so FD003 follows the identical procedure.

1. **Models:** at cap 125 on all features, compare the Ridge alpha grid and the
   seeded XGBoost random search. Each family keeps its candidate with the lowest
   median RMSE, with median NASA score breaking ties.
2. **Features:** each family's best candidate is compared across the feature sets
   `all`, `no_age`, `no_settings`, and `compact`.
3. **Cap:** each family's result is compared at caps 125 and 130.

A later stage replaces the earlier choice only when it lowers median RMSE by at
least 1%. The earlier, simpler choice wins otherwise.

## Champion rule

XGBoost replaces Ridge only when its median RMSE is at least 5% lower and its
median NASA score is not worse. Otherwise Ridge remains champion. The final model
refits on all training engines, using the selected settings and the median
early-stopped tree count.

## Target and escalation

The target is a median development RMSE of about 16.6 cycles or lower. LSTM is
considered only if the selected model misses it. The target does not change the
champion rule, and a miss is reported rather than hidden.

## Interval

After selection, a grouped 10-fold cross-fit over all training engines produces
out-of-fold predictions with the locked settings.

- Each eligible row is weighted by one over its engine's eligible rows, so each
  engine contributes equally, as one test engine would.
- Mondrian bands follow **predicted** RUL, because true RUL is unknown at test
  time: `<=30`, `31-60`, `61-100`, `>100`. A band with fewer than 20 engines
  merges into its neighbor.
- Each band stores two signed-residual quantiles at 5% and 95%, using the
  finite-sample correction for that band's engine count.

The 90% coverage and sub-45 mean-width target are checked with leave-engines-out
evaluation on training engines. This cross-fit carries no formal finite-sample
guarantee: scores come from slightly smaller models, and the eligibility window
is an assumption about the test population.

## Risk rule

`interval_low <= T` flags an engine as critical. A lower `T` raises fewer alarms,
so `T` is the smallest multiple of 5 from 0 to 60 whose leave-engines-out median
recall is at least 95%. If none reaches 95%, the smallest threshold with the highest
recall is used and the miss is reported. Miss and false-alarm counts are published
side by side for every threshold. `T + 30` is the elevated-band boundary. A
separate classifier is not added.

## Uncertainty and reproducibility

- Official metrics carry engine-level bootstrap 95% intervals with 2,000 seeded
  resamples.
- Ten seeds repeat the split, folds, and model seed with locked settings. Selection
  is not rerun for each seed.
- Two identical runs are compared artifact by artifact to document what repeats
  byte for byte.

## Amendment 1

Written on 2026-11-04 after the first FD001 selection result on training engines,
before any LSTM or CQR result. No test label had been opened under v2.

That result missed two targets: median RMSE 18.57 against about 16.6, and mean
interval width 55.7 against 45. The window stays at 150 even though a window of
130 would report 16.35, because moving it after seeing that number would only
change the measurement.

- **LSTM:** three fixed configurations run on the same folds and draws as the
  other models. Each reads the last 30 cycles of settings and sensors, with fold-local
  scaling, cap 125, engine-equal weights, and early stopping on the inner engine
  split. LSTM replaces XGBoost only if its median RMSE is at least 5% lower and its
  median NASA score is not worse. A miss is reported, and XGBoost stays.
- **CQR:** quantile XGBoost at 5% and 95% with the locked settings, conformalized
  with the same engine weighting and leave-engines-out evaluation. It replaces the
  Mondrian interval only if leave-engines-out coverage is at least 89% and mean width
  is at least 5% lower. A miss is reported, and Mondrian stays.
- **FD003:** it reruns the automated stages. It uses LSTM or CQR only if that method
  earned its place on FD001 under these rules.

## Boundaries

This protocol does not make a simulated benchmark evidence of real-fleet
performance. FD001 v2 test results are not clean, because those labels were seen.
