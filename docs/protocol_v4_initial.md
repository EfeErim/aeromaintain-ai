# Protocol v4

Frozen 2026-11-06 before v4 fitting. This is a development experiment, not a
new clean C-MAPSS test. All official C-MAPSS tests were previously opened.

## P0: selection

- FD002 and FD004 training trajectories only. Seed 42 permutes sorted engine IDs;
  first 60% (floor) are selection engines, next 20% calibration, remainder evaluation.
  Evaluation engines are excluded from selection and calibration. These are new
  v4 partitions of previously used training data, not independent external evidence.
- Five grouped folds on selection engines, seed 42. Early stopping uses a separate
  20% inner engine split, seed 42 + fold. Fit condition normalization and all other
  preprocessing on inner fit engines, then refit on outer training engines after
  fixing tree count. Final tree count is the median outer-fold count.
- Five candidates: normalized capped XGBoost reference, normalized uncapped
  XGBoost, raw uncapped XGBoost, normalized uncapped Ridge, raw uncapped Ridge.
  All use compact causal features, uniform engine-equal row weights. Ridge alpha=1.
  XGBoost uses v3 settings: depth 5, learning rate .04, min child weight 10,
  colsample .7, subsample 1, L1=1, L2=5, at most 1500 trees, patience 75.
  The reference target cap is 125; uncapped targets have no upper limit.
- Score uncapped truth using 200 shared draws (seed 43), one uniform cycle per
  engine with no RUL window. Aggregate over draws with equal engine weight.
  Report RMSE, MAE, mean NASA penalty per engine, signed bias, >150 RMSE/bias,
  band counts and critical (truth <=30, prediction >30) misses per draw.
- Paired 95% percentile bootstrap: 2000 resamples of engine columns, seed 44,
  retaining all cuts for each sampled engine. Draws are not independent engines.
- A candidate must improve >150 RMSE on BOTH subsets, not worsen overall RMSE or
  mean NASA penalty on either, and increase critical miss rate by at most 0.02
  absolute on either. Gates use point estimates; confidence intervals show uncertainty.
  Among qualifying candidates choose lowest mean subset RMSE/reference RMSE.
  Otherwise retain the capped reference. No threshold changes after results.

## P1: interval and replay

Refit the selected specification on selection engines only. Calibrate existing
engine-weighted signed-residual Mondrian on all calibration cycles (edges 30,60,100,150;
minimum 20 engines, small bands merge). Compare a split-conformal absolute-residual
reference using one uniformly sampled cycle per calibration engine, seed 45, with
finite sample rank ceil((n+1)*.90). Neither method is tuned on evaluation engines.
Mondrian's repeated-cycle weighting is empirical, not an exchangeability guarantee.

Evaluation uses 200 cuts per engine (seed 46). Report coverage, width, interval score
(alpha=.10), lower/upper misses overall, by true RUL (<=30,31–150,>150), and operating
condition. Flag groups with fewer than 20 distinct engines. Report both methods;
prefer split conformal a priori for its engine-level calibration sampling.

Replay every evaluation engine in cycle order. Alert threshold 30 for point and
lower bound. A persistent event starts on the THIRD consecutive alert cycle;
uninterrupted alert runs count once, a non-alert cycle resets the run. Lead time is
true RUL at confirmation, not first tentative alert. A false event confirms above
30 RUL. A critical engine is missed if no confirmed alert is active at any cycle
with RUL <=30. Publish engine records, lead time, missed engines and false events
per engine alongside interval statistics. Lower-bound alarms prioritize recall
and can trigger earlier/more false events; no production safety claim.

## P2: independent simulation validation

N-CMAPSS is simulation evidence only. DS02 measured sensors (X_s) and operating
conditions (W) may be inputs, with cycle mean/std/min/max summaries. Auxiliary
unit/cycle fields identify groups only. X_v, T, health parameters and Y are forbidden
as features. Y is the supervised target only. Ridge alpha=1, fit-only imputation
and scaling. A three-test-engine DS02 pilot is descriptive, not sufficient evidence
for interval coverage. Require at least 20 genuinely held-out engines for the
confirmatory coverage comparison. Publish data version/hash, engine partitions,
protocol and model hashes before opening those labels. If unavailable, report P2
as blocked rather than claim validation. No transfer experiment is implicitly added.

Holdout claims must use an interprocess lock and atomic ledger replacement. Failed
evaluation consumes the claim conservatively; another model cannot reopen it.

## Reproduction

The machine configuration is `configs/protocol_v4.json`. Local artifacts go under
`runs/v4`; reviewed evidence goes in `docs`. Never overwrite v3 artifacts or relabel
the existing demo as v4 without package and smoke checks.
