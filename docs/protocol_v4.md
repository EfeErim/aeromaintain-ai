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

### Independent-data amendment 1 (before N-CMAPSS label reading or fitting)

The publication describes DS08 with 54 engines, so the planned confirmatory experiment
uses DS08 rather than pooling DS02's three test engines. This is a separate within-DS08
Ridge simulation benchmark, not a claim that C-MAPSS weights transfer to N-CMAPSS.
Permute sorted DS08 engine IDs using seed 42: first 20 fit, next 14 calibrate, last
20 evaluate, regardless of the archive's dev/test suffix. Reject overlapping unit IDs
across suffixes. Use measured-sensor and operating-condition cycle mean/std/min/max
only (no engine ID, flight class, health state or cycle age as input).
Fit Ridge alpha=1 with engine-equal weights; impute/scale on fit engines only.
Split conformal uses one cycle per calibration engine (seed 45), rank
ceil((14+1)*.9)=14. Score 200 full-range cuts per evaluation engine, seed 46.
Bootstrap evaluation engine columns 2000 times, seed 44, for RMSE and coverage.
Predeclared success: empirical coverage >=.85, and RMSE no worse than an engine-equal
mean-RUL training baseline. Publish failures as well as passes; 20 engines still
give limited precision. Record source hash and partitions in a locally published
lock before reading evaluation Y arrays. DS02 is a separate descriptive pilot:
fit original dev engines and evaluate original test engines after locking Ridge.

### Independent-data amendment 2 (archive inspection; no N-CMAPSS labels read)

Amendment 1's DS08 assumption is not supported by the downloaded archive: DS08a
has 15 units, DS08c 10, and DS08d fails HDF5 opening (stored EOF is 32 bytes beyond
the file length declared by the ZIP). Do not pad, repair or use this member.
Use the five intact DS03–DS07 members instead. Within each file, permute sorted
unit IDs with seed 42; first four engines evaluate, next three calibrate, remaining
engines fit. Namespace unit IDs by file; identical numeric IDs in different files
are different simulated units. This gives 20 evaluation and 15 calibration engines.
It is a within-subset-mixture benchmark, not a held-out operating-regime transfer
experiment. The pooled model, features, alpha, sampling, bootstrap and success gates
from amendment 1 remain fixed. Split-conformal rank is ceil((15+1)*.9)=15.
Report subset membership and the limited evidence from four evaluated engines per
subset. Freeze and locally publish model locks before reading evaluation labels.

Holdout claims must use an interprocess lock and atomic ledger replacement. Failed
evaluation consumes the claim conservatively; another model cannot reopen it.

## Reproduction

The machine configuration is `configs/protocol_v4.json`. Local artifacts go under
`runs/v4`; reviewed evidence goes in `docs`. Never overwrite v3 artifacts or relabel
the existing demo as v4 without package and smoke checks.
