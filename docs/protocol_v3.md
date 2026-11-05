# Protocol v3

**Status: frozen on 2026-11-05, before any v3 result was produced.** FD002 and FD004
test labels were unopened at that point. `configs/protocol_v3.yaml` holds the parameters,
and every v3 model lock records the SHA-256 of this file and of that config.

## Goal

Test whether the v2 approach holds under six operating conditions, on the two C-MAPSS
subsets whose test labels are still clean. Candidates that might improve the model and
interval are judged first, on training engines only, because each clean test opens once.

## Fixed rules

All v2 rules carry over. In addition:

1. FD001 and FD003 test labels have been seen. No v3 decision may use them.
2. FD002 and FD004 test labels are opened once each, by the locks this protocol produces.
   Results are published whatever they are.
3. FD001 and FD003 v3 results appear only as secondary results, labeled
   `test previously seen`, after both clean tests have been opened.
4. The v2 runs stay published as v2, and `fd003-v2-clean` keeps its ledger entry.
5. After locks are published, any change that could affect a clean test's result needs a
   new protocol version and can no longer call that test clean.

## Data

Counts below come from the verified archive. RUL label files were not read.

| Subset | Conditions | Fault modes | Train engines | Train rows | Test engines | Test rows |
|---|---:|---:|---:|---:|---:|---:|
| FD002 | 6 | 1 | 260 | 53,759 | 259 | 33,991 |
| FD004 | 6 | 2 | 249 | 61,249 | 248 | 41,214 |

NASA's dataset page lists FD004 as 248 train and 249 test trajectories, the reverse of
the archive. The pipeline follows the archive and fails closed unless the FD004 label
file holds 248 rows.

Engines split 80% / 20% by lifetime quartile: 208/52 for FD002 and 199/50 for FD004.

## Candidates

All candidates are written here before any result exists. Each must clear its rule on
training engines.

### Condition handling (FD002 and FD004 only)

- Fit KMeans with six clusters on standardized operating settings, inside each training
  partition only. The fit fails closed if a setting lies more than 0.1 standard deviations
  from its centroid.
- Standardize each sensor within its condition, using training-partition statistics.
- Compute the existing rolling statistics on normalized sensors, and add
  `cycles_since_condition_change`.

Raw-feature results are the comparison baseline. Normalized features are computed per
fold; raw features stay shared because they use no fitted statistics.

### Training emphasis

| Name | Row weight, on top of engine-equal weights |
|---|---|
| `uniform` (v2) | 1 for all rows |
| `window_focus` | 1 for true RUL at most 150, 0.25 above it |

### Ensemble

Average five seeds of the selected XGBoost settings, as the final stage.

### Interval

The v2 CQR trial reused a tree count tuned for squared error. In v3:

- Each quantile model early-stops on pinball loss over the inner engine split, and the
  final model uses the median stopped tree count.
- Both capped and uncapped targets are tried, because capping limits the upper quantile.
- Each predicted-RUL band gets one conformal margin, with the same engine weighting and
  leave-engines-out evaluation as Mondrian.

## Selection rules

- **Scoring:** the v2 definition: 200 draws, one row per engine, window of 150, with
  sensitivity at 130, 175, and no limit. The window stays at 150 on all subsets; it was
  an analysis choice made on FD001, and the sensitivity table shows its effect.
- **Search:** each subset runs its own search over the same candidates.
- **Stages:** models, feature sets, and caps as in v2, then condition handling, training
  emphasis, and ensemble. Every stage after the first must lower median RMSE at least 1%
  without a worse median NASA score. This is stricter than v2, which checked only RMSE
  at stages.
- **Champion:** XGBoost replaces Ridge under the v2 rule. LSTM is not rerun.
- **Interval:** CQR replaces Mondrian only with leave-engines-out coverage of at least 89%
  and mean width at least 5% lower. If both targets qualify, the narrower one wins.
- **Risk threshold:** the v2 rule, chosen separately per subset from the adopted interval.

## Targets

These are measured on training engines. Missing one is reported, not reinterpreted.

| Target | Value | Basis |
|---|---|---|
| FD001 and FD003 development RMSE | About 16.6 or lower | Carried over from v2 |
| FD001 and FD003 mean interval width | Below 45 | Carried over from v2 |
| FD002 and FD004 development RMSE | At least 10% below the raw-feature baseline | Measures condition handling; no absolute value was chosen |

No target comes from FD002 or FD004 test labels, and no literature figure sets one,
because protocols differ.

## Reporting

For each test, in the v2 format: RMSE, MAE, NASA score, bootstrap intervals, both alert
rules with Wilson intervals, and coverage and width by RUL band. NASA's score sums over
engines and these subsets have about 250, so mean NASA score per engine is also reported.

Seed sweeps score test labels, so they stay limited to FD001 and FD003. Clean tests get
bootstrap intervals only.

## Work order

1. Prepare all subsets and confirm that FD001 raw-feature candidates reproduce v2's first
   selection stage.
2. Lock all four subsets on training engines. No test label is opened.
3. Publish the code, this protocol, the locks' hashes, and development evidence to `main`.
4. Open FD002, then FD004, once each. Publish both.
5. Add secondary FD001 and FD003 results and seed sweeps, then update documents and demo.

A problem found in step 1 or 2 may be fixed if it concerns correctness. Adding or changing
a candidate needs a dated amendment here, written before its result.

## Amendment 1

Written on 2026-11-05 after FD001 and FD003 training-engine evidence, before any FD002 or
FD004 model was fitted. Both tests stayed unopened. This changes run time, not the rules.

`configs/protocol_v3_six_conditions.yaml` applies to FD002 and FD004 only:

- **XGBoost search:** 4 candidates in place of 12. On FD001 the 12 candidates spanned median
  RMSE 18.57–19.20, so more candidates bought little.
- **CQR:** removed. It lost by 54–80% in width on FD001 and FD003 even with tree counts
  tuned on pinball loss, and it costs 20 more fits per subset.
- **Threads:** 6 per run, so that both subsets can run at once.

These subsets could therefore have picked a better setting under the full search. The
`fd001-v3` and `fd003-v3` locks keep `configs/protocol_v3.yaml` and record the document
hash from before this amendment.

## Boundaries

This protocol does not make a simulated benchmark evidence of real-fleet performance.
N-CMAPSS is out of scope for this version. A clean weak result is still published.
