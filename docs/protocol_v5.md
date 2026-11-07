# Protocol v5

Frozen 2026-11-07, before any v5 number is computed. v5 is the project's last experiment.
It has two parts. Part A is development evidence only. Part B is a single clean test on
35 N-CMAPSS engines whose labels have never been read.

**Stopping rule.** Whatever either part shows, there is no second attempt, no new
candidate and no threshold change. Results are published as they come, and the project
closes.

## Part A: capped/uncapped hybrid on C-MAPSS (development only)

**Question.** Protocol v4 found that uncapped XGBoost cut the error on long-life engines
but missed more critical engines. Can a fixed switch keep both?

**Rule.** Use the capped reference prediction when it is below 115 cycles. Otherwise use
the normalized uncapped XGBoost prediction. The threshold of 115 is fixed, ten cycles
under the cap, and is not tuned.

**Data.** The stored v4 selection predictions, `runs/v4/{FD002,FD004}/oof.parquet`
(columns `reference` and `xgboost_normalized`), scored on the stored v4 cuts in
`cuts.npy`. Nothing is fitted. File hashes are recorded in the result.

**Gates.** Same as v4, against the reference, on both subsets:

- RMSE above 150 cycles improves.
- Overall RMSE and mean NASA penalty do not worsen.
- Critical miss rate rises by at most 0.02.

The v4 paired engine bootstrap (2,000 resamples, seed 44) gives the uncertainty.

**Status of the result.** The rule was designed after the v4 results were seen, on the
same engines. Even if it passes, the result is development evidence. The reference
model, the published results and the demo do not change.

## Part B: final clean test on N-CMAPSS

### Data and roles

| Role | Files | Engines | Labels |
|---|---|---:|---|
| Training | DS02-006, DS03-012, DS04, DS05, DS06, DS07 | 64 | Previously opened; free to use |
| Final test | DS01-005, DS08a-009, DS08c-008 | 35 | Never read |

- **Training engines.** All engines in the six training files are used, including both
  archive splits.
- **Final test engines.** These are fixed by
  [`final_holdout_reservation.json`](ncmapss_locks/final_holdout_reservation.json),
  published in commit `6dcbec0`.
- **Excluded file.** DS08d stays excluded because the file is truncated.
- **Engine IDs.** Unit IDs are namespaced by file: `1000 * file_index + unit`. Training
  files take indices 0–5 in sorted order, and test files take 10–12.

### What the test measures

Training and test engines come from different simulation subsets. The test therefore
measures transfer to unseen subsets, not just to unseen engines.

The project reads no flight-class (`Fc`) or health (`hs`) field, so the shift is defined
only by file identity. The archive's own description covers DS02 only: lives of 59–89
cycles, with HPT and combined HPT+LPT failure modes. Per-subset failure modes and flight
classes for DS01 and DS08 were not verified from a primary source and are not claimed
here. A poor result can therefore reflect subset shift as much as model weakness.

### Features, candidates and selection

- **Features.** 72 measured cycle summaries: mean, standard deviation, minimum and
  maximum of the 14 `X_s` sensors and the 4 `W` operating channels. No cycle index, unit
  ID, flight class, `X_v`, `T` or `hs` field is used as input. `Y` is the target only.
  This is the same as v4 P2.
- **Target.** Uncapped RUL, with engine-equal sample weights.
- **Candidate 1, `ridge`.** Median imputation, standard scaling and Ridge with alpha 1,
  each fitted on training rows only.
- **Candidate 2, `xgboost`.** v3 settings: depth 5, learning rate 0.04, min child weight
  10, colsample 0.7, subsample 1, L1 1, L2 5, at most 1,500 trees, patience 75.
  - Early stopping holds out 20% of engines inside each fold (seed 42 + fold).
  - The final tree count is the median of the fold counts.
- **Cross-validation.** Five engine-grouped folds over the 64 training engines (seed 42)
  give out-of-fold (OOF) predictions, clipped at zero.
- **Selection.** Score the OOF predictions on 200 shared cuts (seed 43): one uniform
  cycle per engine per draw, with no RUL window. The candidate with the lower RMSE is
  chosen; Ridge wins an exact tie. The test engines play no part in selection.
- **Final model.** Refit the chosen candidate on all 64 training engines.
- **Baseline.** The engine-equal mean RUL over training cycles.

### Intervals (90% nominal)

- **Mondrian (primary).** Engine-weighted signed-residual quantiles on all OOF training
  cycles. Bands follow predicted RUL with edges 30, 60, 100 and 150, and any band with
  fewer than 20 engines is merged. This is the same function as v3 and v4.
- **Split conformal (secondary).** The absolute OOF residual at one cycle per training
  engine (seed 45), with rank `ceil((64 + 1) * 0.9) = 59`.

Calibrating on OOF residuals and then applying the result to a model refitted on all
engines is cross-conformal in spirit, not a finite-sample guarantee.

### Lock before opening

The `prepare` stage runs only on training files. It does not open, hash-read or
summarize any final test file. It writes the model, the OOF development evidence and
`model_lock.json`, and it copies the lock to
`docs/ncmapss_locks/final_v5.json`. The lock records:

- protocol, config and source hashes;
- training data hashes;
- the reserved test hashes, copied from the reservation;
- the training and test engine lists;
- the chosen candidate and tree count;
- the Mondrian bands, the split margin and the baseline;
- the Git commit.

The lock and its development evidence are committed and pushed before evaluation.

The `evaluate` stage refuses to run unless all of these hold:

- the Git working tree is clean;
- `docs/ncmapss_locks/final_v5.json` in `HEAD` is byte-identical to the run's lock;
- the source hash still matches the lock;
- each test file's SHA256 matches the reservation.

It then claims `FINAL-v5` in `runs/ncmapss_holdout_ledger.json` with the lock hash.
Only after that does it summarize the test files, check their unit sets against the
reservation and read `Y`. A failed evaluation still consumes the claim.

### Scoring

- **Cuts.** 200 per test engine (seed 46), one uniform cycle per engine per draw, with
  equal engine weight. The 35 engines are the unit of evidence, not the cuts.
- **Point metrics.** RMSE, MAE, mean NASA penalty, bias and sampled critical miss rate
  (truth at most 30, prediction above 30), overall and per file. A file with 10 or 15
  engines is descriptive only.
- **Intervals.** Coverage, width and interval score for both methods, overall, by true
  RUL band (at most 30, 31–60, above 60) and per file.
- **Uncertainty.** An engine-column bootstrap (2,000 resamples, seed 44) gives 95%
  intervals for RMSE, the RMSE gap to the baseline and Mondrian coverage.
- **Alarm replay.** Every test engine is replayed in cycle order, with a threshold of 30
  and persistence of 3 consecutive cycles, as in v4.
  - The alarm signal is the point prediction. Mondrian and split lower-bound alarms are
    also reported.
  - An engine counts as **timely detected** only if a confirmed alarm is active at some
    cycle with true RUL between 10 and 30. An alarm that confirms only in the last 10
    cycles counts as missed.
  - Lead time and false events per engine are reported.

### Gates (fixed now, on point estimates)

| Gate | Pass condition |
|---|---|
| G1 point accuracy | Test RMSE is at most the baseline's test RMSE |
| G2 interval coverage | Mondrian coverage is at least 0.85 |
| G3 timely alarm | Engines not timely detected by the point alarm are at most 10% of test engines with a critical cycle |

v5 passes only if all three pass. Confidence intervals are reported beside every gate
and do not change its outcome. With 35 engines, G3 allows at most 3 missed engines.

### Boundaries

- N-CMAPSS is simulated. A pass is simulation evidence, not airworthiness or
  fleet evidence.
- The 35 engines give limited precision, and per-file results are descriptive.
- These test engines can never be reused, whatever the outcome.

## Reproduction

- **Configuration.** The machine configuration is `configs/protocol_v5.json`.
- **Part A.**

  ```
  python -m aeromaintain.models.v5 hybrid --output runs/v5/hybrid
  ```

- **Part B.**

  ```
  python -m aeromaintain.models.v5 prepare --output runs/v5/final
  python -m aeromaintain.models.v5 evaluate --output runs/v5/final
  ```

  Commit and push the lock between the two commands.
