# Protocol v5 results

Run on 2026-11-08 under the frozen [protocol v5](protocol_v5.md).

**Order of events:**

1. The protocol and code were committed in `e6e66be`.
2. The final model lock was committed in `82fe6f9` and pushed before any reserved label
   was read.
3. The holdout ledger records one claim, `FINAL-v5`, with lock hash
   `e058e97837741323782d6ea46c7532df8a5ca3134d3ea3445fc04918cdb71844`, made from commit
   `7df9e98` (now `82fe6f9`, see the note below).

Note: the repository history was rewritten twice, on 2026-11-08 to remove commit-message
trailers and on 2026-11-10 to drop local planning files and normalize the author email.
Apart from those planning files, file contents and commit dates are unchanged, but every
commit hash changed: `e6e66be` was `82f56af`, originally `4695a75`, and `82fe6f9` was `1272ed1`,
originally `7df9e98`. The lock hash above does not depend on commit hashes.

Evidence is in [`protocol_v5_evidence.json`](protocol_v5_evidence.json), and development
evidence is in [`protocol_v5_development.json`](protocol_v5_development.json).

## Part B: final N-CMAPSS clean test

**All three predeclared gates pass.** The test covers 35 never-before-opened simulated
engines from DS01, DS08a and DS08c. The model is XGBoost, trained on the 64 DS02–DS07
engines and selected on out-of-fold development evidence.

Ranges are 95% engine-level bootstrap intervals (2,000 resamples). Scores average 200
random cuts per engine, and the 35 engines are the unit of evidence. The ranges below
are the corrected ones; see the erratum after the table.

| Gate | Result | Threshold | Status |
|---|---:|---:|---|
| G1 point accuracy | RMSE **13.06** (11.13–15.00), against a mean baseline of 22.70 (21.40–23.99) | Not worse than baseline | Pass |
| G2 interval coverage | Mondrian **92.5%** (88.4–95.9%) | At least 85% | Pass |
| G3 timely alarm | **3 of 35** engines (8.6%) not alerted with 10–30 cycles left | At most 10%, so at most 3 engines | Pass, at the limit |

### Erratum: bootstrap ranges (2026-11-09)

The ranges first published here, and still stored in
[`protocol_v5_evidence.json`](protocol_v5_evidence.json), were wrong. The frozen
`engine_bootstrap` in `models/v5.py` took the engine count from a
`(values, transform)` tuple, which has length 2, so every resample drew only engines 0
and 1. An earlier version of this page explained the odd RMSE range with a story about
a few high-error engines; that explanation was wrong.

| Metric | Point | Published range | Corrected range |
|---|---:|---:|---:|
| RMSE | 13.06 | 8.92–13.15 | 11.13–15.00 |
| Mean baseline RMSE | 22.70 | 20.47–29.34 | 21.40–23.99 |
| Mondrian coverage | 92.5% | 88.0–98.0% | 88.4–95.9% |

Point values and all three gate decisions are unchanged, since none of them used the
bootstrap. The model's RMSE is below the baseline's in all 2,000 corrected resamples.
`models/v5.py` stays as it was, because its hash is part of the published lock. The
corrected ranges come from
[`scripts/v5_bootstrap_erratum.py`](../scripts/v5_bootstrap_erratum.py), which reads the
stored evaluation without reopening labels and writes
[`v5_bootstrap_erratum.json`](v5_bootstrap_erratum.json). `tests/test_bootstrap.py`
covers the replacement function and pins the defect in the frozen one.

### Point accuracy

| | All 35 engines | DS01 (10) | DS08a (15) | DS08c (10) |
|---|---:|---:|---:|---:|
| RMSE | 13.06 | 14.52 | 13.12 | 11.30 |
| Mean baseline RMSE | 22.70 | 26.94 | 21.19 | 20.13 |
| MAE | 9.51 | 10.21 | 9.32 | 9.08 |
| Signed bias | -0.05 | -6.96 | 1.12 | 5.10 |
| Sampled critical miss rate | 28.4% | 8.2% | 31.8% | 36.4% |

The test RMSE is in line with the out-of-fold development estimate of 12.87. The model
did not lose accuracy on subsets it never saw.

Per-file results rest on 10–15 engines and are descriptive. The DS08 files lean toward
overprediction, which is the unsafe direction.

### Intervals

| Group | Mondrian coverage | Mondrian width | Split conformal coverage | Split conformal width |
|---|---:|---:|---:|---:|
| Overall | 92.5% | 41.7 | 93.0% | 43.2 |
| True RUL at most 30 | 94.3% | 30.1 | 94.8% | 39.0 |
| True RUL 31–60 | 95.9% | 51.4 | 98.9% | 46.9 |
| True RUL above 60 (20 engines) | **76.1%** | 53.0 | **68.9%** | 47.0 |

Both methods stay above nominal overall. They fall short only above 60 cycles, where the
model underpredicts long remaining life, the same direction as the C-MAPSS cap problem.
Near failure, misses are mostly on the optimistic side: in the band at most 30, 154
sampled cuts fall below the Mondrian lower bound.

### Alarm replay

The rule is a persistent alarm: three consecutive cycles predicted at or below 30. An
alarm counts only if it is confirmed while true RUL is between 10 and 30.

| Signal | Missed | Median lead at confirmation | False events per engine |
|---|---:|---:|---:|
| Point prediction (gate G3) | 3 of 35 | 22 cycles | 0.26 |
| Point prediction, any time before failure | 2 of 35 | 22 cycles | 0.26 |
| Mondrian lower bound | 0 of 35 | 55 cycles | 2.57 |
| Split-conformal lower bound | 0 of 35 | 55 cycles | 2.63 |

The three missed engines are:

- DS08a units 12 and 15, which never confirmed an alarm;
- DS08c unit 1, whose alarm confirmed with only 9 cycles left.

Lower-bound alarms catch every engine, but they confirm much earlier and raise about ten
times as many false events. That is a recall-for-false-alarm trade, not a free gain.

### What this does and does not show

- **It shows transfer across simulation subsets.** Under a lock committed and pushed
  before opening, a measured-sensor model trained on DS02–DS07 transfers to unseen
  DS01 and DS08 engines. It beats the baseline by a wide margin, intervals stay near
  nominal, and the timely-alarm gate passes.
- **G3 passed at its limit.** One more missed engine would have failed it. The
  development estimate was 12.5%, so the pass is not robust evidence that the alarm
  meets 90%.
- **Sampled critical misses are still high.** At a random critical cycle, 28% of
  predictions sit above 30. The persistent alarm passes because it needs only three
  consecutive alert cycles somewhere in the window.
- **It is simulation evidence only.** It is not airworthiness, maintenance-approval or
  fleet evidence. These 35 engines cannot be reused.

## Part A: capped/uncapped hybrid on C-MAPSS (development only)

The rule: use the capped reference prediction below 115 cycles, otherwise the uncapped
normalized XGBoost prediction. It was scored on the stored v4 selection predictions and
cuts. Nothing was fitted.

| Dataset | Model | RMSE | RMSE above 150 | Critical miss rate | Gate |
|---|---|---:|---:|---:|---|
| FD002 | Capped reference | 41.21 | 79.28 | 16.4% | |
| FD002 | Uncapped | 33.35 | 52.10 | 21.4% | Failed in v4 |
| FD002 | **Hybrid** | **33.93** | **55.88** | **16.4%** | Pass |
| FD004 | Capped reference | 65.07 | 110.33 | 15.8% | |
| FD004 | Uncapped | 44.63 | 64.26 | 24.1% | Failed in v4 |
| FD004 | **Hybrid** | **45.04** | **67.47** | **15.8%** | Pass |

The hybrid keeps most of the uncapped model's long-life gain and leaves the critical
miss rate unchanged. The 95% paired intervals for the hybrid minus the reference are:

- RMSE: −9.1 to −5.4 on FD002, −24.6 to −15.3 on FD004.
- Critical miss rate: exactly zero on both.

**Why this is development evidence only:**

- The rule was designed after the v4 results were seen, on the same engines.
- By construction it cannot add critical misses, because below 115 it is the reference.
- No clean C-MAPSS test remains to confirm it.

As fixed in the protocol, the reference model, the published C-MAPSS results and the
demo do not change.
