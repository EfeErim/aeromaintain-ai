# Protocol history

Every experiment in this project, in the order it was run. The [README](../README.md)
gives the final picture; this page keeps the details behind it. Each protocol was
frozen before the labels it was tested on were opened.

| Protocol | Data | Status | Outcome |
|---|---|---|---|
| v2 | C-MAPSS FD003 (FD001 seen) | Clean single-open test | RMSE 15.07, coverage 92% |
| v3 | C-MAPSS FD002, FD004 (FD001, FD003 seen) | Clean single-open tests | RMSE 27.4, coverage 83%; RUL cap diagnosed after opening |
| v4 | C-MAPSS FD002/FD004 training engines, N-CMAPSS DS02–DS07 | Development; simulation benchmark | No uncapped model passed; capped model kept |
| Retrospective | Stored v3 test predictions | Labels already opened | RMSE 14.74 and 15.33 with truth capped at 125 |
| v5 | N-CMAPSS DS01, DS08a, DS08c; C-MAPSS v4 predictions | Final clean test; development | All gates passed; hybrid rule passed on development data only |

Full results: [v2 and v3](results.md), [v4](results_v4.md),
[retrospective](results_retrospective.md), [v5](results_v5.md).

## Protocol v5: final N-CMAPSS clean test

See [results_v5.md](results_v5.md). The final test passed all three predeclared gates
on 35 never-read engines: RMSE 13.06 against a 22.70 baseline, Mondrian coverage 92.5%,
and 3 of 35 engines without a timely alarm, at the limit. The capped/uncapped hybrid
rule passed the v4 gates on C-MAPSS development data, but was designed after seeing v4.

## Removing the RUL cap: protocol v4

[Protocol v4](protocol_v4.md) asked whether uncapped targets fix the long-RUL
underprediction found under v3, without losing critical engines. They do not clear that
bar, so **the capped v3 model is retained**. On v4 development engines, uncapped XGBoost
lowered RMSE on engines above 150 cycles from 79.3 to 52.1 on FD002 and from 110.3 to
64.3 on FD004, but its critical miss rate rose from 16.4% to 21.4% and from 15.8% to
24.1%, past the predeclared limit of +2 points. Every uncapped candidate failed that
gate on both subsets.

These are not clean tests: official test labels were not used, but the training engines
and hyperparameters were seen under v3. Mondrian intervals covered 92% overall and at
least 90% in each RUL band, at the cost of widths of 180–240 cycles above 150. A Ridge
benchmark on 20 held-out N-CMAPSS simulation engines passed its two predeclared gates
(RMSE 15.51 against a 23.56 baseline, coverage 92.8%), yet still predicted above 30 on
48% of sampled critical cycles. See the [full v4 results](results_v4.md).

## Six operating conditions: protocol v3

[Protocol v3](protocol_v3.md) added operating-condition normalization and was
frozen before FD002 or FD004 labels opened. Normalization lowered development RMSE by
10.6% on FD002 and 10.1% on FD004, which met its 10% target. The clean tests then came
in far above development: RMSE 27.4 against 18.9 and 19.6, with interval coverage below
its nominal 90%.

| Metric | FD002, clean | FD004, clean |
|---|---:|---:|
| Run | `fd002-v3-clean` | `fd004-v3-clean` |
| RMSE | **27.36** (23.77–30.82) | **27.40** (24.53–30.23) |
| MAE | 18.46 (15.96–21.01) | 19.65 (17.30–21.99) |
| Signed bias | -8.98 (-12.12 to -5.71) | -9.34 (-12.62 to -6.31) |
| NASA score, total | 10,388 (5,193–16,823) | 5,126 (3,627–6,816) |
| NASA score per engine | 40.1 | 20.7 |
| Interval coverage, nominal 90% | 82.6% (78.0–87.3%) | 82.7% (77.8–87.1%) |
| Mean interval width | 55.8 (52.9–58.7) | 58.5 (55.6–61.3) |

**Why the gap: a diagnosis made after opening.** Predictions top out near 127 cycles
because training targets are capped at 125, while true RUL reaches 194–195 on these
tests. The 14–17% of engines above 150 cycles produce 67–69% of the squared error. The
development window of 150 excluded such rows, so development scoring could not warn of
this. FD001 and FD003 never exposed it, since their true RUL stops at 145.

| Engines by true RUL | FD002 RMSE, coverage | FD004 RMSE, coverage |
|---|---:|---:|
| At most 150 | 16.39, 92% (223 engines) | 17.38, 93% (206 engines) |
| Above 150 | 61.00, 25% (36 engines) | 54.31, 31% (42 engines) |

These subgroup figures are explanatory and do not replace the headline results. Both
tests have now been opened, so a fix would need a new protocol and could not call
either one clean again.

Alert results hold up better, because critical engines sit far from the cap:

| Rule | Dataset | Recall | Precision | Missed | False alarms |
|---|---|---:|---:|---:|---:|
| `prediction <= 30` | FD002 | 57/61, 93% (84–97%) | 93% (84–97%) | 4 | 4 |
| `interval_low <= 25` | FD002 | 60/61, 98% (91–100%) | 73% (63–82%) | 1 | 22 |
| `prediction <= 30` | FD004 | 45/53, 85% (73–92%) | 90% (79–96%) | 8 | 5 |
| `interval_low <= 20` | FD004 | 51/53, 96% (87–99%) | 76% (65–85%) | 2 | 16 |

## One operating condition: protocol v2

Both runs use the same frozen [protocol v2](protocol_v2.md). Ranges are 95%
engine-level bootstrap intervals. Secondary v3 results on these subsets, labeled
`test previously seen`, are in [the results page](results.md).

| Metric | FD003, clean single-open test | FD001, test previously seen |
|---|---:|---:|
| Run | `fd003-v2-clean` | `fd001-v2` |
| RMSE | **15.07** (12.61–17.50) | 15.80 (12.87–18.78) |
| MAE | 11.23 (9.34–13.28) | 11.15 (9.10–13.43) |
| Motor-normalized NASA score | 399 (240–620) | 398 (233–625) |
| Interval coverage, nominal 90% | 92% (86–97%) | 94% (89–98%) |
| Mean interval width | 53.3 (49.3–57.3) | 55.4 (51.2–59.4) |

**Read the FD001 column with care.** Its test labels were seen before v2 was
designed, so it is not independent evidence. Seed 42 is also the best of ten seeds
on FD001: across seeds, RMSE averages 16.03 (SD 0.14, range 15.80–16.21).

### Critical-engine alerts

A truly critical engine has at most 30 remaining cycles. Each threshold was chosen
on training engines to reach 95% recall with the fewest alarms. Ranges here are
Wilson 95% intervals.

| Rule | Dataset | Recall | Precision | Missed | False alarms |
|---|---|---:|---:|---:|---:|
| `prediction <= 30` | FD003 | 16/20, 80% (58–92%) | 94% (73–99%) | 4 | 1 |
| `interval_low <= 25` | FD003 | 20/20, 100% (84–100%) | 71% (53–85%) | 0 | 8 |
| `prediction <= 30` | FD001 | 20/25, 80% (61–91%) | 100% (84–100%) | 5 | 0 |
| `interval_low <= 20` | FD001 | 24/25, 96% (80–99%) | 86% (69–94%) | 1 | 4 |

The interface uses the interval rule: it misses fewer critical engines and raises
more false alarms. The interval is not a safety guarantee.

## Targets

| Target | Result | Status |
|---|---|---|
| FD001 and FD003 development RMSE at most about 16.6 | v2: 18.57, 17.62. v3: 18.57, 17.86 | **Missed** |
| FD001 and FD003 mean interval width below 45 | v2 training: 55.7, 52.0. v3 training: 55.7, 53.2 | **Missed** |
| FD002 and FD004 development RMSE at least 10% below raw features | 10.6%, 10.1% | Met |
| Risk rule at 95% development recall | FD001 96%, FD002 98%, FD003 100%, FD004 97% | Met |
| One clean holdout check per subset | FD002, FD003, and FD004 each opened once | Met |

No target was set for FD002 or FD004 test RMSE, so 27.4 misses no written figure, but
the coverage shortfall is real: both bootstrap ranges stop below 90%. On FD001 and
FD003, test RMSE falls below 16.6, but that target was defined on development data and
stays missed.

The other v3 candidates did not change the picture. Training emphasis passed nowhere,
and the five-seed ensemble passed only on FD003. Quantile regression with its own tuned
tree counts still produced intervals 54–80% wider than Mondrian on FD001 and FD003. In
v2, LSTM reached 18.39 on FD001 with a worse NASA score.

## C-MAPSS model selection (v2 and v3)

Development scoring matches the test: one random row per engine, scored against
true RUL, repeated over 200 seeded draws. XGBoost can replace Ridge only by lowering
median RMSE at least 5% without a worse median NASA score. It did so on all four
datasets and won at least 99% of paired draws on each.

These draws use rows with true RUL of at most 150. That window was chosen after
FD001 test labels had been seen, so each run reports the alternatives: XGBoost also
wins at 130 and 175, while Ridge wins with no window because of its better NASA
score. Scores with no window are far worse for both models, because random cuts then
reach rows with more remaining life than any test engine has.
