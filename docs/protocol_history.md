# Protocol history

Every experiment, in the order it was run. Each protocol was frozen before the labels it
was tested on were opened.

| Protocol | Data | Status | Outcome |
|---|---|---|---|
| v2 | C-MAPSS FD003 (FD001 seen) | Clean single-open test | RMSE 15.07, coverage 92% |
| v3 | C-MAPSS FD002, FD004 (FD001, FD003 seen) | Clean single-open tests | RMSE 27.4, coverage 83%; RUL cap diagnosed after opening |
| v4 | C-MAPSS FD002/FD004 training engines, N-CMAPSS DS02–DS07 | Development; simulation benchmark | No uncapped model passed; capped model kept |
| Retrospective | Stored v3 test predictions | Labels already opened | RMSE 14.74 and 15.33 with truth capped at 125 |
| v5 | N-CMAPSS DS01, DS08a, DS08c; C-MAPSS v4 predictions | Final clean test; development | All gates passed; hybrid rule passed on development data only |

**v2, one operating condition.** [Protocol v2](protocol_v2.md) replaced v1's all-rows
selection with test-like truncated scoring. FD003 (clean) reached RMSE 15.07 with 92%
coverage. FD001 labels were seen before v2 was designed, and seed 42 is the best of ten
seeds on FD001, so FD001 is not independent evidence. [Results](results.md#protocol-v2-runs).

**v3, six operating conditions.** [Protocol v3](protocol_v3.md) added per-condition
normalization, which lowered development RMSE by 10.6% on FD002 and 10.1% on FD004. The
clean tests came in at 27.4 against 18.9 and 19.6 in development. The cause, found after
opening: targets are capped at 125 while true test RUL reaches 194–195, and the 150-cycle
development window never scored such engines. The 14–17% of engines above 150 cycles
produce 67–69% of the squared error. [Results](results.md#protocol-v3-runs).

**v4, removing the cap.** [Protocol v4](protocol_v4.md) tested uncapped targets on
development engines. They cut long-RUL error but raised the critical miss rate past the
predeclared +2 point limit on both subsets, so the capped model stayed. A Ridge benchmark
on 20 held-out N-CMAPSS engines passed its two gates. [Results](results_v4.md).

**v5, final N-CMAPSS test.** 35 never-read engines, all three predeclared gates passed,
the alarm gate at its limit. [Results](results_v5.md).

## Targets

| Target | Result | Status |
|---|---|---|
| FD001 and FD003 development RMSE at most about 16.6 | v2: 18.57, 17.62. v3: 18.57, 17.86 | **Missed** |
| FD001 and FD003 mean interval width below 45 | v2 training: 55.7, 52.0. v3 training: 55.7, 53.2 | **Missed** |
| FD002 and FD004 development RMSE at least 10% below raw features | 10.6%, 10.1% | Met |
| Risk rule at 95% development recall | FD001 96%, FD002 98%, FD003 100%, FD004 97% | Met |
| One clean holdout check per subset | FD002, FD003, and FD004 each opened once | Met |

No target was set for FD002 or FD004 test RMSE, but the coverage shortfall is real: both
bootstrap ranges stop below 90%.
