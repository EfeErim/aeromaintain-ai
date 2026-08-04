# Reference Run Results

## Run

The local immutable run `m4-fd001-seed42-20260729` was produced on 2026-07-29.
Its model/data-derived files are Git-ignored; sanitized aggregate evidence is
stored in [`reference_evidence.json`](reference_evidence.json).

## Development selection

Ridge remained champion under the fixed rule. The best XGBoost candidate reduced
development RMSE by only `1.76396%`, below the required 5%, and had a worse
motor-normalized NASA score. Official test results did not influence this choice.

## Locked official-test result

| Metric | Result |
|---|---:|
| Engines | 100 |
| MAE | 15.369728 |
| RMSE | 19.622062 |
| Motor-normalized NASA score | 625.326953 |
| Signed bias | -1.777402 |
| Overprediction rate | 0.480000 |
| Point-threshold critical precision (`prediction <= 30`) | 1.000000 |
| Point-threshold critical recall (`prediction <= 30`) | 0.480000 |
| Point-threshold critical F1 (`prediction <= 30`) | 0.648649 |
| Nominal interval coverage | 0.90 |
| Observed official-test coverage | 0.89 |
| Mean interval width | 60.623140 |

The 0.48 critical recall means the point threshold missed 13 of 25 truly
critical engines. The interface risk bands instead use the empirical interval
lower bound. That interval is not a safety guarantee and its observed coverage
was one percentage point below the nominal target.

## Active result boundary

The repository no longer reports maintenance policies, synthetic costs,
capacity sensitivity, or a CP-SAT schedule. No reviewed public source supplied
the complete operational schema required to replace those invented values.
See [`real_data_scope.md`](real_data_scope.md) for the source review and scope
decision.

## Screenshots

- [Overview](screenshots/overview.png)
- [Engine risk](screenshots/engine-health.png)

Both pages load the named run through the same verified artefact loader used by
the command-line smoke test.
