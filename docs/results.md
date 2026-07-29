# Reference Run Results

## Run

These results come from the local immutable run
`m4-fd001-seed42-20260729`, produced on 2026-07-29.

The run is Git-ignored because it contains model and data-derived artefacts.
Its final manifest status is `pipeline_complete`; report and source-manifest
hashes are retained locally.

## RUL prediction

Ridge remained champion under the fixed selection rule. The independent M4 run
reproduced the previously locked official-test metrics:

| Metric | Result |
|---|---:|
| Engines | 100 |
| MAE | 15.369728 |
| RMSE | 19.622062 |
| Motor-normalized NASA score | 625.326953 |
| Signed bias | -1.777402 |
| Overprediction rate | 0.480000 |
| Critical-RUL precision | 1.000000 |
| Critical-RUL recall | 0.480000 |
| Critical-RUL F1 | 0.648649 |
| Nominal interval coverage | 0.90 |
| Observed official-test coverage | 0.89 |
| Mean interval width | 60.623140 |

The empirical interval is not a safety guarantee. Critical-RUL recall and the
coverage shortfall are reported alongside the headline error metrics.

## Policy comparison

All operational and cost fields are synthetic. `cost_units` are not currency.

| Policy | Scheduled | Due deferrals | Unplanned failures | Total synthetic cost |
|---|---:|---:|---:|---:|
| Reactive | 0 | 20 | 20 | 10,000 |
| Fixed 90 | 13 | 7 | 18 | 10,349 |
| Predicted RUL 30 | 16 | 4 | 18 | 10,662 |
| CP-SAT | 17 | 3 | 13 | 8,287 |

The base CP-SAT solver status was `FEASIBLE`; lexicographic optimality was
unproven. It recorded zero operating-capacity shortfall, but still deferred
three due jobs and retained 13 retrospective simulated failures.

## Capacity sensitivity

| Scenario | Bays | Minimum operating demand | Scheduled | Due deferrals | Unplanned failures | Total synthetic cost |
|---|---:|---:|---:|---:|---:|---:|
| Constrained | 1 | 90% | 10 | 10 | 17 | 9,542 |
| Base | 2 | 80% | 17 | 3 | 13 | 8,287 |
| Expanded | 3 | 70% | 19 | 1 | 12 | 8,081 |

All three outcomes were `FEASIBLE` with unproven optimality. This table is a
capacity sensitivity check, not a tuned policy recommendation.

## Screenshots

- [Overview](screenshots/overview.png)
- [Engine risk](screenshots/engine-health.png)
- [Maintenance plan](screenshots/maintenance-schedule.png)
- [Policy analysis](screenshots/policy-comparison.png)

All four pages were opened against the named run without browser-console
errors. The Streamlit test fixture also covers every page and one validated
capacity replan.
