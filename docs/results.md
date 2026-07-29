# Verified Results

## Evidence scope

These results come from the local immutable run
`m4-fd001-seed42-20260729`, produced on 2026-07-29 by:

```powershell
aeromaintain pipeline --run-id m4-fd001-seed42-20260729
```

The run is Git-ignored because it contains generated model and data-derived
artefacts. Its final manifest status is `pipeline_complete`; the report and
source manifest SHA-256 values are retained locally. The committed screenshots
are presentation evidence, not substitutes for the machine-readable run.

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

The empirical interval does not provide a safety guarantee. Critical-RUL recall
and the coverage shortfall are visible negative results.

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

## Application evidence

- [Overview](screenshots/overview.png)
- [Engine Health](screenshots/engine-health.png)
- [Maintenance Schedule](screenshots/maintenance-schedule.png)
- [Policy Comparison & What-if](screenshots/policy-comparison.png)

The browser verification covered all four pages with the explicit release run
and reported no console errors. The automated Streamlit fixture also exercises
every page and a validated truth-free what-if solve.
