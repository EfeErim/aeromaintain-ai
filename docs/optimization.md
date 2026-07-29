# Maintenance optimization

## Scope and safety boundary

The optimizer turns a locked FD001 evaluation into a synthetic maintenance
experiment. It is not an airworthiness, maintenance-approval,
production-planning, or real-fleet system. Staffing, bays, parts,
operating-demand, duration, and cost fields are synthetic. Costs are
dimensionless `cost_units`, not currency.

The planning boundary accepts only engine ID, last observed cycle, point RUL
prediction, nominal empirical interval, and risk band. A frame containing
`rul_true`, `true_rul`, `actual_rul`, or the official interval-coverage flag is
rejected. Official true RUL is joined only after a schedule has been frozen by
the separate retrospective evaluator. The optimizer schema and persisted
scenario therefore contain no true-RUL field.

## Scenario

The default scenario uses the 20 official test engines with the lowest
`interval_low`, breaking ties by engine ID, a 30-day horizon, and seed `42`.
The generator draws the integer ranges declared in `configs/scenario.yaml`:

| Synthetic field | Range or values |
|---|---|
| cycles per day | 1-4 |
| maintenance duration | 2-5 days |
| technicians required | 2-6 |
| part kit | `kit_A`, `kit_B`, or `kit_C` |
| kit quantity | 1-2 |

The base case has two six-technician teams, two bays, four starting units of
each kit, three-unit replenishments at the start of days 10 and 20, and an 80%
minimum daily operating-capacity demand. The generated `scenario.json` records
every value plus generator version; `scenario_data_dictionary.json` labels
source, prediction-derived, and synthetic fields.

For each engine:

```text
point_rul_cycles = floor(prediction)
lower_rul_cycles = floor(interval_low)
safe_due_day = max(0, floor(lower_rul_cycles / cycles_per_day) - 1)
```

A due day beyond the horizon is marked as not due within this horizon.

## Policies and common evaluation

Four policies use the same generated fleet and resources:

1. reactive: no proactive work is scheduled; failures are observed only
   retrospectively;
2. fixed 90 cycles: the next 90-cycle boundary triggers work;
3. predicted RUL 30: projected point RUL reaching 30 triggers work;
4. CP-SAT: interval-lower-bound safe due dates drive constrained scheduling.

Rule policies place triggered work on the earliest feasible day in
safe-due-day and engine-ID order, and explicitly defer work that cannot be
placed. The common evaluator checks the same team, bay, parts, operating
demand, and horizon rules for every schedule before computing decision metrics.

## CP-SAT model

Binary integer variables represent each engine/start-day/team/bay assignment
and explicit deferral. A scheduled job keeps the same team and bay for its full
duration. Constraints enforce:

- one scheduled assignment or one explicit deferral per engine;
- six-technician capacity for each team on every day;
- no overlapping work in one bay;
- cumulative kit consumption not exceeding initial stock plus replenishments;
- available daily engine-cycle capacity meeting the scenario demand; and
- completion by day 30.

Stage 1 minimizes due-engine deferrals lexicographically before total late
days. Stage 2 fixes the best stage-1 score found, then minimizes planned
maintenance, unused predicted-cycle, and low-risk deferral `cost_units`. Each
stage uses seed `42`, one search worker, and a 30-second limit. `FEASIBLE` is
kept distinct from `OPTIMAL`; when stage 1 is not proven optimal, the schedule
is labelled `unproven`. `INFEASIBLE`, `MODEL_INVALID`, and `UNKNOWN` results
contain no schedule.

## Reference optimization run

Run: `m2-fd001-seed42-20260729`

The base solver found a valid `FEASIBLE` schedule within the two-stage time
limits but did not prove lexicographic optimality.

| Policy | Scheduled | Due deferrals | Late days | Retrospective failures | Total synthetic cost |
|---|---:|---:|---:|---:|---:|
| reactive | 0 | 20 | 594 | 20 | 10,000 |
| fixed 90 | 13 | 7 | 396 | 18 | 10,349 |
| predicted RUL 30 | 16 | 4 | 328 | 18 | 10,662 |
| CP-SAT | 17 | 3 | 264 | 13 | 8,287 |

All four schedules report zero operating-capacity shortfall and pass the common
resource validator.

| Capacity | Bays | Demand | Scheduled | Due deferrals | Failures | Total synthetic cost |
|---|---:|---:|---:|---:|---:|---:|
| constrained | 1 | 90% | 10 | 10 | 17 | 9,542 |
| base | 2 | 80% | 17 | 3 | 13 | 8,287 |
| expanded | 3 | 70% | 19 | 1 | 12 | 8,081 |

These results are a capacity sensitivity check, not a tuned policy claim. The
retrospective failure counts use NASA simulation truth after schedule freeze;
they are not available to any scenario, policy, or solver decision.

## Artefacts and reproducibility

Before writing a new immutable `optimization/` directory, the optimization
stage verifies the model lock and official prediction hashes. The directory
contains:

- `scenario.json` and `scenario_data_dictionary.json`;
- four policy schedules and three capacity schedules;
- policy and capacity comparisons in JSON and CSV; and
- `manifest.json` with generator/config/source hashes and every output hash.

The same seed and prediction input reproduce the same scenario. Solver wall
time and a time-limited feasible schedule can vary between environments.
