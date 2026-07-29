# AeroMaintain AI

RUL prediction and synthetic maintenance scheduling with NASA C-MAPSS FD001.

AeroMaintain AI is an educational decision-support prototype built on NASA's
simulated C-MAPSS FD001 dataset. It predicts remaining useful life (RUL), then
uses the locked predictions in a capacity-constrained maintenance schedule.

> This is not an airworthiness, maintenance-approval, or production fleet
> system. FD001 is simulated, and every staffing, bay, parts, duration, demand,
> and cost assumption is synthetic.

## Project scope

The project covers one path from sensor history to a maintenance plan:

```text
Sensor history
    → leakage-resistant RUL model
    → nominal prediction interval and risk band
    → synthetic fleet and resource scenario
    → constraint-checked CP-SAT schedule
    → decision interface
```

Three rules shape the implementation:

- engines, not individual cycle rows, define model splits and validation folds;
- official test RUL is unavailable to model selection and maintenance planning;
- the application verifies the model, evaluation, optimization, and report
  hash chain before showing a result.

The Streamlit app has four pages for fleet risk, engine review, the maintenance
plan, and policy/capacity comparison.

## Reference run

The release-candidate run `m4-fd001-seed42-20260729` completed the full
prediction-to-decision pipeline. Ridge remained the champion under the
predefined selection rule.

### Prediction quality

| Metric | Official test result |
|---|---:|
| Test engines | 100 |
| MAE | 15.37 cycles |
| RMSE | 19.62 cycles |
| Motor-normalized NASA score | 625.33 |
| Nominal interval | 90% |
| Observed interval coverage | 89% |
| Critical-RUL precision | 100% |
| Critical-RUL recall | 48% |

At the fixed threshold, the model found 48% of the truly critical engines. The
empirical interval covered 89% of the official test cases against a nominal 90%
target.

### Maintenance decision

The planning experiment selected the 20 engines with the lowest interval lower
bounds and compared four policies under the same synthetic scenario.

| Policy | Scheduled | Due deferrals | Retrospective failures | Synthetic cost |
|---|---:|---:|---:|---:|
| Reactive | 0 | 20 | 20 | 10,000 |
| Fixed 90 | 13 | 7 | 18 | 10,349 |
| Predicted RUL 30 | 16 | 4 | 18 | 10,662 |
| **CP-SAT** | **17** | **3** | **13** | **8,287** |

The CP-SAT plan satisfied the modeled team, bay, parts, operating-demand, and
horizon constraints. Its status was `FEASIBLE`, not `OPTIMAL`; the solver did
not prove that no better plan exists. Failure counts are retrospective only:
true test RUL is joined after each schedule is frozen and is never available to
the planner.

Capacity sensitivity showed the expected trade-off:

| Scenario | Bays | Minimum demand | Scheduled | Due deferrals |
|---|---:|---:|---:|---:|
| Constrained | 1 | 90% | 10 | 10 |
| Base | 2 | 80% | 17 | 3 |
| Expanded | 3 | 70% | 19 | 1 |

[Read the full result report](docs/results.md).

## Application

| Fleet overview | Engine risk |
|---|---|
| [![Fleet overview](docs/screenshots/overview.png)](docs/screenshots/overview.png) | [![Engine risk](docs/screenshots/engine-health.png)](docs/screenshots/engine-health.png) |
| **Maintenance plan** | **Policy analysis** |
| [![Maintenance plan](docs/screenshots/maintenance-schedule.png)](docs/screenshots/maintenance-schedule.png) | [![Policy analysis](docs/screenshots/policy-comparison.png)](docs/screenshots/policy-comparison.png) |

Solver status, deferred work, the synthetic-data warning, and the run ID remain
visible in the interface.

## Implementation

- Causal rolling features use only the current and earlier cycles.
- Preprocessing is fitted separately inside each engine-grouped fold.
- The champion model, preprocessing, calibration values, and hashes are locked
  before official test labels are read.
- The optimizer receives predictions and synthetic planning fields, never true
  test RUL.
- CP-SAT models team, bay, parts, operating-demand, duration, and horizon
  constraints. No-solution states return no schedule.
- The app loads one named completed run and checks its artefact hashes.

## Repository contents

The repository includes source code, configuration, tests, aggregate results,
method notes, model limitations, and four application screenshots:

- [Reference run results](docs/results.md)
- [Model card](docs/model_card.md)
- [System architecture](docs/architecture.md)
- [Optimization formulation](docs/optimization.md)
- [FD001 data card](docs/data_card.md)
- [Research basis](docs/research.md)

Raw NASA data, derived row-level tables, trained model binaries, and full run
directories are excluded. They are generated locally; NASA's dataset page does
not specify a redistribution license. The committed reports contain aggregate
results only.

## Limitations

- FD001 represents one simulated operating condition and one simulated fault
  mode; it is not operational fleet telemetry.
- The nominal empirical interval is not a safety guarantee.
- Critical-RUL recall is 48% on the fixed official test set.
- All maintenance resources, durations, demand, and costs are synthetic.
- `cost_units` are dimensionless comparison values, not currency.
- The release schedule is feasible but not proven optimal.
- Results do not establish performance under real distribution shift.

## Run locally

With Python `>=3.11,<3.12`, install the package and launch a named run:

```powershell
python -m pip install -e .
aeromaintain pipeline --run-id fd001-demo
aeromaintain app --run-id fd001-demo
```

The pipeline verifies and prepares FD001, trains and locks the model, performs
official evaluation, builds the synthetic optimization scenario, solves the
maintenance policies, and writes the final report. Existing run directories
are never overwritten.

## License

Project code and documentation are licensed under the MIT License. This does
not grant rights to redistribute NASA source data.
