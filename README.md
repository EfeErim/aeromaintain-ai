# AeroMaintain AI

**From turbofan sensor history to a capacity-constrained maintenance plan.**

AeroMaintain AI is an educational decision-support prototype built on NASA's
simulated C-MAPSS FD001 dataset. It connects remaining-useful-life prediction
to maintenance scheduling, so the output is not just a model score but an
explicit plan with visible resource constraints, deferrals, and uncertainty.

> This is not an airworthiness, maintenance-approval, or production fleet
> system. FD001 is simulated, and every staffing, bay, parts, duration, demand,
> and cost assumption is synthetic.

## The problem

Predicting when an engine may need maintenance is only half of the decision.
A planner still needs to decide which engines to service, when to service them,
and what to defer when technicians, bays, parts, and operating capacity are
limited.

That creates two connected risks:

- a model can look accurate while leaking future or test information; and
- a maintenance recommendation can look plausible while violating real
  planning constraints.

## The solution

AeroMaintain AI treats prediction and planning as one verified workflow:

```text
Sensor history
    → leakage-resistant RUL model
    → nominal prediction interval and risk band
    → synthetic fleet and resource scenario
    → constraint-checked CP-SAT schedule
    → decision interface
```

The workflow is designed around three safeguards:

- engines, not individual cycle rows, define model splits and validation folds;
- official test RUL is unavailable to model selection and maintenance planning;
- the application verifies the model, evaluation, optimization, and report
  hash chain before showing a result.

The final interface answers four practical questions:

1. What is the current fleet-level risk?
2. Which engines should be reviewed first?
3. What maintenance plan fits the available resources?
4. How do alternative policies and capacity assumptions change the outcome?

## Verified results

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

The 48% critical-RUL recall is an important negative result: the point
prediction threshold missed more than half of the truly critical engines. The
empirical interval also fell one percentage point below its nominal target.

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

[Read the complete result report →](docs/results.md)

## Product views

| Fleet overview | Engine risk |
|---|---|
| [![Fleet overview](docs/screenshots/overview.png)](docs/screenshots/overview.png) | [![Engine risk](docs/screenshots/engine-health.png)](docs/screenshots/engine-health.png) |
| **Maintenance plan** | **Policy analysis** |
| [![Maintenance plan](docs/screenshots/maintenance-schedule.png)](docs/screenshots/maintenance-schedule.png) | [![Policy analysis](docs/screenshots/policy-comparison.png)](docs/screenshots/policy-comparison.png) |

The application exposes solver status, unproven optimality, deferred work,
synthetic assumptions, and run identity instead of presenting every result as
an unconditional success.

## How it works

### 1. Leakage-resistant modeling

Causal rolling features use only the current and earlier cycles. Preprocessing
is fitted inside each engine-grouped fold. A development target mean, Ridge,
and bounded XGBoost search are compared under one fixed protocol.

### 2. Model lock and independent evaluation

The champion, feature order, preprocessing decisions, calibration state, and
file hashes are frozen before official test labels are opened. The evaluator
rejects a missing or changed lock.

### 3. Prediction-to-decision boundary

The optimizer receives only engine identity, last observed cycle, predicted
RUL, interval bounds, and risk band. It rejects any scenario containing true
RUL.

### 4. Constrained scheduling

OR-Tools CP-SAT schedules maintenance across teams, bays, parts inventory,
daily operating demand, job durations, and a 30-day horizon. Solver failure or
unknown status never becomes a plausible-looking schedule.

### 5. Verified presentation

The Streamlit application opens one explicitly selected completed run and
checks the full artefact hash chain before rendering fleet risk, engine review,
maintenance planning, and policy analysis.

## Public evidence

The repository contains the source code, configuration, tests, aggregate
results, methodology, model limitations, and four verified application
screenshots:

- [Verified results](docs/results.md)
- [Model card](docs/model_card.md)
- [System architecture](docs/architecture.md)
- [Optimization formulation](docs/optimization.md)
- [FD001 data card](docs/data_card.md)
- [Research basis](docs/research.md)
- [Milestone evidence](PROJECT_STATE.md)

Raw NASA data, derived row-level tables, trained model binaries, and full run
directories are intentionally excluded. They are generated locally and may
carry redistribution-license ambiguity or large, reproducible intermediate
artefacts. The committed reports expose the aggregate outcome without
redistributing NASA records.

## Limitations

- FD001 represents one simulated operating condition and one simulated fault
  mode; it is not operational fleet telemetry.
- The nominal empirical interval is not a safety guarantee.
- Critical-RUL recall is 48% on the fixed official test set.
- All maintenance resources, durations, demand, and costs are synthetic.
- `cost_units` are dimensionless comparison values, not currency.
- The release schedule is feasible but not proven optimal.
- Results do not establish performance under real distribution shift.

<details>
<summary><strong>Run locally</strong></summary>

### Requirements

- Python `>=3.11,<3.12`
- Git

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the environment check, complete pipeline, and application:

```powershell
aeromaintain doctor
aeromaintain pipeline --run-id your-fd001-run
aeromaintain app --run-id your-fd001-run
```

The pipeline verifies and prepares FD001, trains and locks the model, performs
official evaluation, builds the synthetic optimization scenario, solves the
maintenance policies, and writes the final report. Existing run directories
are never overwritten.

Quality checks:

```powershell
ruff check .
ruff format --check .
pytest --cov=src/aeromaintain --cov-report=term-missing --cov-fail-under=80
```

</details>

## License

Project code and documentation are licensed under the MIT License. This does
not grant rights to redistribute NASA source data.
