# AeroMaintain AI Agent Guide

## Scope and sources of truth

- This file applies to the whole repository.
- Read `PROJECT_STATE.md`, then `PROJECT_PLAN.md`, then the relevant repo skill
  before changing files.
- Treat `PROJECT_PLAN.md` as the product and experiment contract.
- Treat `PROJECT_STATE.md` as status and evidence only; it must not silently
  redefine the plan.
- `README.md` is the public-facing summary. Resolve conflicts in favor of
  `PROJECT_PLAN.md` and update README when the affected milestone requires it.

## Skill routing

- Milestone selection, phase completion, status reporting, release checks:
  `$aeromaintain-phase-gates`
- NASA FD001 download, parsing, quality checks, RUL targets, engine splits:
  `$aeromaintain-data-pipeline`
- Causal features, cross-validation, Ridge/XGBoost, intervals, explanations,
  model lock and official evaluation: `$aeromaintain-rul-modeling`
- Synthetic maintenance scenarios, baseline policies, CP-SAT scheduling and
  policy comparison: `$aeromaintain-maintenance-optimization`
- For a task spanning domains, use phase gates first and then the minimum
  relevant domain skills.

## Working rules

- Work on the earliest incomplete milestone unless the user names another one.
- Do not skip a phase gate. A file existing is not evidence that a milestone is
  complete.
- Before implementation, state the milestone, expected deliverables and checks.
- After implementation, run the focused tests, Ruff on touched Python files and
  `git diff --check` when a Git repository exists.
- Record commands and concise evidence in `PROJECT_STATE.md`. Mark a milestone
  complete only when every acceptance criterion has passed.
- Do not overwrite an existing run directory. Runs must retain config, seed,
  data/model hashes and metrics in `manifest.json`.
- Keep notebooks exploratory. Reusable data, metric, model and solver logic
  belongs under `src/aeromaintain/`.
- Prefer Python 3.11 and PowerShell-compatible commands on this Windows
  workspace.
- Do not add a dependency unless it is required by `PROJECT_PLAN.md` or removes
  more complexity than it introduces.

## Scientific and safety invariants

- Never redistribute raw NASA data in Git or release artefacts.
- Split and cross-validate by engine, never by individual cycle rows.
- Fit preprocessing and constant-column decisions only on the active training
  partition.
- Do not load official test RUL labels before the model lock is complete.
- Do not tune models or thresholds after viewing official test results.
- Never pass true test RUL to scenario generation or optimization.
- Mark every non-NASA operational and cost field as synthetic.
- Describe prediction intervals as nominal empirical intervals, not safety
  guarantees.
- Do not make airworthiness, production-readiness or real-fleet claims.

## Code review rules

- Flag any row-wise split or feature computation that can cross engine or time
  boundaries.
- Flag any path by which official test labels influence model selection,
  calibration, scenario generation or optimization.
- Flag schedules that exceed team, bay, parts, operating-demand or horizon
  constraints.
- Flag solver failures that are converted into a plausible-looking schedule.
- Flag undocumented changes to fixed assumptions in `PROJECT_PLAN.md`.
