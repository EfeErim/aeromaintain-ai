# AeroMaintain AI

AeroMaintain AI is a local educational prototype that predicts remaining useful
life (RUL) from NASA C-MAPSS FD001 sensor histories and converts those
predictions into a synthetic, capacity-constrained maintenance schedule.

The project is designed to demonstrate a careful prediction-to-decision
workflow:

1. leakage-resistant RUL modeling;
2. nominal empirical uncertainty and model explanation; and
3. maintenance scheduling under team, bay, parts, and operating-demand
   constraints.

It is not an airworthiness, maintenance-approval, or production fleet system.

## Current delivery

Phases 0, 1, and 2 are complete. The Python 3.11 foundation, governance
documentation, deterministic FD001 preparation pipeline, leakage-safe RUL
model selection, nominal empirical interval, model lock, and locked official
evaluation are in place. Later milestones implement CP-SAT optimization and
the Streamlit application.

The detailed product and experiment contract is
[`PROJECT_PLAN.md`](PROJECT_PLAN.md). Evidence-based milestone status is in
[`PROJECT_STATE.md`](PROJECT_STATE.md).

## Data boundary

V1 uses only NASA's simulated C-MAPSS `FD001` subset. It is not operational
fleet telemetry. The NASA dataset page currently does not specify a
redistribution license, so raw archives and extracted files are never committed
or attached to releases.

- [Research basis and method sources](docs/research.md)
- [FD001 data card and governance rules](docs/data_card.md)

All maintenance duration, staffing, bay, part, demand, and cost fields are
synthetic. Costs are reported as `cost_units`, not real currency.

## Requirements and installation

- Python `>=3.11,<3.12`
- Git

PowerShell:

```powershell
Set-Location 'D:\kişisel projeler\02-aeromaintain-ai'
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Check the environment:

```powershell
aeromaintain doctor
```

The same CLI can be invoked without the console-script wrapper:

```powershell
python -m aeromaintain doctor
```

## Prepare FD001

Download the frozen NASA archive, verify its size and SHA-256, safely select
only the FD001 members, and generate the validated tables and reports:

```powershell
aeromaintain prepare
```

An already downloaded archive can be supplied explicitly:

```powershell
aeromaintain prepare --archive 'C:\path\to\CMAPSSData.zip'
```

The command fails closed on a changed archive, unsafe ZIP path, invalid data
contract, or conflicting existing output. A repeat run with the same inputs
verifies and reuses byte-identical files under `data/processed/fd001/`.
Generated outputs include:

- train and test Parquet tables;
- uncapped and capped train RUL targets;
- the deterministic development/calibration split manifest;
- a data-quality report and development-only EDA report; and
- official test labels isolated under `evaluation/` for later locked
  evaluation.

The NASA archive, extracted members, processed tables, and reports remain
Git-ignored local data.

## Train, lock, and evaluate RUL

Create a new immutable modeling run. The command uses only development engines
for five engine-grouped folds, compares the development target mean, four Ridge
candidates, and 12 bounded XGBoost candidates, then calibrates one score per
calibration engine:

```powershell
aeromaintain train --run-id m2-fd001-seed42
```

The run records the 403 causal feature names and order, fold-local preprocessing
decisions, common metrics, XGBoost early-stopping results, automatic champion
decision, nominal 90% empirical interval, model-behavior explanation, trusted
local model hash, and `model_lock.json`. XGBoost becomes champion only when it
improves development RMSE over Ridge by at least 5% without worsening the
motor-normalized NASA score.

Official test labels are read only by the evaluation command, after the lock
and every referenced data, config, feature, and model hash have been verified:

```powershell
aeromaintain evaluate --run-id m2-fd001-seed42
```

The evaluation writes per-engine predictions, interval and risk bands, official
MAE/RMSE/NASA metrics, critical-RUL precision/recall/F1, interval coverage, and
visible error analysis under the run's `official_test/` directory. Repeating
evaluation with the same valid lock reuses only byte-identical outputs. The
interval is empirical and nominal; it is not a safety guarantee.

## Quality checks

```powershell
ruff check .
ruff format --check .
pytest
pytest --cov=src/aeromaintain --cov-report=term-missing --cov-fail-under=80
```

GitHub Actions runs the same checks on Ubuntu with Python 3.11. CI uses
synthetic smoke fixtures and does not download NASA data.

## Repository structure

```text
src/aeromaintain/
  data/          # acquisition, validation, parsing, and split logic
  features/      # causal rolling features
  models/        # baselines, XGBoost, uncertainty, and model lock
  optimization/  # scenarios, policies, and CP-SAT scheduling
  evaluation/    # prediction and retrospective decision metrics
  app/           # Streamlit application
  cli.py
  config.py
configs/
docs/
notebooks/       # exploration and visualization only
tests/
data/raw/        # local only; Git ignored
data/processed/  # local only; Git ignored
artifacts/       # local only; Git ignored
runs/            # local only; Git ignored
```

Reusable data, metric, model, and solver logic belongs in
`src/aeromaintain/`. Notebooks remain exploratory.

## Planned CLI contract

```text
aeromaintain prepare
aeromaintain train
aeromaintain evaluate
aeromaintain optimize
aeromaintain pipeline
aeromaintain app --run-id ID
aeromaintain doctor
```

`doctor`, `prepare`, `train`, and `evaluate` are implemented through Phase 2.
Later commands are delivered at their corresponding gated milestones. Commands
will not overwrite existing run directories; completed runs retain config,
seed, data/model hashes, and metrics in `manifest.json`.

## License

Project code and documentation are licensed under the MIT License. This does
not grant rights to redistribute NASA source data.
