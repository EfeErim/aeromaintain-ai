# AeroMaintain AI

AeroMaintain AI is a local educational prototype that predicts remaining useful
life from NASA C-MAPSS FD001 and converts predictions into a synthetic,
capacity-constrained maintenance schedule.

It is not an airworthiness, maintenance-approval, or production fleet system.

## Current delivery

Phases 0 and 1 are complete. The foundation, governance, quality infrastructure,
and deterministic FD001 preparation pipeline are in place. Later milestones
implement RUL modeling, optimization, and the Streamlit application.

The product and experiment contract is in `PROJECT_PLAN.md`; evidence-based
status is in `PROJECT_STATE.md`.

## Data boundary

FD001 is simulated NASA data, not operational fleet telemetry. Because the
dataset page does not specify a redistribution license, raw archives and
extracted members are excluded from Git and releases.

## Installation

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
aeromaintain doctor
```

## Prepare FD001

Download and hash-verify the NASA archive, safely select only FD001 members,
validate the fixed schema, create RUL targets and engine splits, and generate
development-only EDA:

```powershell
aeromaintain prepare
```

An existing verified archive can be supplied with `--archive`. Repeated runs
reuse only byte-identical raw and processed outputs. Official test labels are
isolated under `evaluation/` for later locked evaluation.

## Quality checks

```powershell
ruff check .
ruff format --check .
pytest
pytest --cov=src/aeromaintain --cov-report=term-missing --cov-fail-under=80
```

Generated raw, processed, model, and run artefacts remain Git-ignored.

## Planned CLI

```text
aeromaintain prepare
aeromaintain train
aeromaintain evaluate
aeromaintain optimize
aeromaintain pipeline
aeromaintain app --run-id ID
aeromaintain doctor
```

`doctor` and `prepare` are implemented through Phase 1.

## License

Project code and documentation are licensed under the MIT License. This does
not grant rights to redistribute NASA source data.
