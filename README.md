# AeroMaintain AI

AeroMaintain AI is a local educational prototype for leakage-resistant remaining
useful life prediction and synthetic maintenance scheduling with NASA C-MAPSS
FD001.

It is not an airworthiness, maintenance-approval, or production fleet system.

## Current delivery

Phase 0 is complete. The Python 3.11 project foundation, research and data
governance documents, quality tooling, CI, and environment doctor are in place.

The product and experiment contract is in `PROJECT_PLAN.md`; evidence-based
status is in `PROJECT_STATE.md`.

## Data boundary

V1 uses only NASA's simulated C-MAPSS FD001 subset. It is not operational fleet
telemetry. Raw archives and extracted files are excluded from Git and releases.
All operational, resource, and cost fields planned for later phases are
synthetic.

## Installation

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
aeromaintain doctor
```

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

Only `doctor` is implemented in Phase 0.

## License

Project code and documentation are licensed under the MIT License. This does
not grant rights to redistribute NASA source data.
