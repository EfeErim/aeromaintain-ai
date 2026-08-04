# AeroMaintain AI

[![CI](https://github.com/EfeErim/aeromaintain-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/EfeErim/aeromaintain-ai/actions/workflows/ci.yml)

AeroMaintain AI is a reproducible remaining-useful-life (RUL) evaluation project
built on NASA's C-MAPSS FD001 turbofan benchmark. It turns multivariate engine
sensor histories into per-engine RUL estimates, empirical prediction intervals,
risk rankings, and model-behavior explanations.

The repository covers the complete model-evaluation path rather than stopping at
a notebook: verified data preparation, engine-grouped validation, bounded model
comparison, calibration, an immutable model lock, isolated official-test
evaluation, and a local Streamlit review application.

> FD001 contains simulated degradation trajectories. AeroMaintain AI is an
> educational portfolio project, not an airworthiness, maintenance-approval, or
> production-fleet system.

## Application

The Streamlit application opens one explicitly selected, hash-verified run.

- **Overview** presents the locked test metrics, empirical interval coverage,
  fleet risk distribution, and run identity.
- **Engine risk** presents the ranked engine list, RUL interval, sensor history,
  and Ridge coefficient review.

### Overview

[![AeroMaintain overview](docs/screenshots/overview.png)](docs/screenshots/overview.png)

### Engine risk

[![AeroMaintain engine risk](docs/screenshots/engine-health.png)](docs/screenshots/engine-health.png)

## Evaluation workflow

```text
Verify the NASA FD001 archive
    → validate and prepare engine trajectories
    → compare Ridge and XGBoost under one grouped protocol
    → calibrate the empirical interval and lock the model
    → open official test labels only after the lock
    → review aggregate metrics and engine-level predictions
```

### Data preparation

FD001 contains 100 training engines and 100 test engines. Each cycle records
three operating settings and 21 sensor channels. The pipeline verifies the
archive size and SHA-256, safely extracts only the required FD001 members,
validates the 26-column schema, and keeps raw data outside Git.

### Leakage controls

- Splits and cross-validation operate on complete engines, never individual
  cycle rows.
- Rolling features use only the current and previous cycles.
- Preprocessing is fitted inside each active training fold.
- Twenty calibration engines remain separate from model selection.
- Official test RUL labels remain inaccessible until `model_lock.json` exists.
- Existing run directories are never overwritten, and persisted artefacts are
  connected by SHA-256 hashes.

### Model selection

The development comparison includes a target-mean baseline, Ridge regression,
and a bounded XGBoost search. XGBoost could replace Ridge only by improving
development RMSE by at least 5% without worsening the motor-normalized NASA
score.

The best XGBoost candidate improved RMSE by 1.76% and produced a worse NASA
score, so Ridge remained the locked reference model under the predefined rule.

## Reference run

The immutable run `m4-fd001-seed42-20260729` was evaluated on all 100 official
FD001 test engines.

| Metric | Result |
|---|---:|
| MAE | 15.37 cycles |
| RMSE | 19.62 cycles |
| Motor-normalized NASA score | 625.33 |
| Nominal / observed interval coverage | 90% / 89% |
| Critical precision at `prediction <= 30` | 100% |
| Critical recall at `prediction <= 30` | 48% |

The 48% recall is a material weakness: the point-prediction threshold missed 13
of the 25 truly critical test engines. The interface therefore ranks engines
using the more conservative empirical interval lower bound. The interval remains
an empirical estimate, not a safety guarantee.

See the [complete reference results](docs/results.md) and
[machine-readable aggregate evidence](docs/reference_evidence.json).

## Run locally

Python `>=3.11,<3.12` is required. From PowerShell:

```powershell
git clone https://github.com/EfeErim/aeromaintain-ai.git
cd aeromaintain-ai
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -c constraints/python311-tested.txt -e .
aeromaintain doctor
aeromaintain pipeline --run-id fd001-demo
aeromaintain app --run-id fd001-demo
```

If the NASA archive is not already present, the pipeline downloads it and
verifies its byte size and SHA-256 before extraction. A verified local archive
can be supplied with `--archive PATH`.

Run the repository quality gate with:

```powershell
python -m pip install -c constraints/python311-tested.txt -e ".[dev]"
ruff check .
ruff format --check .
pytest --cov=src/aeromaintain --cov-fail-under=80
```

## Documentation

- [Reference run results](docs/results.md)
- [Machine-readable reference evidence](docs/reference_evidence.json)
- [RUL model card](docs/model_card.md)
- [FD001 data card](docs/data_card.md)
- [Architecture and trust boundaries](docs/architecture.md)
- [Research basis](docs/research.md)

## Limitations

- FD001 represents one simulated operating condition and one simulated fault
  mode; it is not operational fleet telemetry.
- The nominal empirical interval is not a safety guarantee.
- Point-threshold critical-RUL recall is 48% on the fixed official test set.
- Results do not establish performance under real distribution shift.

## License

Project code and documentation are licensed under the MIT License. This does
not grant rights to redistribute NASA source data.
