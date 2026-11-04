# AeroMaintain AI

[![CI](https://github.com/EfeErim/aeromaintain-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/EfeErim/aeromaintain-ai/actions/workflows/ci.yml)

AeroMaintain AI is a reproducible remaining-useful-life (RUL) evaluation project
on NASA's simulated C-MAPSS turbofan benchmark. It turns multivariate engine sensor
histories into per-engine RUL estimates, empirical prediction intervals, risk
rankings, and model-behavior explanations.

The main result is a clean check: with its protocol frozen, the pipeline opened
the FD003 test labels once and scored **RMSE 15.07 cycles** (95% CI 12.61–17.50)
across 100 engines.

> C-MAPSS contains simulated degradation trajectories. AeroMaintain AI is an
> educational portfolio project, not an airworthiness, maintenance-approval, or
> production-fleet system.

## Results

Both runs use the same frozen [protocol v2](docs/protocol_v2.md). Ranges are 95%
engine-level bootstrap intervals.

| Metric | FD003, clean single-open test | FD001, test previously seen |
|---|---:|---:|
| Run | `fd003-v2-clean` | `fd001-v2` |
| RMSE | **15.07** (12.61–17.50) | 15.80 (12.87–18.78) |
| MAE | 11.23 (9.34–13.28) | 11.15 (9.10–13.43) |
| Motor-normalized NASA score | 399 (240–620) | 398 (233–625) |
| Interval coverage, nominal 90% | 92% (86–97%) | 94% (89–98%) |
| Mean interval width | 53.3 (49.3–57.3) | 55.4 (51.2–59.4) |

**Read the FD001 column with care.** Its test labels were seen before v2 was
designed, so it is not independent evidence. Seed 42 is also the best of ten seeds
on FD001: across seeds, RMSE averages 16.03 (SD 0.14, range 15.80–16.21).

### Critical-engine alerts

A truly critical engine has at most 30 remaining cycles. Each threshold was chosen
on training engines to reach 95% recall with the fewest alarms. Ranges here are
Wilson 95% intervals.

| Rule | Dataset | Recall | Precision | Missed | False alarms |
|---|---|---:|---:|---:|---:|
| `prediction <= 30` | FD003 | 16/20, 80% (58–92%) | 94% (73–99%) | 4 | 1 |
| `interval_low <= 25` | FD003 | 20/20, 100% (84–100%) | 71% (53–85%) | 0 | 8 |
| `prediction <= 30` | FD001 | 20/25, 80% (61–91%) | 100% (84–100%) | 5 | 0 |
| `interval_low <= 20` | FD001 | 24/25, 96% (80–99%) | 86% (69–94%) | 1 | 4 |

The interface uses the interval rule: it misses fewer critical engines and raises
more false alarms. The interval is not a safety guarantee.

### Targets

| Target | Result | Status |
|---|---|---|
| Development RMSE at most about 16.6 | FD001 18.57, FD003 17.62 | **Missed** |
| Mean interval width below 45, with coverage near 90% | Training 55.7 / 52.0, test 55.4 / 53.3 | **Missed** |
| Risk rule at 95% development recall | FD001 96%, FD003 100% | Met |
| One clean holdout check | FD003 opened once, by `fd003-v2-clean` | Met |

Test RMSE falls below 16.6, but the target was defined on development data and that
figure stays missed. The two challengers allowed after the first result did not
help on FD001: LSTM reached 18.39 with a worse NASA score, and a bounded CQR trial
produced a 77% wider interval.

## Application

The Streamlit application opens one explicitly selected, hash-verified run.

- **Overview** presents test metrics, both alert rules side by side, interval
  coverage, fleet risk distribution, and run identity.
- **Engine risk** presents the ranked engine list, RUL interval, sensor history, and
  model explanation.

The hosted demo reads a [row-free bundle](demo/): NASA's dataset page specifies no
license, so the bundle holds aggregate metrics and model outputs only, without
source rows or true RUL.

### Overview

[![AeroMaintain overview](docs/screenshots/overview.png)](docs/screenshots/overview.png)

### Engine risk

[![AeroMaintain engine risk](docs/screenshots/engine-health.png)](docs/screenshots/engine-health.png)

## Evaluation workflow

```text
Verify the NASA archive
    → validate and prepare engine trajectories
    → compare Ridge and XGBoost on test-like truncated rows
    → cross-fit the interval and risk threshold on training engines
    → lock the model
    → open official test labels only after the lock
    → review aggregate metrics and engine-level predictions
```

### Leakage controls

- Splits and cross-validation operate on complete engines, never individual cycle
  rows.
- Rolling features use only the current and previous cycles.
- Preprocessing is fitted inside each active training partition.
- XGBoost early stopping uses an inner engine split, never the scored fold.
- Official test RUL labels are stored separately, and the evaluator loads them only
  after `model_lock.json` and its hashes validate.
- A ledger refuses to open a clean holdout for a second model lock.
- Existing run directories are never overwritten, and persisted artefacts are
  connected by SHA-256 hashes.

### Model selection

Development scoring matches the test: one random row per engine, scored against
true RUL, repeated over 200 seeded draws. XGBoost can replace Ridge only by lowering
median RMSE at least 5% without a worse median NASA score. It did so on both datasets
and won 99% of paired draws on each.

These draws use rows with true RUL of at most 150. That window was chosen after
FD001 test labels had been seen, so each run reports the alternatives: XGBoost also
wins at 130 and 175, while Ridge wins with no window because of its better NASA
score.

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

If the NASA archive is not already present, the pipeline downloads it and verifies
its byte size and SHA-256 before extraction. A verified local archive can be supplied
with `--archive PATH`. `--dataset FD003` selects FD003; opening its labels with a new
lock is refused once your local ledger has recorded a run.

Run the repository quality gate with:

```powershell
python -m pip install -c constraints/python311-tested.txt -e ".[dev]"
ruff check .
ruff format --check .
pytest --cov=src/aeromaintain --cov-fail-under=80
```

## Documentation

- [Reference run results](docs/results.md)
- [Protocol v2](docs/protocol_v2.md)
- [Machine-readable reference evidence](docs/reference_evidence.json)
- [RUL model card](docs/model_card.md)
- [C-MAPSS data card](docs/data_card.md)
- [Architecture and trust boundaries](docs/architecture.md)
- [Research basis](docs/research.md)
- [Changelog](CHANGELOG.md)

## Limitations

- The data is simulated, with one operating condition per dataset; it is not
  operational fleet telemetry.
- Critical-engine figures rest on 20–25 engines, so their intervals are wide.
- Development RMSE and interval-width targets were missed.
- The eligibility window of 150 was an analysis choice, and it changes the
  Ridge-versus-XGBoost outcome when removed.
- The nominal empirical interval is not a safety guarantee, and results do not
  establish performance under real distribution shift.

## License

Project code and documentation are licensed under the MIT License. This does not
grant rights to redistribute NASA source data.
