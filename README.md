# AeroMaintain AI

[![CI](https://github.com/EfeErim/aeromaintain-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/EfeErim/aeromaintain-ai/actions/workflows/ci.yml)
[![Open demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aeromaintain-ai-demo.streamlit.app/)

AeroMaintain AI is a reproducible remaining-useful-life (RUL) evaluation project on
NASA's simulated turbofan benchmarks, C-MAPSS and N-CMAPSS. It turns engine sensor
histories into per-engine RUL estimates, empirical prediction intervals, risk alarms and
model explanations. Its focus is honest evaluation: every protocol was frozen before the
labels it was tested on were opened, and each test set was opened once.

**Status: complete (2026-11-08, v1.0.0).** The final clean test has been run and
published, and no further experiments are planned. Open ideas are listed under
[Future work](#future-work).

> The data is simulated. AeroMaintain AI is an educational portfolio project, not an
> airworthiness, maintenance-approval, or production-fleet system.

## What was shown

Four clean tests, each with its protocol and model lock published before the test labels
were read, and each test set opened exactly once:

| Clean test | Data | Engines | RMSE (95% CI) | Mean baseline RMSE | Interval coverage, nominal 90% |
|---|---|---:|---:|---:|---:|
| Protocol v2 | C-MAPSS FD003, 1 condition | 100 | **15.07** (12.61–17.50) | – | 92% (86–97%) |
| Protocol v3 | C-MAPSS FD002, 6 conditions | 259 | **27.36** (23.77–30.82) | – | 83% (78–87%) |
| Protocol v3 | C-MAPSS FD004, 6 conditions | 248 | **27.40** (24.53–30.23) | – | 83% (78–87%) |
| Protocol v5 | N-CMAPSS DS01, DS08a, DS08c | 35 | **13.06** (8.92–13.15) | 22.70 | 92.5% (88.0–98.0%) |

- **One operating condition held up.** On FD003 the frozen v2 protocol met its nominal
  interval coverage, and the lower-bound alert rule caught all 20 critical engines.
- **The final test passed every predeclared gate.** Protocol v5 trained on 64 N-CMAPSS
  engines from DS02–DS07 and tested once on 35 engines from other simulation subsets.
  It beat the mean baseline, kept coverage above 85%, and had a confirmed alarm on 32 of
  35 engines while 10–30 cycles remained. The alarm gate allowed at most 3 misses, and 3
  were missed, so it passed at the limit. See the [v5 results](docs/results_v5.md).
- **On the literature's scale, C-MAPSS accuracy is competitive.** With test truth capped
  at 125, as most papers report, the same FD002 and FD004 predictions score 14.74 and
  15.33. That rescoring is retrospective, after the labels were opened
  ([comparison](docs/results_retrospective.md)).

## What was not shown

- **The RUL cap problem is still open.** The C-MAPSS model caps training targets at 125,
  so it underpredicts engines with more life left. Those engines produce most of the
  FD002/FD004 error, and coverage there fell to 83%. Uncapped models fixed the long-life
  error but missed more critical engines (protocol v4). A hybrid rule kept both on
  development data, but it was designed after seeing v4 and no clean C-MAPSS test is
  left to confirm it, so the demo keeps the capped model.
- **Single-cycle predictions near failure are still optimistic.** On the final test, 28%
  of sampled critical cycles were predicted above 30, and coverage dropped to 76% above
  60 remaining cycles. The persistent alarm passes because three alert cycles in a row
  suffice.
- **No state-of-the-art claim.** Published deep models report similar or lower capped
  RMSE, but usually after repeated use of the official test set. Differences of about one
  cycle do not rank methods.
- **Nothing here transfers to real engines.** Both benchmarks are simulations, and the
  final test rests on 35 engines.

The full experiment sequence, with every table, is in the
[protocol history](docs/protocol_history.md).

## Method

- **Freeze first.** Each protocol fixes data roles, candidates, selection rules, interval
  method and pass/fail gates in a committed document before any result it governs is
  computed.
- **Lock, then open once.** The selected model, its intervals and thresholds are hashed
  into a lock that is published before test labels are read. A holdout ledger refuses a
  second model on the same test set. In v5 the evaluator also refuses to run from a dirty
  tree or with a lock that differs from the one committed.
- **Publish what comes out.** Failed and borderline results (v3's coverage, v4's
  rejected candidates, v5's alarm gate at its limit) are reported in the same detail as
  passes.

## Application

The Streamlit application opens one explicitly selected, hash-verified run.

- **Overview** presents test metrics, both alert rules side by side, interval
  coverage, fleet risk distribution, and run identity.
- **Engine risk** presents the ranked engine list, RUL interval, sensor history, and
  model explanation.

The [hosted demo](https://aeromaintain-ai-demo.streamlit.app/) reads a
[row-free bundle](demo/): NASA's dataset page specifies no
license, so the bundle holds aggregate metrics and model outputs only, without
source rows or true RUL.

### Overview

[![AeroMaintain overview](docs/screenshots/overview.png)](docs/screenshots/overview.png)

### Engine risk

[![AeroMaintain engine risk](docs/screenshots/engine-health.png)](docs/screenshots/engine-health.png)

## Evaluation workflow (C-MAPSS)

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
with `--archive PATH`. `--dataset` selects FD001 through FD004. Opening the labels of
FD002, FD003, or FD004 with a new lock is refused once your local ledger has recorded a
run for that subset.

Run the repository quality gate with:

```powershell
python -m pip install -c constraints/python311-tested.txt -e ".[dev,ncmapss]"
ruff check .
ruff format --check .
pytest --cov=src/aeromaintain --cov-fail-under=80
```

## Documentation

- [Protocol history](docs/protocol_history.md): every experiment and its tables
- Protocols: [v2](docs/protocol_v2.md), [v3](docs/protocol_v3.md),
  [v4](docs/protocol_v4.md), [v5](docs/protocol_v5.md)
- Results: [v2 and v3](docs/results.md), [v4](docs/results_v4.md),
  [retrospective literature comparison](docs/results_retrospective.md),
  [v5](docs/results_v5.md)
- Evidence: [reference](docs/reference_evidence.json),
  [v3 development](docs/protocol_v3_development.json),
  [v4](docs/protocol_v4_evidence.json), [v5](docs/protocol_v5_evidence.json),
  [N-CMAPSS locks](docs/ncmapss_locks/)
- [RUL model card](docs/model_card.md) and [data card](docs/data_card.md)
- [Architecture and trust boundaries](docs/architecture.md)
- [Research basis](docs/research.md)
- [Changelog](CHANGELOG.md)

## Future work

These ideas are recorded and deliberately not implemented in this project:

- **Asymmetric conformal intervals** (for example conformalized quantile regression with
  a tighter lower tail), aimed at optimistic predictions near failure.
- **Deep sequence models** (CNN, LSTM, transformer). Without an unopened C-MAPSS test,
  any gain could not be confirmed independently.
- **The CruiseBench N-CMAPSS protocol**, which uses cruise-phase windows instead of cycle
  summaries.
- **A clean test of the capped/uncapped hybrid rule** on data that was never used to
  design it.
- **Real fleet data**, which would need a new data agreement, new validation, and
  different safety claims.

## Limitations

- The data is simulated; it is not operational fleet telemetry.
- The C-MAPSS RUL cap limits predictions to about 127 cycles, so long-life engines are
  underpredicted, and on FD002 and FD004 their intervals usually miss.
- Interval coverage on FD002 and FD004 fell short of its nominal 90%.
- The final N-CMAPSS test has 35 engines; per-file results rest on 10–15 engines, and
  the alarm gate passed with no margin.
- Critical-engine figures rest on 20–61 engines, so their intervals are wide.
- FD001 and FD003 development RMSE and interval-width targets were missed.
- The C-MAPSS eligibility window of 150 was an analysis choice that hid the cap's cost
  during development.
- The nominal empirical interval is not a safety guarantee.

## License

Project code and documentation are licensed under the MIT License. This does not
grant rights to redistribute NASA source data.
