# AeroMaintain AI

[![CI](https://github.com/EfeErim/aeromaintain-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/EfeErim/aeromaintain-ai/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aeromaintain-ai-demo.streamlit.app/)

Remaining-useful-life (RUL) prediction for turbofan engines on NASA's C-MAPSS and
N-CMAPSS benchmarks, with prediction intervals, maintenance alarms and a Streamlit
review app.

Most C-MAPSS results are tuned against the official test set many times. I wanted
numbers I could defend, so every test set here was opened exactly once, after the
model and its thresholds were locked and committed.

![Overview page](docs/screenshots/overview.png)

## Results

| Test set | Engines | RMSE (95% CI) | 90% interval coverage |
|---|---:|---:|---:|
| C-MAPSS FD003 | 100 | 15.07 (12.61–17.50) | 92% |
| C-MAPSS FD002 | 259 | 27.36 (23.77–30.82) | 83% |
| C-MAPSS FD004 | 248 | 27.40 (24.53–30.23) | 83% |
| N-CMAPSS DS01, DS08a, DS08c | 35 | 13.06 (11.13–15.00), baseline 22.70 | 92.5% |

The N-CMAPSS range was corrected on 2026-11-09: the frozen v5 bootstrap resampled only
two engines. The point values and pass/fail results are unchanged
([erratum](docs/results_v5.md#erratum-bootstrap-ranges-2026-11-09)).

- On the final N-CMAPSS test the alarm fired for 32 of 35 engines while 10–30 cycles
  remained. The predeclared limit was 3 misses, so it passed with no margin.
- Scored the way most papers do (true RUL capped at 125), FD002 and FD004 come out at
  14.74 and 15.33. That rescoring was done after the labels were open, so I report it
  separately ([comparison](docs/results_retrospective.md)).
- FD002/FD004 error comes mostly from long-life engines: the training target is capped
  at 125, so the model underpredicts them. Removing the cap fixed that but missed more
  critical engines, so I kept the capped model ([details](docs/results_v4.md)).

The full sequence of experiments, including the ones that failed, is in the
[protocol history](docs/protocol_history.md).

## How it works

```text
NASA archive (size + SHA-256 checked)
  → engine-level train/validation splits, causal rolling features
  → Ridge vs. XGBoost, selected on truncated rows that look like the test set
  → empirical (conformal-style) intervals and alarm threshold, cross-fitted per engine
  → model lock: model, intervals and thresholds hashed and committed
  → test labels loaded only if the lock verifies; a ledger refuses a second lock
```

Leakage controls that matter most:

- splits and CV folds are by engine, never by cycle row;
- rolling features only look backwards;
- XGBoost early stopping uses an inner engine split, not the scored fold;
- test labels live in a separate file that the evaluator reads only after the lock
  hash validates.

## Design decisions

- **Open each test set once.** The usual practice is to report the best of many runs
  against the official test set. That makes every number optimistic by an unknown
  amount, so each protocol here is committed before its test, and a holdout ledger
  refuses a second model lock. The cost is that a bad result stays published (protocol
  v4 was a negative result, and v5 passed G3 with no margin).
- **Score development on one random cut per engine.** Scoring every cycle row would
  weight long-lived engines more and reward easy late-life rows. The test has one
  prediction per engine, so development draws 200 random cuts per engine and takes the
  median. The window of 150 cycles for those cuts is a disclosed choice.
- **Keep the RUL cap at 125.** An uncapped target fixed the underprediction of
  long-life engines on FD002/FD004 but missed more engines near failure, which is the
  costly error for maintenance ([v4 results](docs/results_v4.md)).
- **Alarm on three consecutive cycles, not one.** Single predictions near failure are
  noisy and optimistic. On the v5 test the persistent point alarm confirmed a median
  22 cycles before failure with 0.26 false events per engine. Alarming on the interval
  lower bound instead gave 55 cycles of lead but 2.57 false events per engine.
- **Empirical intervals banded by predicted RUL (Mondrian), not one global margin.**
  Errors grow with predicted life, so a single margin is too wide near failure and too
  narrow far from it. Coverage is measured on the test, not assumed.

Design notes are in [architecture](docs/architecture.md), the
[model card](docs/model_card.md) and the [data card](docs/data_card.md). Method
choices and the literature they rest on are in [research notes](docs/research.md).

## Quickstart

Requires Python 3.11.

```bash
git clone https://github.com/EfeErim/aeromaintain-ai.git
cd aeromaintain-ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -c constraints/python311-tested.txt -e .

aeromaintain pipeline --run-id fd001-demo   # downloads and verifies the NASA archive
aeromaintain app --run-id fd001-demo
```

`--dataset` selects FD001–FD004, and `--archive PATH` uses a local copy of the archive.
The FD001 pipeline takes about 11 minutes on CPU. [docs/demo.md](docs/demo.md) has the
captured output of these commands from a fresh clone. The hosted demo sleeps when idle
and takes a minute to wake.

Development checks:

```bash
pip install -c constraints/python311-tested.txt -e ".[dev,ncmapss]"
ruff check . && ruff format --check .
pytest --cov=src/aeromaintain --cov-fail-under=80
```

## Project layout

```text
src/aeromaintain/   data pipeline, features, models, evaluation, Streamlit app
configs/            frozen experiment configurations
docs/               protocols, results, model and data cards
demo/               aggregate-only bundle used by the hosted demo
tests/              unit and protocol tests
```

Protocol documents, configs and the code they ran (`models/v*.py`, `models/rul.py`,
`evaluation/`, `data/ncmapss.py`) are versioned on purpose: their SHA-256 hashes are
part of the published model locks, so they are never edited after a test is run.
`tests/test_protocol_v5.py` fails if any file pinned by the v5 lock changes. Known
defects in frozen code are corrected in the results pages instead, as with the
[v5 bootstrap erratum](docs/results_v5.md#erratum-bootstrap-ranges-2026-11-09).

## Limitations

- Both benchmarks are simulated. Nothing here says anything about real engines.
- The final test has 35 engines, and per-subset results rest on 10–15 engines.
- Coverage on FD002 and FD004 fell short of the nominal 90%.
- Single-cycle predictions close to failure are still optimistic. The alarm works
  because it needs three alert cycles in a row, not because each prediction is good.
- I make no state-of-the-art claim. Published deep models report similar or lower
  capped RMSE, usually after repeated use of the test set.

## Data and license

Code and documentation are MIT licensed. NASA's dataset page specifies no license, so
the repository and the hosted demo contain no source rows or true RUL values, only
aggregate metrics and model outputs.
