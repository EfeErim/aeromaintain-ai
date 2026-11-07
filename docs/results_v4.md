# Protocol v4 results

The capped v3 specification is retained: every uncapped candidate improves long-RUL error but fails the predeclared critical-miss guardrail. This is a completed experiment with a negative model-adoption result.

[Frozen rules and dated amendments](protocol_v4.md) · [Inspectable evidence, paired intervals, band counts and engine replay records](protocol_v4_evidence.json)

## C-MAPSS development comparison

C-MAPSS development-only full-range comparison. Official test labels were not used in v4. These training engines and v3 hyperparameters were previously used, so this is not an independent clean C-MAPSS test.

The five candidates share engine folds and 200 uniform cuts per engine, with uncapped truth and no RUL window. Selection has 156 FD002 and 149 FD004 engines; calibration/evaluation have 52/52 and 50/50 respectively. NASA is the mean penalty per engine, not the total over engines. Miss rate means a sampled true-RUL <=30 cycle has prediction >30. Miss counts below are average engines per draw, not unique engines over all draws.

| Dataset | Candidate | RMSE | MAE | NASA/engine | >150 RMSE | >150 bias | Critical miss rate | Misses/draw |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FD002 | reference | 41.21 | 27.14 | 19766.95 | 79.27 | -71.30 | 16.4% | 4.17 |
| FD002 | ridge_normalized | 36.45 | 27.51 | 4198.46 | 54.60 | -38.38 | 35.9% | 9.15 |
| FD002 | ridge_raw | 36.49 | 27.67 | 2531.85 | 54.45 | -39.09 | 36.7% | 9.34 |
| FD002 | xgboost_normalized | 33.35 | 23.49 | 951.44 | 52.10 | -32.75 | 21.4% | 5.45 |
| FD002 | xgboost_raw | 34.22 | 24.80 | 676.46 | 51.82 | -31.88 | 26.3% | 6.69 |
| FD004 | reference | 65.06 | 41.31 | 272899533.85 | 110.33 | -96.22 | 15.8% | 3.32 |
| FD004 | ridge_normalized | 48.61 | 36.82 | 73451.29 | 64.25 | -36.82 | 34.0% | 7.15 |
| FD004 | ridge_raw | 48.42 | 36.87 | 106444.07 | 63.73 | -36.40 | 39.8% | 8.37 |
| FD004 | xgboost_normalized | 44.63 | 31.54 | 125283.62 | 64.26 | -25.31 | 24.1% | 5.06 |
| FD004 | xgboost_raw | 47.14 | 34.22 | 79567.26 | 66.24 | -30.94 | 36.4% | 7.67 |

Acceptance required improvement in >150 RMSE, no deterioration in overall RMSE or NASA, and no more than +2 percentage points in critical miss rate on **both** subsets. All four alternatives fail the last gate on both subsets. No threshold was relaxed after observing results.

Read the NASA column as tail sensitivity, not typical engine cost. Its exponential penalty on uncapped truth is dominated by a few long-lived engines the capped reference underpredicts by hundreds of cycles. One engine contributes 64% of the FD002 reference mean and 93% of FD004; the reference's median engine penalties are 21 and 186. The NASA gate therefore says little about most engines, though it did not decide this outcome.

Paired bootstrap resamples whole engine columns (2000 resamples), retaining all cuts of each engine. Candidate-minus-reference 95% intervals for normalized uncapped XGBoost:

| Dataset | RMSE difference | >150 RMSE difference | Critical miss-rate difference |
|---|---:|---:|---:|
| FD002 | -10.07 to -5.61 | -29.94 to -24.81 | +3.6% to +6.6% |
| FD004 | -25.26 to -15.32 | -52.46 to -39.79 | +5.8% to +10.7% |

## Intervals and persistent alarms

Refitted reference point models use only selection engines. Calibration uses separate engines and all RULs. Evaluation engines are excluded from v4 selection and calibration. Intervals have nominal 90% coverage. The split-conformal rule was preferred in advance for engine-level calibration sampling; that preference does not imply better observed coverage. Mondrian has better observed coverage and interval score here.

Alarms confirm after three consecutive cycles at or below 30. A false event confirms with true RUL >30; uninterrupted runs merge into one event. Lead time is measured at confirmation. The reported median excludes engines with no alarm, whose count is retained in the JSON. A point alarm carries the point prediction, while lower-bound alarms trade earlier warning for more false events. Missed engines here is a replay measure: no confirmed alert at any cycle with true RUL <=30. It is more lenient than the sampled-cut critical miss rate above, so zero missed engines does not mean every critical cycle was flagged.

| Dataset | Interval | Alarm | Coverage | Width | Interval score | Median lead | Missed engines | False events/engine |
|---|---|---|---:|---:|---:|---:|---:|---:|
| FD002 | mondrian | point | 91.7% | 120.8 | 139.3 | 24.5 | 0/52 | 0.46 |
| FD002 | mondrian | lower bound | 91.7% | 120.8 | 139.3 | 49.5 | 0/52 | 2.19 |
| FD002 | split_conformal | point | 88.1% | 115.2 | 181.4 | 24.5 | 0/52 | 0.46 |
| FD002 | split_conformal | lower bound | 88.1% | 115.2 | 181.4 | 84.5 | 0/52 | 2.08 |
| FD004 | mondrian | point | 92.4% | 163.2 | 180.3 | 24.5 | 0/50 | 0.34 |
| FD004 | mondrian | lower bound | 92.4% | 163.2 | 180.3 | 48.0 | 0/50 | 1.54 |
| FD004 | split_conformal | point | 88.6% | 167.5 | 276.4 | 24.5 | 0/50 | 0.34 |
| FD004 | split_conformal | lower bound | 88.6% | 167.5 | 276.4 | 177.0 | 0/50 | 3.50 |

Overall coverage hides how each method distributes its misses. Split conformal uses one width for every RUL, so it over-covers short RUL and under-covers long RUL. Mondrian stays near nominal in each band, but pays with very wide long-RUL intervals because the capped point model still underpredicts there.

| Dataset | Interval | True RUL | Engines | Coverage | Width |
|---|---|---|---:|---:|---:|
| FD002 | mondrian | <=30 | 52 | 94.6% | 24.9 |
| FD002 | mondrian | 31-150 | 52 | 90.2% | 124.1 |
| FD002 | mondrian | >150 | 48 | 93.7% | 179.6 |
| FD002 | split_conformal | <=30 | 52 | 100.0% | 80.5 |
| FD002 | split_conformal | 31-150 | 52 | 99.1% | 120.9 |
| FD002 | split_conformal | >150 | 48 | 50.6% | 124.7 |
| FD004 | mondrian | <=30 | 50 | 92.2% | 27.0 |
| FD004 | mondrian | 31-150 | 50 | 91.0% | 155.2 |
| FD004 | mondrian | >150 | 47 | 95.0% | 239.9 |
| FD004 | split_conformal | <=30 | 50 | 100.0% | 112.1 |
| FD004 | split_conformal | 31-150 | 50 | 100.0% | 172.5 |
| FD004 | split_conformal | >150 | 47 | 63.6% | 184.4 |

The JSON reports lower/upper misses, coverage, width and interval score for every true-RUL band and operating condition. It flags fewer than 20 distinct engines; 200 cuts from an engine are not 200 independent observations. Neither interval supports a safety guarantee. The public demo remains on its labeled v3 bundle.

## Independent N-CMAPSS simulation evidence

Measured sensors and operating settings only: 72 cycle-level mean/std/min/max features, Ridge alpha=1, engine-equal weights and fit-only preprocessing. No virtual sensors, simulator health variables, labels, unit IDs or cycle age enter the model. DS02 is a six-fit/three-test-engine descriptive pilot. The confirmatory mixture uses DS03-DS07, with four test and three calibration engines in each subset; remaining engines fit. Source file namespaces prevent unit-ID collisions. This is not cross-regime or real-fleet transfer.

| Experiment | Test engines | Ridge RMSE | Mean baseline RMSE | Critical miss rate | Coverage | Width |
|---|---:|---:|---:|---:|---:|---:|
| DS02 | 3 | 11.55 | 19.42 | 56.1% | Not calibrated | — |
| DS03-DS07 | 20 | 15.51 | 23.56 | 48.3% | 92.8% | 52.9 |

Confirmatory 95% engine-bootstrap intervals: RMSE 13.49 to 17.71; coverage 87.8% to 96.5%. Twenty evaluation engines, including only four per subset, still give limited precision. The maximum of 15 calibration residuals defines the 90% interval.

Predeclared gates: `coverage_at_least_85_percent`: **PASS**; `rmse_not_worse_than_mean`: **PASS**. These gates did not include critical misses: Ridge still predicts above 30 on about half of sampled true-RUL <=30 cycles, so passing them shows better-than-baseline error and empirical coverage, not useful critical-engine detection.

Before evaluation labels were read, protocol/model/data hashes and engine partitions were published locally in [DS02 lock](ncmapss_locks/ds02.json) and [confirmatory lock](ncmapss_locks/confirmatory.json), then claimed in the atomic interprocess ledger. This is a local preregistration record, not an independently timestamped public preregistration.

The DS08 plan was superseded before fitting or reading N-CMAPSS labels: the downloaded archive did not match the 54-unit assumption and DS08d was truncated according to its HDF5 header. It was not repaired or used. [NASA source and download](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) · [Data paper](https://doi.org/10.3390/data6010005).

## Reproduction and provenance

Use Python 3.11 and the repository environment. Raw data, models and per-cycle predictions remain Git-ignored; aggregate evidence and locks are published in docs. Hashes identify every source HDF5 file, model, configuration and protocol snapshot.

```powershell
python -m pip install -e ".[dev,ncmapss]"
python -m aeromaintain.models.v4 --project-root . --output runs/v4
python -m aeromaintain.models.v4 --project-root . --output runs/v4-repro
python scripts/fetch_ncmapss.py --subsets DS02 DS03 DS04 DS05 DS06 DS07
$raw = 'data/raw/ncmapss/N-CMAPSS'
python -m aeromaintain.data.ncmapss ${raw}_DS02-006.h5 --output runs/ncmapss-ds02
python -m aeromaintain.data.ncmapss ${raw}_DS03-012.h5 ${raw}_DS04.h5 ${raw}_DS05.h5 ${raw}_DS06.h5 ${raw}_DS07.h5 --confirmatory --output runs/ncmapss-confirmatory
python scripts/publish_v4_evidence.py
ruff check .
ruff format --check .
pytest --cov=src/aeromaintain --cov-report=term-missing --cov-fail-under=80
```

**Source changes during the runs.** `models/v4.py` was edited while the recorded C-MAPSS run was in progress, so that run's source hash (`c3654c61aba0`) differs from the published file (`e339b17845e9`), and the earlier revision was not kept. A full rerun of the published code in a new directory matched all 2797 recorded selection and evaluation values exactly, including tree counts and interval margins. Only `model.joblib` bytes and the wording of a reason string differ, so the model hashes in the two runs are not interchangeable. `data/ncmapss.py` was also edited 12 seconds after the DS02 pilot was evaluated and before the confirmatory lock; `evaluation/ledger.py` also changed between those runs. DS02 cannot be rerun without reopening a claimed holdout, so its pilot numbers have no equivalent check; it is descriptive only. The confirmatory run used the published `data/ncmapss.py` by file timestamp, which is not a cryptographic record.

Run directories are exclusive; a second model cannot reclaim an already opened holdout. Reproduction on a new workspace is a previously-seen benchmark, not another clean test. Inspect existing stored predictions and evidence without reopening labels. C-MAPSS began under [the initial protocol](protocol_v4_initial.md); interval locks reference [amendment 1](protocol_v4_amendment1.md). P0/P1 rules did not change. The historical amendment-1 assertion about DS08 was corrected by amendment 2.
