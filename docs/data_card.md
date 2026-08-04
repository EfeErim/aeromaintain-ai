# FD001 Data Card

Checked: 2026-07-29

## Dataset identity

- Name: C-MAPSS Jet Engine Simulated Data, subset FD001
- Publisher: NASA Prognostics Center of Excellence
- Dataset page:
  <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data>
- Archive:
  <https://data.nasa.gov/docs/legacy/CMAPSSData.zip>
- Archive size: `12,425,978` bytes
- Archive SHA-256:
  `74BEF434A34DB25C7BF72E668EA4CD52AFE5F2CF8E44367C55A82BFD91A5A34F`
- Primary generation reference:
  <https://ntrs.nasa.gov/citations/20090029214>

The size and hash above were rechecked from a direct NASA download on
2026-07-29.

## What the data represents

FD001 contains simulated degradation trajectories from the C-MAPSS turbofan
engine model. It does not contain measurements collected from a deployed
aircraft fleet.

NASA characterizes FD001 as:

- one operating condition;
- one high-pressure-compressor degradation mode;
- 100 training engines;
- 100 test engines;
- 20,631 training rows;
- 13,096 test rows; and
- 100 official test RUL labels.

Each trajectory starts under normal operation with unknown initial wear and
manufacturing variation. Training trajectories continue to simulated failure.
Test trajectories stop earlier, and the separate label file records remaining
cycles after the final observed test cycle.

## Schema

Each whitespace-separated row has exactly 26 numeric columns:

1. `unit_id`
2. `cycle`
3. `setting_1` through `setting_3`
4. `sensor_1` through `sensor_21`

`unit_id` and `cycle` are integers. Settings and sensor values are floating
point. The M1 parser fails closed on missing, duplicated, non-finite,
misordered, or schema-invalid rows.

## Intended use

- educational RUL regression;
- engine-grouped cross-validation and leakage-control demonstrations;
- nominal empirical prediction-interval evaluation;
- reproducible experiment runs.

## Out-of-scope and prohibited claims

- real-aircraft maintenance approval or airworthiness decisions;
- production deployment or safety guarantees;
- claims that model explanations identify physical causes;
- claims that FD001 results generalize to other conditions, fault modes, engine
  types, or operational fleets; and
- maintenance resource, duration, parts, capacity, or cost decisions.

## Acquisition and integrity policy

M1 downloads only from the configured NASA URL or accepts an explicitly
provided local archive. Before extraction it verifies both the full archive
size and SHA-256. Extraction materializes only the FD001 training, test, and
RUL members and rejects path traversal, symbolic links, duplicate source
members, and changed existing raw files.

Raw inputs live under `data/raw/`; generated tables live under
`data/processed/`; models, reports, and run outputs live under `artifacts/` or
`runs/`. These paths are excluded from Git.

Processed FD001 outputs live under `data/processed/fd001/`. The train and test
tables, split manifest, data-quality report, development-only EDA summary and
HTML trend report are content-hashed in `manifest.json`. Official test labels
are stored separately at `evaluation/test_rul.parquet` and are not joined to
the test feature table.

## License and redistribution

The NASA Open Data page currently displays `License not specified`. Therefore,
the project does not grant redistribution rights for the source archive or
extracted NASA members. Raw NASA data must not be committed to Git, attached to
releases, or copied into test fixtures. Users acquire the archive directly from
NASA and retain responsibility for confirming applicable terms.

The repository's MIT license applies to project code and documentation only; it
does not relicense NASA data.

## Split and label isolation

- All splits and folds operate on whole engines, never individual cycle rows.
- Seed `42` and engine-lifetime quartiles define the reproducible 80-engine
  development and 20-engine calibration roles.
- Official test RUL labels are isolated from training, feature selection,
  hyperparameter search, and champion selection.
- The evaluator may read official test labels only after `model_lock.json` is
  valid.

## Target construction

For training rows:

```text
rul_true = engine_max_cycle - cycle
rul_target = min(rul_true, 125)
```

The `125` cap is a modeling assumption for the early healthy period, not a
physical truth. Official test metrics use NASA's original uncapped labels.

## Privacy, sensitivity, and known limitations

The dataset is simulated and contains no personal data. Its principal risks are
scientific misuse and overclaiming rather than personal privacy.

Known limitations include one operating condition, one modeled degradation
mode, simulated sensor noise, a fixed historical benchmark, and no real
maintenance resource or cost information. The active project does not invent
those missing operational fields or produce a maintenance schedule.
