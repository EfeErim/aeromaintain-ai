# FD001 Data Contract

Checked: 2026-07-29

## Authoritative sources

- NASA dataset page:
  https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- NASA legacy ZIP:
  https://data.nasa.gov/docs/legacy/CMAPSSData.zip
- NASA DASHlink description and citation:
  https://c3.ndc.nasa.gov/dashlink/resources/139/

The dataset is simulated run-to-failure turbofan data. Do not describe it as
measurements from an operational fleet. The NASA Open Data entry does not state
a redistribution license, so keep raw files outside Git and releases.

## Frozen acquisition values

- ZIP SHA-256:
  `74BEF434A34DB25C7BF72E668EA4CD52AFE5F2CF8E44367C55A82BFD91A5A34F`
- FD001 train: 20,631 rows and 100 engines
- FD001 test: 13,096 rows and 100 engines
- FD001 test labels: 100 values
- Schema width: 26 columns

## Column order

1. `unit_id`
2. `cycle`
3. `setting_1` through `setting_3`
4. `sensor_1` through `sensor_21`

Use integer types for `unit_id` and `cycle`; use floating-point types for
settings and sensors.

## Split contract

- Compute each train engine's maximum cycle.
- Stratify engines by lifetime quartile.
- Assign 80% of engines to development and 20% to calibration with seed 42.
- Persist engine IDs and the split algorithm version.
- Never split individual rows across roles.
