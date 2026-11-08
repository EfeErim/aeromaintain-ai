# C-MAPSS and N-CMAPSS Data Card

Checked: 2026-11-05. The project uses subsets FD001 through FD004.

## Dataset identity

- Name: C-MAPSS Jet Engine Simulated Data, subsets FD001 through FD004
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

NASA characterizes FD003 as one operating condition and two degradation modes
(high-pressure compressor and fan), with 100 training and 100 test engines. The pipeline
requires 24,720 training rows, 16,596 test rows, and 100 official test RUL labels.

FD002 and FD004 add six operating conditions, with one and two degradation modes
respectively. Counts below come from the verified archive.

| Subset | Train engines | Train rows | Test engines | Test rows | Largest test RUL |
|---|---:|---:|---:|---:|---:|
| FD001 | 100 | 20,631 | 100 | 13,096 | 145 |
| FD002 | 260 | 53,759 | 259 | 33,991 | 194 |
| FD003 | 100 | 24,720 | 100 | 16,596 | 145 |
| FD004 | 249 | 61,249 | 248 | 41,214 | 195 |

NASA's dataset page lists FD004 as 248 train and 249 test trajectories, the reverse of
the archive. The pipeline follows the archive and fails closed unless the FD004 label
file holds 248 rows.

FD001 test labels were seen before protocol v2 was designed. FD003 test labels opened
once, by `fd003-v2-clean`, after that protocol was frozen. FD002 and FD004 test labels
opened once each, by `fd002-v3-clean` and `fd004-v3-clean`, after protocol v3 and their
locks were published.

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
- claims that results generalize to other conditions, fault modes, engine types,
  or operational fleets; and
- maintenance resource, duration, parts, capacity, or cost decisions.

## Acquisition and integrity policy

M1 downloads only from the configured NASA URL or accepts an explicitly
provided local archive. Before extraction it verifies both the full archive
size and SHA-256. Extraction materializes only the selected subset's training,
test, and RUL members and rejects path traversal, symbolic links, duplicate source
members, and changed existing raw files.

Raw inputs live under `data/raw/`; generated tables live under
`data/processed/`; models, reports, and run outputs live under `artifacts/` or
`runs/`. These paths are excluded from Git.

Processed outputs live under `data/processed/<subset>/`. The train and test
tables, split manifest, data-quality report, development-only EDA summary and
HTML trend report are content-hashed in `manifest.json`. Official test labels
are stored separately at `evaluation/test_rul.parquet` and are not joined to
the test feature table.

## License and redistribution

The NASA Open Data page displayed `License not specified` and access level `public`
when rechecked on 2026-11-04. Therefore, the project does not grant redistribution
rights for the source archive or extracted NASA members. Raw NASA data must not be
committed to Git, attached to releases, or copied into test fixtures. Users acquire the
archive directly from NASA and retain responsibility for confirming applicable terms.

The hosted demo's `demo/` bundle holds aggregate metrics, model explanations, and
model outputs per engine. It holds no sensor rows, settings, or true RUL values. It
does list each test engine's observed cycle count, which derives from the data. This
is a project judgment and not legal advice.

The repository's MIT license applies to project code and documentation only; it
does not relicense NASA data.

## Split and label isolation

- All splits and folds operate on whole engines, never individual cycle rows.
- Seed `42` and engine-lifetime quartiles define the reproducible 80% development
  and 20% calibration roles.
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

The `125` cap is a modeling assumption for the early healthy period, not a physical
truth. Protocols v2 and v3 also try `130` on development engines; `fd003-v2-clean`
selected it. Official test metrics use NASA's original uncapped labels, so the cap
costs accuracy on engines with far more life left: FD002 and FD004 test RUL reaches 195.

## Privacy, sensitivity, and known limitations

The dataset is simulated and contains no personal data. Its principal risks are
scientific misuse and overclaiming rather than personal privacy.

Known limitations include one or six simulated operating conditions, one or two modeled
degradation modes, simulated sensor noise, a fixed historical benchmark, and no real
maintenance resource or cost information.

## N-CMAPSS (protocols v4 and v5)

Checked: 2026-11-08.

- **Name.** Turbofan Engine Degradation Simulation Data Set 2 (N-CMAPSS), from the NASA
  Prognostics Center of Excellence.
- **Archive.** `17. Turbofan Engine Degradation Simulation Data Set 2.zip` on NASA's PHM
  data mirror, 15,760,443,389 bytes. The URL and ETag are recorded locally in
  `data/raw/ncmapss/source.json`. Per-file SHA-256 hashes are in the published locks.
- **Reference.** Arias Chao et al., *Aircraft Engine Run-to-Failure Dataset under Real
  Flight Conditions for Prognostics and Diagnostics*, Data 6(1), 2021.
- **What it represents.** Simulated run-to-failure trajectories driven by real recorded
  flight conditions, sampled at 1 Hz within each flight cycle. It is still simulation,
  not fleet telemetry.
- **Inputs used.** Only the measured sensors `X_s` (14 channels) and the operating
  conditions `W` (4 channels), summarized per flight cycle. Virtual sensors `X_v`,
  model parameters `T`, health parameters `hs` and flight class `Fc` are never used as
  inputs. `Y` (RUL) is the target only.

### Roles

| Files | Engines | Role |
|---|---:|---|
| DS02-006 | 9 | v4 descriptive pilot; v5 training |
| DS03-012, DS04, DS05, DS06, DS07 | 55 | v4 benchmark; v5 training |
| DS01-005, DS08a-009, DS08c-008 | 35 | v5 final clean test, opened once |
| DS08d-010 | – | Excluded; the archive member is truncated |

- **Engine IDs.** IDs are namespaced by file, because equal unit numbers in different
  files are different engines.
- **Final test reservation.** The 35 final test engines were reserved in
  [`final_holdout_reservation.json`](ncmapss_locks/final_holdout_reservation.json)
  before the v5 protocol existed. Until the lock was committed, only their unit IDs
  were read. Per-engine cycle counts were not computed either, because in run-to-failure
  data they reveal engine life.
- **Distribution shift.** Per-subset failure modes and flight classes for DS01 and DS08
  were not verified from a primary source, so the v5 test's shift is defined only by
  file identity.
- **Redistribution.** As with C-MAPSS, no N-CMAPSS rows are redistributed.

