# AeroMaintain AI Project State

Last updated: 2026-07-29

## Current position

- Active phase: Phase 4 complete
- Active milestone: None — release candidate verified
- Overall status: Phases 0 through 4 complete
- Current blocker: None

## Status rules

- Allowed states: `pending`, `in_progress`, `blocked`, `complete`.
- A milestone becomes `complete` only after all deliverables exist and all
  acceptance checks pass.
- Add evidence as commands, artefact paths or metric/report identifiers.
- Do not replace failed evidence with a narrative claim.

## Milestone register

| ID | Milestone | Status | Evidence |
|---|---|---|---|
| M0.1 | Project skeleton | complete | Clean Python 3.11 editable install; package import `0.1.0`; `doctor` 5/5 PASS; Git exclusions verified |
| M0.2 | Research and data governance | complete | `docs/research.md`; `docs/data_card.md`; live NASA archive size/hash match |
| M0.3 | Quality infrastructure | complete | Ruff clean; 4 tests pass; 83.33% coverage; Python 3.11 CI configured |
| M1.1 | Safe data acquisition | complete | NASA ZIP size/hash verified; safe FD001-only extraction; changed ZIP, traversal and conflicting raw data rejected |
| M1.2 | Parser and data contract | complete | Train `20,631` rows/100 engines; test `13,096` rows/100 engines; 26-column, key, order, null and finite checks passed |
| M1.3 | RUL target and engine split | complete | Non-negative uncapped/capped RUL; deterministic seed-42 split with 80 development/20 calibration engines; isolated test labels |
| M1.4 | EDA and quality report | complete | Development-only EDA summary and HTML trends; quality report; two prepare runs produced the same eight processed SHA-256 hashes |
| M2.1 | Causal feature generation | complete | 403 ordered causal features; future-row and train/inference-order tests pass |
| M2.2 | Baselines and grouped CV | complete | Common five-fold engine-grouped protocol; target mean and four Ridge candidates; no engine overlap |
| M2.3 | XGBoost and champion selection | complete | 12 candidates with fold early stopping; fixed rule selected Ridge; two real runs produced byte-identical decisions and model artefacts |
| M2.4 | Uncertainty, explanation and model lock | complete | 20 engine-level calibration scores; nominal 90% empirical `q=31.701395`; coefficient explanation; verified model/data/config/feature hashes |
| M2.5 | Locked official test evaluation | complete | 100 engines; MAE `15.369728`, RMSE `19.622062`, NASA `625.326953`; repeated evaluation byte-identical |
| M3.1 | Synthetic maintenance scenario | complete | Seed-42 risk-sorted 20-engine scenario is byte-reproducible; truth-free schema and synthetic dictionary verified |
| M3.2 | Baseline maintenance policies | complete | Reactive, fixed-90 and predicted-RUL-30 schedules use one scenario and common resource evaluator |
| M3.3 | CP-SAT schedule | complete | Hand fixture matches; base solver FEASIBLE with 17 scheduled, 3 due deferrals, 0 capacity shortfall; optimality unproven and visible |
| M3.4 | Policy and capacity comparison | complete | Four-policy and constrained/base/expanded JSON/CSV comparisons plus `docs/optimization.md` verified |
| M4.1 | Streamlit decision application | complete | Verified explicit-run loader; four-page fixture/browser smoke; CSV and truth-free what-if checks |
| M4.2 | End-to-end pipeline | complete | Real `m4-fd001-seed42-20260729` run is `pipeline_complete`; overwrite and partial-run tests pass |
| M4.3 | Documentation and release preparation | complete | Final README, model card, results, architecture and four verified screenshots |
| M4.4 | Final release candidate | complete | Clean Python 3.11 install/doctor/app smoke; 41 tests; 84.45% coverage; Ruff/hash/Git exclusion gates pass |

## Current milestone evidence

### Phase 0 closure — 2026-07-29

M0.1:

- `git init -b main` — repository initialized on `main`.
- `py -3.11 -m venv .venv` — clean Python 3.11 environment created.
- `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"` — editable package
  and all declared runtime/development dependencies installed successfully.
- `.\.venv\Scripts\python.exe -m pip check` — `No broken requirements found.`
- `.\.venv\Scripts\python.exe -c "import aeromaintain; ..."` — imported
  package version `0.1.0`.
- `.\.venv\Scripts\aeromaintain.exe doctor` — Python `3.11.9`; package,
  configs, local directories and writable project root all PASS (`5/5`).
- `.\.venv\Scripts\python.exe -m aeromaintain doctor` — module entry point
  also PASS (`5/5`).
- `git check-ignore -v data/raw/probe.txt data/processed/probe.parquet
  artifacts/model.json runs/example/manifest.json` — all four generated-data
  paths matched `.gitignore`.
- `git ls-files data/raw data/processed artifacts runs` — no tracked raw,
  processed, artifact or run files.

M0.2:

- `docs/research.md` records the FD001 scope decision and primary/authoritative
  sources for C-MAPSS, Ridge, XGBoost, SHAP, nominal empirical intervals and
  CP-SAT.
- `docs/data_card.md` records simulation status, schema, intended/prohibited
  uses, label isolation and the no-redistribution policy.
- Direct NASA archive check:
  `Bytes=12425978`,
  `SHA256=74BEF434A34DB25C7BF72E668EA4CD52AFE5F2CF8E44367C55A82BFD91A5A34F`.
- NASA Open Data currently reports `License not specified`; raw NASA files are
  excluded from Git and releases.

M0.3:

- `.\.venv\Scripts\ruff.exe check .` — all checks passed.
- `.\.venv\Scripts\ruff.exe format --check .` — all files formatted.
- `.\.venv\Scripts\pytest.exe -q` — `4 passed`.
- `.\.venv\Scripts\pytest.exe --cov=src/aeromaintain
  --cov-report=term-missing --cov-fail-under=80` — `4 passed`, total coverage
  `83.33%`, required threshold reached.
- YAML smoke parse covered `configs/project.yaml`, `configs/scenario.yaml` and
  `.github/workflows/ci.yml`.
- `git diff --check` — passed.
- `.github/workflows/ci.yml` runs editable install, Ruff lint/format, pytest and
  the 80% coverage gate on Ubuntu with Python 3.11 without downloading NASA
  data.

### Phase 1 closure — 2026-07-29

M1.1:

- `.\.venv\Scripts\python.exe -m aeromaintain prepare --project-root .`
  downloaded the configured NASA archive and verified
  `Bytes=12425978`,
  `SHA256=74BEF434A34DB25C7BF72E668EA4CD52AFE5F2CF8E44367C55A82BFD91A5A34F`.
- The command selected only `train_FD001.txt`, `test_FD001.txt` and
  `RUL_FD001.txt`; generated raw and processed data remain Git-ignored.
- Focused tests reject an altered archive, `../` path traversal and a
  conflicting existing extracted member. Identical reruns reuse existing
  bytes without silent replacement.

M1.2:

- Real-data `prepare` produced train `20,631` rows/`100` engines and test
  `13,096` rows/`100` engines with the fixed 26-column schema.
- `data/processed/fd001/data_quality_report.json` records zero nulls, zero
  duplicate `(unit_id, cycle)` keys, finite numeric values and ordered,
  stepwise cycles.
- Processed hashes include
  `train.parquet=85FDCB9BCD821CBDAF993E02698FCC2A20BD39AC57D3FA7E7A5DB6E46191BC7B`
  and
  `test.parquet=245E6E002199F5EA52D718E2EF2C320934321C93EE7509CA3DF312BF697E6479`.

M1.3:

- Train output contains non-negative `rul_true` and
  `rul_target=min(rul_true, 125)`; the observed capped maximum is `125`.
- `data/processed/fd001/split_manifest.json` records algorithm
  `lifetime-quartile-stratified-v1`, seed `42`, engine lifetimes/quartiles, 80
  development IDs and 20 disjoint calibration IDs. Its SHA-256 is
  `DA3B2053096A0DD8FAF9DE130C5597E80E7AD9ED5D6B74974548BD682A6DA504`.
- Official test labels contain 100 rows and are stored only at
  `data/processed/fd001/evaluation/test_rul.parquet`; the processed test
  feature table has no true-RUL column. The manifest marks labels for locked
  evaluation only.

M1.4:

- `data/processed/fd001/eda_summary.json` covers only the 80 development
  engines and reports sensor variability, engine lifetimes, null counts, IQR
  outlier counts and selected trend sensors.
- `data/processed/fd001/eda_report.html` contains interactive selected-sensor
  trends for a deterministic engine sample and the development-only scope
  warning.
- Two consecutive real-data `prepare` runs produced the same SHA-256 set for
  all eight processed files, including
  `manifest.json=3ABA2C041982DC16AFDE1437F555CEBE4C59748185F6EF744BCC32B5D96F03BD`.
- `.\.venv\Scripts\pytest.exe --cov=src/aeromaintain
  --cov-report=term-missing --cov-fail-under=80` — `12 passed`, total coverage
  `81.62%`.
- `.\.venv\Scripts\ruff.exe check .`,
  `.\.venv\Scripts\ruff.exe format --check .` and `git diff --check` passed.
- `git ls-files data/raw data/processed artifacts runs` returned no paths.

### Phase 2 closure — 2026-07-29

M2.1:

- `src/aeromaintain/features/causal.py` generates 403 stable features: three
  current operating settings, 21 current sensors, engine age, and for every
  sensor the causal mean, standard deviation, minimum, maximum, linear slope,
  and last-minus-mean over 5/10/20 cycles.
- Fold-local preprocessing fits median imputation and constant-column removal
  only on the active training partition; Ridge scaling is also training-fold
  local. The input and retained feature order are persisted.
- Focused tests changed a future sensor row and proved all earlier feature rows
  remained byte-equivalent; shuffled inference columns fail closed.

M2.2:

- `runs/m2-fd001-seed42-20260729/cv_folds.json` records five shared
  `GroupKFold` partitions across 80 development engines with disjoint
  train/validation engine IDs.
- The development target mean and Ridge `alpha ∈ {0.1, 1, 10, 100}` use the
  same folds, engine-equal total sample weights, and common MAE, RMSE,
  motor-normalized NASA score, signed bias, overprediction, critical-RUL, and
  RUL-band metrics.
- Best Ridge: `alpha=100`; development OOF RMSE `50.893429`, MAE `34.393061`,
  motor-normalized NASA score `3248762.702696`.

M2.3:

- `.\.venv\Scripts\python.exe -m aeromaintain train --run-id
  m2-fd001-seed42-20260729 --project-root .` completed 12 XGBoost candidates,
  five folds each, with `tree_method=hist`, seed `42`, at most 1,500 trees, and
  75-round early stopping.
- Best XGBoost candidate `5` used fold tree counts
  `[358, 445, 357, 416, 282]`; its RMSE improvement over Ridge was only
  `1.763960%`, and its NASA score was worse. The fixed 5%-plus-NASA rule
  therefore selected Ridge without calibration or official-test feedback.
- The independent run `m2-fd001-seed42-repro-20260729` reproduced the same
  champion and `q`. Nine decision/model/calibration artefacts were byte
  identical, including
  `champion_decision.json`
  SHA-256 `99C8D26AFB21EFBC7732365E00CBDEDAE036E37C8C9194B71157A69C5CE6E662`,
  `model_comparison.json`
  `ECC15595662E64F3EA6B58E92079D95A832F5C92FBCA057A5C42A60EF43E1CF9`,
  and `model.joblib`
  `1F136D51A0E1FC61F284674A04102E685AF3D3DE0AB1EEDEC9CBE789FF86A7AF`.

M2.4:

- The 20 sorted calibration engines were assigned cyclic cutoffs
  `20/60/100/126`; each produced exactly one absolute residual.
- The nominal 90% finite-sample order statistic used rank `19/20` and produced
  `q=31.701394860140987`. Interval lower bounds are clamped to zero and risk
  bands use `interval_low`; reports explicitly state that the interval is
  empirical, nominal, and not a safety guarantee.
- Ridge standardized coefficients provide global and selected-engine local
  model-behavior explanations; no physical-causality claim is made.
- `model_lock.json` SHA-256 is
  `6A418C3128F6A61BCDEC4E3087B1CB7B4DE47AAE40581A489968B8BBFF72FFB7`;
  it covers data/split/config/feature/model/calibration identities and hashes.
  Save/load prediction equivalence and changed-artifact rejection pass in the
  focused tests.

M2.5:

- `.\.venv\Scripts\python.exe -m aeromaintain evaluate --run-id
  m2-fd001-seed42-20260729 --project-root .` verified the lock before reading
  isolated official labels and evaluated all 100 test engines.
- Official test metrics: MAE `15.369728`, RMSE `19.622062`,
  motor-normalized NASA score `625.326953`, signed bias `-1.777402`, critical
  RUL precision `1.000000`, recall `0.480000`, and F1 `0.648649`.
- The nominal 90% empirical interval observed `0.89` official-test coverage
  with mean width `60.623140`; this is reported as evidence, not a guarantee.
- Two evaluations of the same lock produced byte-identical official artefacts:
  predictions
  `5EC4F218D6CA593CC59E6CDA3087033952D4B7AECC89D5BA0BD960D0C1D23DB5`,
  metrics
  `527615328DCABFD61F0ABB2BAA3F4998EEF10C4697B527E41282F93255921197`,
  error analysis
  `8EF33AC7AAA313E2A44FAAB78F669FFDE398095D839E9042C12D0874013BCF7D`,
  and evaluation manifest
  `3AF57755DBEA410D92AF8DEC4E543C65BBB00AA0CD2402DBD23D0763D942D1FE`.
- `.\.venv\Scripts\pytest.exe --cov=src/aeromaintain
  --cov-report=term-missing --cov-fail-under=80` — `19 passed`, total coverage
  `84.78%`.
- `.\.venv\Scripts\ruff.exe check .` and
  `.\.venv\Scripts\ruff.exe format --check .` — all checks passed; all 31
  files formatted.
- `.\.venv\Scripts\python.exe -m pip check` — no broken requirements;
  `.\.venv\Scripts\python.exe -m aeromaintain doctor --project-root .` —
  `5/5` checks passed.
- `git diff --check` and `git diff --cached --check` passed.
- `git check-ignore` confirms both real run directories are ignored;
  `git ls-files data/raw data/processed artifacts runs` returned no paths.

### Phase 3 closure — 2026-07-29

M3.1:

- `.\.venv\Scripts\python.exe -m aeromaintain optimize --run-id
  m2-fd001-seed42-20260729 --project-root .` verified the locked official
  prediction and model-lock hashes before creating the immutable
  `optimization/` output.
- `scenario.json` selects the 20 lowest `interval_low` engines with engine-ID
  tie-breaking, uses the 30-day horizon and seed `42`, and records all
  generated engine/resource/cost values plus generator version
  `fd001-synthetic-maintenance-v1`.
- Two independent generations produced the same scenario SHA-256:
  `0D5EC52C503C3D095E2D9658D74CE04ADC9DA268F1EF6E8FD79EDE629ED8D4FB`.
- Planning artefact scan found no `rul_true` or `true_rul`; focused sentinel
  tests reject `rul_true`, `true_rul`, and `actual_rul` input columns.
  `scenario_data_dictionary.json` and `configs/scenario.yaml` label every
  operational, resource, duration and cost field as synthetic.

M3.2:

- Reactive, fixed-90-cycle and predicted-RUL-30 policies use the same scenario,
  teams, bays, parts and operating-demand contract. Triggered jobs are attempted
  on the earliest feasible day in safe-due-day/engine-ID order and
  unschedulable work is explicitly deferred.
- All three policies and CP-SAT pass the common evaluator. Base results:
  reactive `0` scheduled/`20` failures/`10000` total synthetic cost units;
  fixed-90 `13`/`18`/`10349`; predicted-RUL-30 `16`/`18`/`10662`.
- True RUL is joined only by `evaluate_retrospective` after schedules are
  frozen; raw truth never appears in the scenario, policy, solver or schedule
  schema.

M3.3:

- The two-stage CP-SAT model enforces one assignment or deferral, full-duration
  same-team/same-bay use, daily technician capacity, bay no-overlap, cumulative
  kit stock, minimum operating capacity and completion within the horizon.
- Stage 1 minimizes due deferrals before late days; stage 2 fixes the best
  stage-1 score found and minimizes planned, early-cycle and low-risk-deferral
  cost. Both stages use seed `42`, one worker and a 30-second limit.
- The hand-solvable one-engine fixture returns `OPTIMAL`, start day `0`, end
  day `2`, zero due deferrals and zero late days. A forced impossible fixture
  returns `INFEASIBLE` with an empty schedule; an `UNKNOWN` status is likewise
  verified to expose no plausible schedule.
- Real base result: `FEASIBLE`, lexicographic optimality `unproven`,
  `17` scheduled, `3` due deferrals, `264` late days, `0` operating-capacity
  shortfall, `13` retrospective failures and `8287` total synthetic cost units.

M3.4:

- `policy_comparison.json/.csv` records the same decision and retrospective
  metrics for all four policies. CP-SAT improves this controlled synthetic
  run over the three baselines on failures and total synthetic cost, without
  claiming proven optimality or real-fleet performance.
- Capacity results preserve the same engine synthetic fields:
  constrained (`1` bay/`90%`) schedules `10` with `10` due deferrals and
  `17` failures; base (`2`/`80%`) schedules `17` with `3` and `13`; expanded
  (`3`/`70%`) schedules `19` with `1` and `12`. All report zero operating
  shortfall.
- `docs/optimization.md` documents the truth boundary, scenario, four
  policies, integer constraints, two-stage objective, negative-status behavior,
  actual FEASIBLE/unproven result and capacity sensitivity.
- Manifest validation found zero hash mismatches across all 13 optimization
  artefacts; config hash and locked source hashes match. A repeated `optimize`
  refuses to overwrite the existing output.
- `.\.venv\Scripts\pytest.exe --cov=src/aeromaintain
  --cov-report=term-missing --cov-fail-under=80` — `35 passed`, total coverage
  `86.13%`.
- `.\.venv\Scripts\ruff.exe check .` and
  `.\.venv\Scripts\ruff.exe format --check .` — all Python files pass after the
  final annotation correction.

### Phase 4 closure — 2026-07-29

M4.1:

- `src/aeromaintain/app/artifacts.py` requires one explicit safe run ID and
  verifies the current config/data hashes, every locked model artefact, official
  evaluation manifest, optimization source/output hashes and completed-pipeline
  report hashes. External paths, missing or changed files, malformed structures
  and planning scenarios containing `rul_true`, `true_rul` or `actual_rul`
  fail closed.
- `src/aeromaintain/app/main.py` renders Overview, Engine Health, Maintenance
  Schedule and Policy Comparison & What-if. Decision tables select a safe schema
  with no official true RUL; policy/capacity/risk/schedule CSV downloads are
  exposed.
- What-if inputs are validated at 1–3 bays and 70%–90% minimum operating demand
  before the solver is called. The solver receives only the verified synthetic
  scenario and returns its real status; no-solution results contain no schedule.
- The small Streamlit fixture rendered all four pages and ran a truth-free
  what-if. Browser verification of the real release run produced
  `docs/screenshots/{overview,engine-health,maintenance-schedule,
  policy-comparison}.png` with no browser-console errors.

M4.2:

- `.\.venv\Scripts\python.exe -m aeromaintain pipeline --run-id
  m4-fd001-seed42-20260729 --project-root .` completed prepare, train/lock,
  locked official evaluation, optimization and report in one command.
- The final run manifest has status `pipeline_complete`, model-lock SHA-256
  `32E5E7A421AD8CCB81FFF4E8B8C17957F49FA855954B27E4E4B16219634AF829`
  and verified hashes for `report.json`, `report.html`,
  `official_test/evaluation_manifest.json` and
  `optimization/manifest.json`.
- Repeating the pipeline with the same ID fails before preparation with
  `Run directory already exists`. The small end-to-end fixture proves stage
  order, final report creation and hash validation; an injected evaluation
  failure remains `model_locked`, writes no report and is never marked complete.

M4.3:

- `README.md` contains tested installation, pipeline, app, smoke and quality
  commands plus the verified release result and screenshots.
- `docs/model_card.md` records intended/prohibited uses, data/split/feature
  contracts, locked metrics, limitations, critical-RUL recall and interval
  coverage shortfall. `docs/results.md` reports all four policies and three
  capacity cases without hiding the FEASIBLE/unproven solver outcome.
- `docs/architecture.md` documents the prediction-to-decision flow, label/truth
  boundaries, immutable run state, application hash checks and actual two-stage
  CP-SAT structure. `docs/optimization.md` remains the detailed optimization
  contract/result document.
- Git tracks no raw NASA data, processed tables, run directory or model file.
  The four PNG screenshots total less than 250 KB.

M4.4:

- A new temporary Python `3.11.9` virtual environment installed `.[dev]`
  successfully; `pip check` returned `No broken requirements found`, package
  version `0.1.0` imported, and `python -m aeromaintain doctor --project-root .`
  returned `5/5` PASS.
- The clean environment ran `python -m aeromaintain app --run-id
  m4-fd001-seed42-20260729 --project-root . --smoke` successfully.
- Current environment gates:
  `.\.venv\Scripts\ruff.exe check .` passed;
  `.\.venv\Scripts\ruff.exe format --check .` reported 41 files formatted;
  `.\.venv\Scripts\pytest.exe --cov=src/aeromaintain
  --cov-report=term-missing --cov-fail-under=80` reported `41 passed` and
  `84.45%` coverage.
- The real-run Streamlit smoke passed again. `load_verified_run` returned
  `pipeline_complete` only after verifying the full release hash chain.
- `git check-ignore -v` matched the real raw archive, processed manifest,
  release run manifest and artefact probe. `git ls-files data/raw
  data/processed artifacts runs` returned no paths.
- `git diff --check` passed.

## Decisions and blockers

- Fixed assumptions live in `PROJECT_PLAN.md`; do not duplicate or change them
  here.
- Record a blocker with the failing command, exact error and required next
  action.
