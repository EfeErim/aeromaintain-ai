# Research Basis and Method Decisions

Checked: 2026-11-04

## Scope

AeroMaintain AI evaluates RUL prediction on NASA's simulated C-MAPSS FD001 and FD003
benchmarks through a reproducible, leakage-controlled model-selection and locked
official-test workflow.

## Why FD001

FD001 is the smallest C-MAPSS subset that still supports engine-grouped
validation, causal time-series features, model comparison, and uncertainty
analysis. It has one operating condition and one simulated degradation mode.
That narrow scope helps make leakage and reproducibility controls reviewable,
but it does not represent real fleet telemetry.

## Why FD003

FD001 labels were seen before protocol v2 was designed, so FD001 cannot test it
independently. FD003 adds a second fault mode and kept its labels closed until the
protocol was frozen. It does not test multiple operating conditions.

## Method basis

| Decision | Project interpretation |
|---|---|
| Engine-grouped validation | Every row from one simulated engine remains in one role or fold. Row-wise random splitting is prohibited. |
| Mean and Ridge baselines | A constant exposes task difficulty; regularized Ridge is an interpretable linear baseline for correlated features. |
| XGBoost candidate | A bounded nonlinear comparator, selected only by the fixed development-only champion rule. |
| Test-like development scoring | Random per-engine truncation matches the test's one prediction per engine. The window of 150 is a disclosed analysis choice. |
| Nominal empirical interval | Mondrian signed-residual bands follow predicted RUL, with engine weighting over all training engines; coverage is measured, not guaranteed. |
| Uncertainty | Wilson intervals cover alert rates, and an engine-level bootstrap covers official metrics. |
| Explanation | Ridge coefficients or SHAP describe the fitted model, not physical causality. |

## Primary sources

1. NASA Open Data, [CMAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
2. Saxena et al., [Damage Propagation Modeling for Aircraft Engine Prognostics](https://ntrs.nasa.gov/citations/20090029214).
3. Hoerl and Kennard, [Ridge Regression](https://doi.org/10.1080/00401706.1970.10488634).
4. Chen and Guestrin, [XGBoost](https://arxiv.org/abs/1603.02754).
5. Lundberg and Lee, [SHAP](https://arxiv.org/abs/1705.07874).
6. Angelopoulos and Bates, [Conformal Prediction overview](https://arxiv.org/abs/2107.07511).
7. Romano et al., [Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222).
8. Wilson, [Probable Inference, the Law of Succession, and Statistical Inference](https://doi.org/10.1080/01621459.1927.10502953).

## Boundaries

- A simulated, single-condition result does not validate an operational fleet.
- Official test labels cannot influence selection, calibration, or thresholds. FD001
  labels were seen before v2, so only FD003 is a clean test of it.
- Explanations are associational model descriptions.
- The nominal interval is not a safety guarantee.
