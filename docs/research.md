# Research Basis and Method Decisions

Checked: 2026-08-04

## Scope

AeroMaintain AI evaluates RUL prediction on NASA's simulated C-MAPSS FD001
benchmark through a reproducible, leakage-controlled model-selection and locked
official-test workflow.

## Why FD001

FD001 is the smallest C-MAPSS subset that still supports engine-grouped
validation, causal time-series features, model comparison, and uncertainty
analysis. It has one operating condition and one simulated degradation mode.
That narrow scope helps make leakage and reproducibility controls reviewable,
but it does not represent real fleet telemetry.

## Method basis

| Decision | Project interpretation |
|---|---|
| Engine-grouped validation | Every row from one simulated engine remains in one role or fold. Row-wise random splitting is prohibited. |
| Mean and Ridge baselines | A constant exposes task difficulty; regularized Ridge is an interpretable linear baseline for correlated features. |
| XGBoost candidate | A bounded nonlinear comparator, selected only by the fixed development-only champion rule. |
| Nominal empirical interval | A held-out calibration engine set supplies one absolute residual per engine; coverage is measured, not guaranteed. |
| Explanation | Ridge coefficients or SHAP describe the fitted model, not physical causality. |

## Primary sources

1. NASA Open Data, [CMAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
2. Saxena et al., [Damage Propagation Modeling for Aircraft Engine Prognostics](https://ntrs.nasa.gov/citations/20090029214).
3. Hoerl and Kennard, [Ridge Regression](https://doi.org/10.1080/00401706.1970.10488634).
4. Chen and Guestrin, [XGBoost](https://arxiv.org/abs/1603.02754).
5. Lundberg and Lee, [SHAP](https://arxiv.org/abs/1705.07874).
6. Angelopoulos and Bates, [Conformal Prediction overview](https://arxiv.org/abs/2107.07511).

## Boundaries

- A simulated, single-condition result does not validate an operational fleet.
- Official test labels cannot influence selection, calibration, or thresholds.
- Explanations are associational model descriptions.
- The nominal interval is not a safety guarantee.
