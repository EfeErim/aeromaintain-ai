# Research basis and method decisions

Checked: 2026-11-05

## Scope

AeroMaintain AI evaluates RUL prediction on NASA's simulated C-MAPSS FD001 through FD004
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

## Why FD002 and FD004

These subsets test whether the approach holds under six operating conditions, and their
labels stayed closed until protocol v3 was frozen and its locks published. They also
have engines with far more remaining life than FD001 or FD003, up to 195 cycles against
145, which exposed the cost of the RUL cap and the development window.

## Method basis

| Decision | Project interpretation |
|---|---|
| Engine-grouped validation | Every row from one simulated engine remains in one role or fold. Row-wise random splitting is prohibited. |
| Mean and Ridge baselines | A constant exposes task difficulty; regularized Ridge is an interpretable linear baseline for correlated features. |
| XGBoost candidate | A bounded nonlinear comparator, selected only by the fixed development-only champion rule. |
| Test-like development scoring | Random per-engine truncation matches the test's one prediction per engine. The window of 150 is a disclosed analysis choice. |
| Nominal empirical interval | Mondrian signed-residual bands follow predicted RUL, with engine weighting over all training engines; coverage is measured, not guaranteed. |
| Condition normalization | Six KMeans clusters on operating settings and per-condition sensor standardization, fitted inside each training partition. |
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

## Comparison with published results

Checked: 2026-11-07. Published C-MAPSS RMSE is usually scored against test truth capped
at 125 or 130 cycles, and the official test set is often reused during model
development. The project's headline uses uncapped truth and single-open tests, so a
direct comparison needs rescoring on the same scale. That rescoring is retrospective and
lives in [`results_retrospective.md`](results_retrospective.md).

9. Babu et al., CNN regression for RUL (2016), and Zheng et al., LSTM for RUL (2017),
   as tabulated in [Deep learning models for predictive maintenance](https://arxiv.org/abs/2010.03207).
10. Li, Ding and Sun, [RUL estimation using deep convolution neural networks](https://econpapers.repec.org/RePEc:eee:reensy:v:172:y:2018:i:c:p:1-11),
    Reliability Engineering & System Safety 172 (2018).
11. Chowdhury et al., [Bi-cLSTM for aero-engine RUL estimation](https://arxiv.org/abs/2603.00745)
    (2026), including its Table V comparison.
12. Gupta, [audit of normalization and pipeline choices on C-MAPSS](https://engrxiv.org/preprint/view/8145).
13. Javanmardi and Hüllermeier, [conformal prediction intervals for RUL](https://arxiv.org/abs/2212.14612).

## Boundaries

- A simulated result does not validate an operational fleet.
- Official test labels cannot influence selection, calibration, or thresholds. FD001
  labels were seen before v2, so only FD003 is a clean test of v2. FD001 and FD003
  labels were seen before v3, so only FD002 and FD004 are clean tests of v3.
- Explanations are associational model descriptions.
- The nominal interval is not a safety guarantee.
