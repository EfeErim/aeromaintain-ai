# Research Basis and Method Decisions

Checked: 2026-07-29

## Scope

AeroMaintain AI is an educational and portfolio prototype. It predicts remaining
useful life (RUL) from NASA's simulated C-MAPSS turbofan histories and converts
the prediction into a maintenance-planning scenario with explicitly synthetic
resources and costs. It is not an airworthiness, maintenance-approval, or
production fleet system.

## Why FD001

NASA describes FD001 as 100 training and 100 test trajectories under one
operating condition with one high-pressure-compressor degradation mode. That
bounded setting is the smallest C-MAPSS subset that still supports:

- engine-grouped validation;
- causal time-series features;
- RUL model comparison and uncertainty analysis; and
- a transparent handoff from prediction to a synthetic scheduling problem.

FD002-FD004 introduce additional operating conditions and/or fault modes. They
remain outside V1 so that leakage controls, reproducibility, and decision logic
can be validated before increasing dataset complexity.

## Frozen data facts

The official archive was downloaded directly from NASA and rechecked on
2026-07-29:

- URL: <https://data.nasa.gov/docs/legacy/CMAPSSData.zip>
- byte size: `12,425,978`
- SHA-256:
  `74BEF434A34DB25C7BF72E668EA4CD52AFE5F2CF8E44367C55A82BFD91A5A34F`

NASA describes the dataset as simulated multivariate run-to-failure time
series. Each row is one engine-cycle snapshot with unit ID, cycle, three
operational settings, and 21 sensor values. Training trajectories reach
failure; test trajectories stop before failure and have a separate RUL vector.

## Method basis

| Decision | Basis and project interpretation |
|---|---|
| Engine-grouped validation | Each trajectory belongs to a distinct simulated engine. All rows from one engine stay in one role or fold; row-wise random splitting is prohibited. |
| Mean and Ridge baselines | A constant model exposes task difficulty. L2-regularized Ridge provides an interpretable linear baseline for correlated engineered sensor features. |
| XGBoost candidate | Chen and Guestrin's tree-boosting system is a strong tabular nonlinear comparator. It is not automatically selected; the fixed champion rule in `PROJECT_PLAN.md` governs selection. |
| Nominal empirical interval | A held-out calibration engine set supplies one absolute residual per engine. The finite-sample quantile creates a simple split-conformal-style interval, reported as nominal rather than guaranteed for a real fleet. |
| Explanation | Ridge coefficients or SHAP explain fitted model behavior, not physical causality. |
| CP-SAT scheduling | OR-Tools CP-SAT represents integer day, resource, and deferral decisions and exposes explicit solver statuses. Only `OPTIMAL` or `FEASIBLE` may produce a schedule. |

## Primary and authoritative sources

1. NASA Open Data, [CMAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
   This is the authoritative dataset description, subset table, schema summary,
   and archive link. The page currently states that no license is specified.
2. Saxena, Goebel, Simon, and Eklund (2008),
   [Damage Propagation Modeling for Aircraft Engine Prognostics](https://ntrs.nasa.gov/citations/20090029214).
   This is the primary C-MAPSS run-to-failure simulation and PHM challenge
   reference, including the prognostics setting and asymmetric score.
3. Hoerl and Kennard (1970),
   [Ridge Regression: Biased Estimation for Nonorthogonal Problems](https://doi.org/10.1080/00401706.1970.10488634).
   This is the original Ridge regression paper.
4. Chen and Guestrin (2016),
   [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754).
   This is the primary XGBoost systems paper.
5. Lundberg and Lee (2017),
   [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874).
   This introduces SHAP's additive feature-attribution framework.
6. Angelopoulos and Bates (2021),
   [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511).
   This provides the finite-sample calibration context used to motivate the
   project's deliberately modest nominal interval.
7. Google OR-Tools,
   [CP-SAT Solver](https://developers.google.com/optimization/cp/cp_solver).
   This is the authoritative solver interface and status contract used by the
   optimization plan.

## Research boundaries

- The NASA data does not validate performance on an operational fleet.
- A single-condition, single-fault-mode result does not establish robustness to
  changing environments or unseen failure mechanisms.
- Model explanations are associational descriptions of predictions.
- Nominal interval coverage is measured on the locked test evaluation; it is
  not a safety guarantee.
- All operational capacity, duration, parts, and cost values are synthetic and
  must remain labeled as such.
