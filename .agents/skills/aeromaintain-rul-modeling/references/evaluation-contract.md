# RUL Evaluation Contract

Checked: 2026-07-29

## Primary documentation

- GroupKFold:
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
- Scikit-learn model persistence:
  https://scikit-learn.org/stable/model_persistence.html
- XGBoost Python API:
  https://xgboost.readthedocs.io/en/stable/python/python_api.html
- SHAP TreeExplainer:
  https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
- Conformal prediction overview:
  https://arxiv.org/abs/2107.07511

GroupKFold keeps a group out of the training fold that evaluates it. XGBoost
early stopping requires an evaluation set and records `best_iteration`.
Scikit-learn warns that pickle-based persistence formats, including `joblib`,
must load only trusted files and require compatible environments.

## NASA asymmetric score

For error `d = predicted_rul - true_rul`:

```text
d < 0: exp(-d / 13) - 1
d >= 0: exp(d / 10) - 1
```

Sum per-row penalties within each engine, divide by that engine's evaluated row
count, then sum engine values. Compute in float64 with `expm1`. Overestimating
RUL is penalized more heavily.

## Leakage checks

- Engine IDs are disjoint across development folds and calibration.
- Modifying a future row cannot modify an earlier feature row.
- Preprocessing is fitted within the active training fold.
- Official test labels are inaccessible before a valid model lock exists.
- A repeated evaluation of the same lock produces identical predictions and
  metrics within deterministic numeric tolerances.
