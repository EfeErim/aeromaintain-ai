"""Leakage-safe RUL training, calibration, locking, and official evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, ParameterSampler
from xgboost import XGBRegressor

from aeromaintain.data.pipeline import sha256_file
from aeromaintain.evaluation import prediction_metrics
from aeromaintain.features import (
    FEATURE_NAMES,
    FoldPreprocessor,
    build_causal_features,
)

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
CALIBRATION_CUTOFFS = (20, 60, 100, 126)
MODEL_SCHEMA_VERSION = 1


class ModelingError(RuntimeError):
    """Raised when an M2 modeling or lock contract fails."""


@dataclass(frozen=True)
class ModelingConfig:
    """Fixed M2 experiment values loaded from the project contract."""

    seed: int = 42
    group_folds: int = 5
    critical_rul_threshold: int = 30
    nominal_interval_coverage: float = 0.90
    xgboost_candidates: int = 12
    xgboost_max_trees: int = 1500
    xgboost_early_stopping_rounds: int = 75
    xgboost_n_jobs: int = 4


@dataclass
class RidgeBundle:
    """Locally persisted Ridge preprocessing and estimator."""

    preprocessor: FoldPreprocessor
    estimator: Ridge

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict non-negative RUL in persisted feature order."""
        transformed = self.preprocessor.transform(features)
        return np.maximum(0.0, self.estimator.predict(transformed))


@dataclass(frozen=True)
class TrainResult:
    """CLI-safe summary for one newly created locked model run."""

    run_id: str
    run_dir: Path
    champion: str
    calibration_q: float
    model_lock: Path


@dataclass(frozen=True)
class EvaluationResult:
    """CLI-safe summary for a locked official test evaluation."""

    run_id: str
    output_dir: Path
    metrics: dict[str, Any]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_new_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise ModelingError(f"Refusing to overwrite different artifact: {path}")
    path.write_bytes(payload)


def _write_json(path: Path, value: Any) -> None:
    _write_new_bytes(path, _json_bytes(value))


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    frame.to_parquet(buffer, index=False, engine="pyarrow")
    return buffer.getvalue()


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_modeling_config(project_root: Path) -> ModelingConfig:
    """Load the shared seed, folds, threshold, and nominal coverage."""
    config_path = project_root / "configs" / "project.yaml"
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        project = payload["project"]
        modeling = payload["modeling"]
        return ModelingConfig(
            seed=int(project["seed"]),
            group_folds=int(modeling["group_folds"]),
            critical_rul_threshold=int(modeling["critical_rul_threshold"]),
            nominal_interval_coverage=float(modeling["nominal_interval_coverage"]),
        )
    except (KeyError, TypeError, ValueError, OSError, yaml.YAMLError) as exc:
        raise ModelingError(f"Invalid modeling configuration: {exc}") from exc


def engine_equal_weights(engine_ids: np.ndarray) -> np.ndarray:
    """Give every engine equal total influence while keeping mean weight one."""
    engines = pd.Series(np.asarray(engine_ids))
    counts = engines.map(engines.value_counts()).to_numpy(dtype=np.float64)
    return len(engines) / (engines.nunique() * counts)


def _load_and_validate_training_inputs(
    project_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    processed = project_root / "data" / "processed" / "fd001"
    manifest_path = processed / "manifest.json"
    split_path = processed / "split_manifest.json"
    train_path = processed / "train.parquet"
    config_path = project_root / "configs" / "project.yaml"
    for path in (manifest_path, split_path, train_path, config_path):
        if not path.is_file():
            raise ModelingError(f"Required Phase 1 artifact is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    expected_hashes = manifest.get("artifacts", {})
    for relative_path in ("train.parquet", "split_manifest.json"):
        observed = sha256_file(processed / relative_path)
        if observed != expected_hashes.get(relative_path):
            raise ModelingError(f"Phase 1 artifact hash mismatch: {relative_path}")
    if sha256_file(config_path) != manifest.get("config_sha256"):
        raise ModelingError(
            "Project config changed after data preparation; rerun Phase 1 prepare"
        )

    train = pd.read_parquet(train_path)
    required = {"unit_id", "cycle", "rul_true", "rul_target", "role"}
    if not required.issubset(train.columns):
        raise ModelingError("Prepared train table is missing M2 contract columns")
    development = set(split["development_engine_ids"])
    calibration = set(split["calibration_engine_ids"])
    if development & calibration:
        raise ModelingError("Development and calibration engine IDs overlap")
    observed_development = set(
        train.loc[train["role"].eq("development"), "unit_id"].unique()
    )
    observed_calibration = set(
        train.loc[train["role"].eq("calibration"), "unit_id"].unique()
    )
    if development != observed_development or calibration != observed_calibration:
        raise ModelingError("Prepared train roles disagree with split manifest")
    return train, manifest, split


def _fold_plan(
    development: pd.DataFrame,
    config: ModelingConfig,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    groups = development["unit_id"].to_numpy()
    splitter = GroupKFold(
        n_splits=config.group_folds,
        shuffle=True,
        random_state=config.seed,
    )
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    records: list[dict[str, Any]] = []
    for fold_index, (train_index, validation_index) in enumerate(
        splitter.split(development, groups=groups),
        start=1,
    ):
        train_engines = sorted(int(value) for value in np.unique(groups[train_index]))
        validation_engines = sorted(
            int(value) for value in np.unique(groups[validation_index])
        )
        if set(train_engines) & set(validation_engines):
            raise ModelingError(f"Engine leakage detected in fold {fold_index}")
        folds.append((train_index, validation_index))
        records.append(
            {
                "fold": fold_index,
                "train_engine_ids": train_engines,
                "validation_engine_ids": validation_engines,
            }
        )
    return folds, records


def _metric_rank(record: dict[str, Any]) -> tuple[float, float]:
    metrics = record["metrics"]
    return (
        float(metrics["nasa_score_motor_normalized"]),
        float(metrics["rmse"]),
    )


def _xgboost_parameter_candidates(config: ModelingConfig) -> list[dict[str, Any]]:
    space = {
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.02, 0.04, 0.06, 0.1],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
    }
    return [
        _safe_json_value(candidate)
        for candidate in ParameterSampler(
            space,
            n_iter=config.xgboost_candidates,
            random_state=config.seed,
        )
    ]


def _fit_ridge_cv(
    features: pd.DataFrame,
    development: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: ModelingConfig,
) -> tuple[list[dict[str, Any]], dict[float, np.ndarray]]:
    records: list[dict[str, Any]] = []
    predictions_by_alpha: dict[float, np.ndarray] = {}
    target = development["rul_target"].to_numpy(dtype=np.float64)
    truth = development["rul_true"].to_numpy(dtype=np.float64)
    engines = development["unit_id"].to_numpy()
    for alpha in RIDGE_ALPHAS:
        oof = np.empty(len(development), dtype=np.float64)
        fold_records: list[dict[str, Any]] = []
        for fold_index, (train_index, validation_index) in enumerate(folds, start=1):
            preprocessor = FoldPreprocessor(scale=True).fit(features.iloc[train_index])
            estimator = Ridge(alpha=alpha)
            estimator.fit(
                preprocessor.transform(features.iloc[train_index]),
                target[train_index],
                sample_weight=engine_equal_weights(engines[train_index]),
            )
            fold_prediction = np.maximum(
                0.0,
                estimator.predict(
                    preprocessor.transform(features.iloc[validation_index])
                ),
            )
            oof[validation_index] = fold_prediction
            fold_records.append(
                {
                    "fold": fold_index,
                    "metrics": prediction_metrics(
                        truth[validation_index],
                        fold_prediction,
                        engines[validation_index],
                        critical_threshold=config.critical_rul_threshold,
                    ),
                    "preprocessing": preprocessor.manifest(),
                }
            )
        record = {
            "model": "ridge",
            "alpha": alpha,
            "folds": fold_records,
            "metrics": prediction_metrics(
                truth,
                oof,
                engines,
                critical_threshold=config.critical_rul_threshold,
            ),
        }
        records.append(record)
        predictions_by_alpha[alpha] = oof
    return records, predictions_by_alpha


def _fit_mean_cv(
    development: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: ModelingConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    target = development["rul_target"].to_numpy(dtype=np.float64)
    truth = development["rul_true"].to_numpy(dtype=np.float64)
    engines = development["unit_id"].to_numpy()
    oof = np.empty(len(development), dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    for fold_index, (train_index, validation_index) in enumerate(folds, start=1):
        mean = float(
            np.average(
                target[train_index],
                weights=engine_equal_weights(engines[train_index]),
            )
        )
        oof[validation_index] = mean
        fold_records.append(
            {
                "fold": fold_index,
                "target_mean": mean,
                "metrics": prediction_metrics(
                    truth[validation_index],
                    oof[validation_index],
                    engines[validation_index],
                    critical_threshold=config.critical_rul_threshold,
                ),
            }
        )
    return (
        {
            "model": "development_target_mean",
            "folds": fold_records,
            "metrics": prediction_metrics(
                truth,
                oof,
                engines,
                critical_threshold=config.critical_rul_threshold,
            ),
        },
        oof,
    )


def _new_xgboost(
    params: dict[str, Any],
    config: ModelingConfig,
    *,
    n_estimators: int,
    early_stopping: bool,
) -> XGBRegressor:
    return XGBRegressor(
        **params,
        objective="reg:squarederror",
        n_estimators=n_estimators,
        tree_method="hist",
        random_state=config.seed,
        n_jobs=config.xgboost_n_jobs,
        eval_metric="rmse",
        early_stopping_rounds=(
            config.xgboost_early_stopping_rounds if early_stopping else None
        ),
    )


def _fit_xgboost_cv(
    features: pd.DataFrame,
    development: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    config: ModelingConfig,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    records: list[dict[str, Any]] = []
    predictions_by_candidate: dict[int, np.ndarray] = {}
    target = development["rul_target"].to_numpy(dtype=np.float64)
    truth = development["rul_true"].to_numpy(dtype=np.float64)
    engines = development["unit_id"].to_numpy()
    for candidate_index, params in enumerate(
        _xgboost_parameter_candidates(config), start=1
    ):
        oof = np.empty(len(development), dtype=np.float64)
        fold_records: list[dict[str, Any]] = []
        for fold_index, (train_index, validation_index) in enumerate(folds, start=1):
            preprocessor = FoldPreprocessor(scale=False).fit(features.iloc[train_index])
            train_features = preprocessor.transform(features.iloc[train_index])
            validation_features = preprocessor.transform(
                features.iloc[validation_index]
            )
            estimator = _new_xgboost(
                params,
                config,
                n_estimators=config.xgboost_max_trees,
                early_stopping=True,
            )
            estimator.fit(
                train_features,
                target[train_index],
                sample_weight=engine_equal_weights(engines[train_index]),
                eval_set=[(validation_features, target[validation_index])],
                sample_weight_eval_set=[
                    engine_equal_weights(engines[validation_index])
                ],
                verbose=False,
            )
            fold_prediction = np.maximum(0.0, estimator.predict(validation_features))
            oof[validation_index] = fold_prediction
            fold_records.append(
                {
                    "fold": fold_index,
                    "best_iteration": int(estimator.best_iteration),
                    "best_tree_count": int(estimator.best_iteration) + 1,
                    "best_score": float(estimator.best_score),
                    "metrics": prediction_metrics(
                        truth[validation_index],
                        fold_prediction,
                        engines[validation_index],
                        critical_threshold=config.critical_rul_threshold,
                    ),
                    "preprocessing": preprocessor.manifest(),
                }
            )
        records.append(
            {
                "model": "xgboost",
                "candidate": candidate_index,
                "parameters": params,
                "folds": fold_records,
                "metrics": prediction_metrics(
                    truth,
                    oof,
                    engines,
                    critical_threshold=config.critical_rul_threshold,
                ),
            }
        )
        predictions_by_candidate[candidate_index] = oof
    return records, predictions_by_candidate


def _select_champion(
    ridge_records: list[dict[str, Any]],
    xgboost_records: list[dict[str, Any]],
) -> dict[str, Any]:
    best_ridge = min(ridge_records, key=_metric_rank)
    best_xgboost = min(xgboost_records, key=_metric_rank)
    ridge_rmse = float(best_ridge["metrics"]["rmse"])
    xgboost_rmse = float(best_xgboost["metrics"]["rmse"])
    improvement = (ridge_rmse - xgboost_rmse) / ridge_rmse
    nasa_not_worse = float(
        best_xgboost["metrics"]["nasa_score_motor_normalized"]
    ) <= float(best_ridge["metrics"]["nasa_score_motor_normalized"])
    xgboost_qualifies = improvement >= 0.05 and nasa_not_worse
    return {
        "rule": (
            "XGBoost only if development RMSE improves over Ridge by at least "
            "5% and motor-normalized NASA score does not worsen"
        ),
        "ridge": {
            "alpha": best_ridge["alpha"],
            "metrics": best_ridge["metrics"],
        },
        "xgboost": {
            "candidate": best_xgboost["candidate"],
            "parameters": best_xgboost["parameters"],
            "metrics": best_xgboost["metrics"],
            "fold_best_tree_counts": [
                fold["best_tree_count"] for fold in best_xgboost["folds"]
            ],
        },
        "xgboost_rmse_improvement_fraction": improvement,
        "xgboost_nasa_not_worse": nasa_not_worse,
        "xgboost_qualifies": xgboost_qualifies,
        "champion": "xgboost" if xgboost_qualifies else "ridge",
    }


def _fit_final_model(
    features: pd.DataFrame,
    development: pd.DataFrame,
    decision: dict[str, Any],
    config: ModelingConfig,
    run_dir: Path,
) -> tuple[Any, FoldPreprocessor, dict[str, str], dict[str, Any]]:
    target = development["rul_target"].to_numpy(dtype=np.float64)
    engines = development["unit_id"].to_numpy()
    weights = engine_equal_weights(engines)
    artifact_hashes: dict[str, str] = {}
    if decision["champion"] == "ridge":
        preprocessor = FoldPreprocessor(scale=True).fit(features)
        estimator = Ridge(alpha=float(decision["ridge"]["alpha"]))
        estimator.fit(
            preprocessor.transform(features),
            target,
            sample_weight=weights,
        )
        bundle = RidgeBundle(preprocessor=preprocessor, estimator=estimator)
        model_path = run_dir / "model.joblib"
        joblib.dump(bundle, model_path)
        artifact_hashes["model.joblib"] = sha256_file(model_path)
        model_spec = {
            "kind": "ridge",
            "format": "trusted-local-joblib",
            "path": "model.joblib",
            "parameters": {"alpha": float(decision["ridge"]["alpha"])},
        }
        return bundle, preprocessor, artifact_hashes, model_spec

    preprocessor = FoldPreprocessor(scale=False).fit(features)
    tree_counts = decision["xgboost"]["fold_best_tree_counts"]
    final_tree_count = int(np.median(np.asarray(tree_counts, dtype=np.int64)))
    estimator = _new_xgboost(
        decision["xgboost"]["parameters"],
        config,
        n_estimators=final_tree_count,
        early_stopping=False,
    )
    estimator.fit(
        preprocessor.transform(features),
        target,
        sample_weight=weights,
        verbose=False,
    )
    model_path = run_dir / "model.json"
    preprocessor_path = run_dir / "preprocessor.joblib"
    estimator.save_model(model_path)
    joblib.dump(preprocessor, preprocessor_path)
    artifact_hashes["model.json"] = sha256_file(model_path)
    artifact_hashes["preprocessor.joblib"] = sha256_file(preprocessor_path)
    model_spec = {
        "kind": "xgboost",
        "format": "xgboost-json-with-trusted-local-joblib-preprocessor",
        "path": "model.json",
        "preprocessor_path": "preprocessor.joblib",
        "parameters": {
            **decision["xgboost"]["parameters"],
            "n_estimators": final_tree_count,
        },
    }
    return estimator, preprocessor, artifact_hashes, model_spec


def _predict_model(
    model: Any,
    preprocessor: FoldPreprocessor,
    features: pd.DataFrame,
    champion: str,
) -> np.ndarray:
    if champion == "ridge":
        return model.predict(features)
    return np.maximum(0.0, model.predict(preprocessor.transform(features)))


def _calibration_rows(calibration: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.Series] = []
    for index, unit_id in enumerate(sorted(calibration["unit_id"].unique())):
        cutoff = CALIBRATION_CUTOFFS[index % len(CALIBRATION_CUTOFFS)]
        engine = calibration.loc[calibration["unit_id"].eq(unit_id)].copy()
        distance = (engine["rul_true"] - cutoff).abs()
        row = (
            engine.loc[distance.eq(distance.min())]
            .sort_values("cycle", ascending=False)
            .iloc[0]
        )
        row = row.copy()
        row["calibration_cutoff"] = cutoff
        selected.append(row)
    return pd.DataFrame(selected).reset_index(drop=True)


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    """Return the finite-sample ascending order statistic."""
    observed = np.sort(np.asarray(scores, dtype=np.float64))
    if not len(observed):
        raise ModelingError("Calibration requires at least one engine score")
    rank = min(len(observed), math.ceil((len(observed) + 1) * coverage))
    return float(observed[rank - 1])


def _build_explanation(
    model: Any,
    preprocessor: FoldPreprocessor,
    sample_features: pd.DataFrame,
    sample_row: pd.Series,
    champion: str,
) -> dict[str, Any]:
    transformed = preprocessor.transform(sample_features)
    names = list(preprocessor.output_features or ())
    if champion == "ridge":
        coefficients = np.asarray(model.estimator.coef_, dtype=np.float64)
        global_rows = sorted(
            (
                {"feature": name, "coefficient": float(coefficient)}
                for name, coefficient in zip(names, coefficients, strict=True)
            ),
            key=lambda row: abs(row["coefficient"]),
            reverse=True,
        )[:30]
        local_values = transformed[0] * coefficients
        local_rows = sorted(
            (
                {"feature": name, "contribution": float(value)}
                for name, value in zip(names, local_values, strict=True)
            ),
            key=lambda row: abs(row["contribution"]),
            reverse=True,
        )[:30]
        method = "standardized Ridge coefficients"
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = np.asarray(explainer.shap_values(transformed))
        global_importance = np.mean(np.abs(shap_values), axis=0)
        global_rows = sorted(
            (
                {"feature": name, "mean_abs_shap": float(value)}
                for name, value in zip(names, global_importance, strict=True)
            ),
            key=lambda row: row["mean_abs_shap"],
            reverse=True,
        )[:30]
        local_rows = sorted(
            (
                {"feature": name, "contribution": float(value)}
                for name, value in zip(names, shap_values[0], strict=True)
            ),
            key=lambda row: abs(row["contribution"]),
            reverse=True,
        )[:30]
        method = "SHAP TreeExplainer"
    return {
        "method": method,
        "scope": "model behavior; not physical causality",
        "global_importance": global_rows,
        "local_explanation": {
            "unit_id": int(sample_row["unit_id"]),
            "cycle": int(sample_row["cycle"]),
            "features": local_rows,
        },
    }


def train_and_lock(
    project_root: Path,
    *,
    run_id: str,
    config: ModelingConfig | None = None,
) -> TrainResult:
    """Train on development engines, calibrate once, and atomically lock a run."""
    root = project_root.resolve()
    if not run_id or Path(run_id).name != run_id:
        raise ModelingError("run_id must be one safe path component")
    active_config = config or load_modeling_config(root)
    train, data_manifest, split_manifest = _load_and_validate_training_inputs(root)
    runs_root = root / "runs"
    final_run_dir = runs_root / run_id
    if final_run_dir.exists():
        raise ModelingError(f"Run directory already exists: {final_run_dir}")
    runs_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=runs_root))

    try:
        development = train.loc[train["role"].eq("development")].reset_index(drop=True)
        calibration = train.loc[train["role"].eq("calibration")].reset_index(drop=True)
        development_features = build_causal_features(development)
        folds, fold_records = _fold_plan(development, active_config)
        mean_record, mean_oof = _fit_mean_cv(development, folds, active_config)
        ridge_records, ridge_predictions = _fit_ridge_cv(
            development_features, development, folds, active_config
        )
        xgboost_records, xgboost_predictions = _fit_xgboost_cv(
            development_features, development, folds, active_config
        )
        decision = _select_champion(ridge_records, xgboost_records)

        best_ridge = min(ridge_records, key=_metric_rank)
        best_xgboost = min(xgboost_records, key=_metric_rank)
        selected_predictions = {
            "development_target_mean": mean_oof,
            "ridge": ridge_predictions[float(best_ridge["alpha"])],
            "xgboost": xgboost_predictions[int(best_xgboost["candidate"])],
        }
        oof = pd.DataFrame(
            {
                "unit_id": development["unit_id"],
                "cycle": development["cycle"],
                "rul_true": development["rul_true"],
                "rul_target": development["rul_target"],
                **{
                    f"{name}_prediction": values
                    for name, values in selected_predictions.items()
                },
            }
        )

        model, preprocessor, model_hashes, model_spec = _fit_final_model(
            development_features,
            development,
            decision,
            active_config,
            temporary,
        )
        calibration_selected = _calibration_rows(calibration)
        calibration_features = build_causal_features(calibration)
        selected_indices = calibration_selected.set_index(["unit_id", "cycle"]).index
        calibration_index = pd.MultiIndex.from_frame(calibration[["unit_id", "cycle"]])
        selected_mask = calibration_index.isin(selected_indices)
        selected_features = calibration_features.loc[selected_mask].reset_index(
            drop=True
        )
        calibration_selected = (
            calibration.loc[selected_mask].copy().reset_index(drop=True)
        )
        cutoff_by_engine = {
            int(unit_id): CALIBRATION_CUTOFFS[index % len(CALIBRATION_CUTOFFS)]
            for index, unit_id in enumerate(
                sorted(calibration_selected["unit_id"].unique())
            )
        }
        calibration_selected["calibration_cutoff"] = calibration_selected[
            "unit_id"
        ].map(cutoff_by_engine)
        calibration_prediction = _predict_model(
            model,
            preprocessor,
            selected_features,
            decision["champion"],
        )
        scores = np.abs(
            calibration_prediction
            - calibration_selected["rul_true"].to_numpy(dtype=np.float64)
        )
        q = finite_sample_quantile(scores, active_config.nominal_interval_coverage)
        calibration_output = calibration_selected.loc[
            :,
            ["unit_id", "cycle", "rul_true", "calibration_cutoff"],
        ].copy()
        calibration_output["prediction"] = calibration_prediction
        calibration_output["absolute_error"] = scores
        calibration_output["interval_low"] = np.maximum(0.0, calibration_prediction - q)
        calibration_output["interval_high"] = calibration_prediction + q

        explanation_sample = selected_features.iloc[: min(200, len(selected_features))]
        explanation = _build_explanation(
            model,
            preprocessor,
            explanation_sample,
            calibration_selected.iloc[0],
            decision["champion"],
        )
        feature_manifest = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "causal": True,
            "windows": [5, 10, 20],
            "statistics": [
                "mean",
                "standard deviation",
                "minimum",
                "maximum",
                "linear slope",
                "last minus mean",
            ],
            "feature_count": len(FEATURE_NAMES),
            "feature_order": list(FEATURE_NAMES),
            "final_preprocessing": preprocessor.manifest(),
        }
        comparison = {
            "shared_protocol": {
                "selection_scope": "development engines only",
                "folds": active_config.group_folds,
                "seed": active_config.seed,
                "engine_equal_total_sample_weight": True,
                "metrics": [
                    "MAE",
                    "RMSE",
                    "motor-normalized NASA score",
                    "signed bias",
                    "overprediction rate",
                    "critical RUL precision/recall/F1",
                    "RUL-band MAE",
                ],
            },
            "development_target_mean": mean_record,
            "ridge_candidates": ridge_records,
            "xgboost_candidates": xgboost_records,
            "champion_decision": decision,
        }
        calibration_summary = {
            "scope": "20 calibration engines only; one score per engine",
            "coverage_label": (
                f"nominal {active_config.nominal_interval_coverage:.0%} "
                "empirical prediction interval; not a safety guarantee"
            ),
            "coverage": active_config.nominal_interval_coverage,
            "scores": len(scores),
            "finite_sample_rank": min(
                len(scores),
                math.ceil((len(scores) + 1) * active_config.nominal_interval_coverage),
            ),
            "q": q,
            "risk_band_source": "interval_low",
            "risk_bands": {
                "critical": "<=30",
                "elevated": "31-60",
                "routine": ">60",
            },
        }

        _write_json(temporary / "cv_folds.json", fold_records)
        _write_json(temporary / "feature_manifest.json", feature_manifest)
        _write_json(temporary / "model_comparison.json", comparison)
        _write_json(temporary / "champion_decision.json", decision)
        _write_new_bytes(temporary / "oof_predictions.parquet", _parquet_bytes(oof))
        _write_new_bytes(
            temporary / "calibration_scores.parquet",
            _parquet_bytes(calibration_output),
        )
        _write_json(temporary / "calibration_summary.json", calibration_summary)
        _write_json(temporary / "explanation.json", explanation)

        generated_hashes = {
            path.name: sha256_file(path)
            for path in temporary.iterdir()
            if path.is_file() and path.name != "model_lock.json"
        }
        generated_hashes.update(model_hashes)
        processed = root / "data" / "processed" / "fd001"
        config_path = root / "configs" / "project.yaml"
        lock = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "run_id": run_id,
            "dataset": "FD001",
            "seed": active_config.seed,
            "selection_scope": "development engines only",
            "development_engine_ids": split_manifest["development_engine_ids"],
            "calibration_engine_ids": split_manifest["calibration_engine_ids"],
            "data": {
                "manifest_sha256": sha256_file(processed / "manifest.json"),
                "split_manifest_sha256": sha256_file(processed / "split_manifest.json"),
                "train_sha256": sha256_file(processed / "train.parquet"),
                "test_sha256": data_manifest["artifacts"]["test.parquet"],
                "official_test_rul_sha256": data_manifest["artifacts"][
                    "evaluation/test_rul.parquet"
                ],
            },
            "config": {
                "path": "configs/project.yaml",
                "sha256": sha256_file(config_path),
                "modeling": asdict(active_config),
            },
            "features": {
                "order": list(FEATURE_NAMES),
                "feature_manifest_sha256": generated_hashes["feature_manifest.json"],
            },
            "champion": model_spec,
            "champion_rule_evidence": decision,
            "calibration": {
                "coverage": active_config.nominal_interval_coverage,
                "label": "nominal empirical interval; not a safety guarantee",
                "q": q,
                "score_count": len(scores),
                "scores_sha256": generated_hashes["calibration_scores.parquet"],
            },
            "artifacts": generated_hashes,
            "official_test_evaluation": "not opened during training or calibration",
        }
        _write_json(temporary / "model_lock.json", lock)
        run_manifest = {
            "run_id": run_id,
            "status": "model_locked",
            "seed": active_config.seed,
            "data_manifest_sha256": lock["data"]["manifest_sha256"],
            "config_sha256": lock["config"]["sha256"],
            "model_lock_sha256": sha256_file(temporary / "model_lock.json"),
            "champion": decision["champion"],
            "development_metrics": decision[decision["champion"]]["metrics"],
            "calibration_q": q,
        }
        _write_json(temporary / "manifest.json", run_manifest)
        temporary.replace(final_run_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return TrainResult(
        run_id=run_id,
        run_dir=final_run_dir,
        champion=decision["champion"],
        calibration_q=q,
        model_lock=final_run_dir / "model_lock.json",
    )


def _resolve_locked_artifact(run_dir: Path, relative_path: str) -> Path:
    path = (run_dir / relative_path).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ModelingError("Model lock references an external artifact") from exc
    if not path.is_file():
        raise ModelingError(f"Locked artifact is missing: {relative_path}")
    return path


def _validate_lock(
    project_root: Path,
    run_id: str,
) -> tuple[Path, dict[str, Any]]:
    run_dir = (project_root / "runs" / run_id).resolve()
    lock_path = run_dir / "model_lock.json"
    if not lock_path.is_file():
        raise ModelingError("Official evaluation requires model_lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("run_id") != run_id or lock.get("schema_version") != 1:
        raise ModelingError("Model lock identity or schema is invalid")

    processed = project_root / "data" / "processed" / "fd001"
    current_hashes = {
        "manifest_sha256": sha256_file(processed / "manifest.json"),
        "split_manifest_sha256": sha256_file(processed / "split_manifest.json"),
        "train_sha256": sha256_file(processed / "train.parquet"),
        "test_sha256": sha256_file(processed / "test.parquet"),
        "official_test_rul_sha256": sha256_file(
            processed / "evaluation" / "test_rul.parquet"
        ),
    }
    for name, observed in current_hashes.items():
        if lock["data"].get(name) != observed:
            raise ModelingError(f"Model lock data hash mismatch: {name}")
    if lock["config"]["sha256"] != sha256_file(
        project_root / "configs" / "project.yaml"
    ):
        raise ModelingError("Model lock config hash mismatch")
    if tuple(lock["features"]["order"]) != FEATURE_NAMES:
        raise ModelingError("Model lock feature order mismatch")
    for relative_path, expected_hash in lock["artifacts"].items():
        artifact = _resolve_locked_artifact(run_dir, relative_path)
        if sha256_file(artifact) != expected_hash:
            raise ModelingError(f"Locked artifact hash mismatch: {relative_path}")
    return run_dir, lock


def _load_locked_model(
    run_dir: Path,
    lock: dict[str, Any],
) -> tuple[Any, FoldPreprocessor, str]:
    champion = lock["champion"]
    kind = champion["kind"]
    if kind == "ridge":
        bundle = joblib.load(_resolve_locked_artifact(run_dir, champion["path"]))
        if not isinstance(bundle, RidgeBundle):
            raise ModelingError("Trusted local Ridge artifact has wrong type")
        return bundle, bundle.preprocessor, kind
    if kind == "xgboost":
        preprocessor = joblib.load(
            _resolve_locked_artifact(run_dir, champion["preprocessor_path"])
        )
        if not isinstance(preprocessor, FoldPreprocessor):
            raise ModelingError("Trusted local preprocessor has wrong type")
        model = XGBRegressor()
        model.load_model(_resolve_locked_artifact(run_dir, champion["path"]))
        return model, preprocessor, kind
    raise ModelingError(f"Unsupported locked champion kind: {kind}")


def evaluate_locked(
    project_root: Path,
    *,
    run_id: str,
) -> EvaluationResult:
    """Evaluate a verified lock against official labels without changing it."""
    root = project_root.resolve()
    run_dir, lock = _validate_lock(root, run_id)
    model, preprocessor, champion = _load_locked_model(run_dir, lock)

    processed = root / "data" / "processed" / "fd001"
    test = pd.read_parquet(processed / "test.parquet")
    test_features = build_causal_features(test)
    final_mask = (
        test.groupby("unit_id", sort=False)["cycle"].transform("max").eq(test["cycle"])
    )
    final_rows = test.loc[final_mask].reset_index(drop=True)
    final_features = test_features.loc[final_mask].reset_index(drop=True)
    prediction = _predict_model(model, preprocessor, final_features, champion)

    # Official labels are semantically opened only after lock and model validation.
    labels = pd.read_parquet(processed / "evaluation" / "test_rul.parquet")
    evaluation = final_rows.loc[:, ["unit_id", "cycle"]].merge(
        labels,
        on="unit_id",
        validate="one_to_one",
    )
    evaluation["prediction"] = prediction
    q = float(lock["calibration"]["q"])
    evaluation["interval_low"] = np.maximum(0.0, prediction - q)
    evaluation["interval_high"] = prediction + q
    evaluation["risk_band"] = np.select(
        [
            evaluation["interval_low"] <= 30,
            evaluation["interval_low"] <= 60,
        ],
        ["critical", "elevated"],
        default="routine",
    )
    evaluation["interval_contains_true_rul"] = evaluation["rul_true"].between(
        evaluation["interval_low"], evaluation["interval_high"]
    )
    metrics = prediction_metrics(
        evaluation["rul_true"].to_numpy(),
        evaluation["prediction"].to_numpy(),
        evaluation["unit_id"].to_numpy(),
        critical_threshold=float(lock["config"]["modeling"]["critical_rul_threshold"]),
    )
    metrics["nominal_empirical_interval"] = {
        "nominal_coverage": lock["calibration"]["coverage"],
        "observed_official_test_coverage": float(
            evaluation["interval_contains_true_rul"].mean()
        ),
        "mean_width": float(
            (evaluation["interval_high"] - evaluation["interval_low"]).mean()
        ),
        "label": "empirical prediction interval; not a safety guarantee",
    }
    errors = evaluation.assign(
        absolute_error=lambda frame: (frame["prediction"] - frame["rul_true"]).abs(),
        signed_error=lambda frame: frame["prediction"] - frame["rul_true"],
    ).sort_values(["absolute_error", "unit_id"], ascending=[False, True])
    error_analysis = {
        "largest_absolute_errors": errors.head(20).to_dict(orient="records"),
        "overpredicted_engines": int(errors["signed_error"].gt(0).sum()),
        "underpredicted_engines": int(errors["signed_error"].lt(0).sum()),
        "note": "Official results did not alter model, features, thresholds, or q",
    }

    output_dir = run_dir / "official_test"
    prediction_payload = _parquet_bytes(evaluation)
    metrics_payload = _json_bytes(metrics)
    errors_payload = _json_bytes(error_analysis)
    _write_new_bytes(output_dir / "predictions.parquet", prediction_payload)
    _write_new_bytes(output_dir / "metrics.json", metrics_payload)
    _write_new_bytes(output_dir / "error_analysis.json", errors_payload)
    evaluation_manifest = {
        "run_id": run_id,
        "model_lock_sha256": sha256_file(run_dir / "model_lock.json"),
        "predictions_sha256": hashlib.sha256(prediction_payload).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_payload).hexdigest(),
        "error_analysis_sha256": hashlib.sha256(errors_payload).hexdigest(),
        "champion_unchanged": lock["champion"]["kind"],
        "official_labels_usage": "locked evaluation only",
    }
    _write_json(output_dir / "evaluation_manifest.json", evaluation_manifest)
    return EvaluationResult(
        run_id=run_id,
        output_dir=output_dir,
        metrics=metrics,
    )
