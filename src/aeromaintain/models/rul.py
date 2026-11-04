"""Protocol v2 RUL selection, calibration, locking, and official evaluation."""

from __future__ import annotations

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
from aeromaintain.evaluation.protocol import (
    draw_engine_rows,
    eligible_rows,
    fit_mondrian_interval,
    leave_engines_out_interval,
    risk_threshold_table,
    select_risk_threshold,
    summarize_draws,
    summarize_interval_draws,
)
from aeromaintain.features import (
    FEATURE_SETS,
    FoldPreprocessor,
    build_causal_features,
)
from aeromaintain.runtime import runtime_fingerprint

RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1_000.0, 10_000.0)
RISK_THRESHOLD_GRID = tuple(float(value) for value in range(0, 65, 5))
ELEVATED_BAND_MARGIN = 30.0
MODEL_SCHEMA_VERSION = 2
PROTOCOL_DOCUMENT = "docs/protocol_v2.md"
SELECTION_SCOPE = "development engines only"
TEST_PREVIOUSLY_SEEN = "previously_seen"
TEST_CLEAN_SINGLE_OPEN = "clean_single_open"
TEST_STATUS_LABELS = {
    TEST_PREVIOUSLY_SEEN: "protocol v2, test previously seen",
    TEST_CLEAN_SINGLE_OPEN: "protocol v2, clean single-open test",
}


class ModelingError(RuntimeError):
    """Raised when a modeling or lock contract fails."""


@dataclass(frozen=True)
class ModelingConfig:
    """Fixed protocol v2 values loaded from the project contract."""

    seed: int = 42
    group_folds: int = 5
    critical_rul_threshold: int = 30
    nominal_interval_coverage: float = 0.90
    xgboost_candidates: int = 12
    xgboost_max_trees: int = 1500
    xgboost_early_stopping_rounds: int = 75
    xgboost_n_jobs: int = 8
    scoring_draws: int = 200
    scoring_rul_window: int | None = 150
    sensitivity_windows: tuple[int | None, ...] = (130, 175, None)
    inner_validation_fraction: float = 0.2
    rul_cap_candidates: tuple[int, ...] = (125, 130)
    stage_min_improvement: float = 0.01
    champion_min_improvement: float = 0.05
    rmse_target: float = 16.6
    cross_fit_folds: int = 10
    interval_band_edges: tuple[float, ...] = (30.0, 60.0, 100.0)
    interval_min_band_engines: int = 20
    risk_recall_target: float = 0.95
    bootstrap_resamples: int = 2_000


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
    interval_half_width: float
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


def _load_project_config(project_root: Path) -> dict[str, Any]:
    config_path = project_root / "configs" / "project.yaml"
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ModelingError(f"Invalid modeling configuration: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelingError("Invalid modeling configuration: expected a mapping")
    return payload


def load_modeling_config(project_root: Path) -> ModelingConfig:
    """Load the shared seed, folds, threshold, and protocol v2 parameters."""
    payload = _load_project_config(project_root)
    defaults = ModelingConfig()
    try:
        project = payload["project"]
        modeling = payload["modeling"]
        windows = modeling.get("sensitivity_windows", defaults.sensitivity_windows)
        return ModelingConfig(
            seed=int(project["seed"]),
            group_folds=int(modeling["group_folds"]),
            critical_rul_threshold=int(modeling["critical_rul_threshold"]),
            nominal_interval_coverage=float(modeling["nominal_interval_coverage"]),
            xgboost_n_jobs=int(modeling.get("xgboost_n_jobs", defaults.xgboost_n_jobs)),
            scoring_draws=int(modeling.get("scoring_draws", defaults.scoring_draws)),
            scoring_rul_window=modeling.get(
                "scoring_rul_window", defaults.scoring_rul_window
            ),
            sensitivity_windows=tuple(windows),
            inner_validation_fraction=float(
                modeling.get(
                    "inner_validation_fraction", defaults.inner_validation_fraction
                )
            ),
            rul_cap_candidates=tuple(
                int(value)
                for value in modeling.get(
                    "rul_cap_candidates", defaults.rul_cap_candidates
                )
            ),
            cross_fit_folds=int(
                modeling.get("cross_fit_folds", defaults.cross_fit_folds)
            ),
            interval_band_edges=tuple(
                float(value)
                for value in modeling.get(
                    "interval_band_edges", defaults.interval_band_edges
                )
            ),
            interval_min_band_engines=int(
                modeling.get(
                    "interval_min_band_engines", defaults.interval_min_band_engines
                )
            ),
            risk_recall_target=float(
                modeling.get("risk_recall_target", defaults.risk_recall_target)
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelingError(f"Invalid modeling configuration: {exc}") from exc


def load_test_status(project_root: Path, dataset: str) -> str:
    """Return whether a subset's test labels were already seen."""
    payload = _load_project_config(project_root)
    datasets = payload.get("data", {}).get("datasets", {})
    status = datasets.get(dataset, {}).get("test_status", TEST_PREVIOUSLY_SEEN)
    if status not in TEST_STATUS_LABELS:
        raise ModelingError(f"Unknown test status for {dataset}: {status}")
    return status


def engine_equal_weights(engine_ids: np.ndarray) -> np.ndarray:
    """Give every engine equal total influence while keeping mean weight one."""
    engines = pd.Series(np.asarray(engine_ids))
    counts = engines.map(engines.value_counts()).to_numpy(dtype=np.float64)
    return len(engines) / (engines.nunique() * counts)


def _processed_dir(project_root: Path, dataset: str) -> Path:
    return project_root / "data" / "processed" / dataset.lower()


def _load_and_validate_training_inputs(
    project_root: Path,
    dataset: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    processed = _processed_dir(project_root, dataset)
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
        raise ModelingError("Prepared train table is missing model contract columns")
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


def group_fold_plan(
    engine_ids: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return shuffled engine-grouped train/validation row positions."""
    engines = np.asarray(engine_ids)
    splitter = GroupKFold(
        n_splits=min(folds, int(pd.Series(engines).nunique())),
        shuffle=True,
        random_state=seed,
    )
    plan = list(splitter.split(engines, groups=engines))
    for train_index, validation_index in plan:
        if set(engines[train_index]) & set(engines[validation_index]):
            raise ModelingError("Engine leakage detected in a grouped fold")
    return plan


def inner_engine_split(
    engine_ids: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole engines for early stopping inside one training fold."""
    engines = np.asarray(engine_ids)
    unique = np.unique(engines)
    if len(unique) < 2:
        raise ModelingError("Early stopping requires at least two training engines")
    generator = np.random.default_rng(seed)
    count = min(len(unique) - 1, max(1, math.ceil(fraction * len(unique))))
    held_out = generator.choice(unique, size=count, replace=False)
    mask = np.isin(engines, held_out)
    return np.flatnonzero(~mask), np.flatnonzero(mask)


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


def _new_xgboost(
    params: dict[str, Any],
    config: ModelingConfig,
    *,
    seed: int,
    n_estimators: int,
    early_stopping: bool,
) -> XGBRegressor:
    return XGBRegressor(
        **params,
        objective="reg:squarederror",
        n_estimators=n_estimators,
        tree_method="hist",
        random_state=seed,
        n_jobs=config.xgboost_n_jobs,
        eval_metric="rmse",
        early_stopping_rounds=(
            config.xgboost_early_stopping_rounds if early_stopping else None
        ),
    )


def _capped_target(frame: pd.DataFrame, cap: int) -> np.ndarray:
    return frame["rul_true"].clip(upper=cap).to_numpy(dtype=np.float64)


def _feature_view(features: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    return features.loc[:, list(FEATURE_SETS[feature_set])]


def _new_preprocessor(spec: dict[str, Any]) -> FoldPreprocessor:
    return FoldPreprocessor(
        scale=spec["family"] == "ridge",
        input_features=FEATURE_SETS[spec["feature_set"]],
    )


def _fit_ridge(
    spec: dict[str, Any],
    features: pd.DataFrame,
    target: np.ndarray,
    engines: np.ndarray,
) -> RidgeBundle:
    preprocessor = _new_preprocessor(spec).fit(features)
    estimator = Ridge(alpha=float(spec["alpha"]))
    estimator.fit(
        preprocessor.transform(features),
        target,
        sample_weight=engine_equal_weights(engines),
    )
    return RidgeBundle(preprocessor=preprocessor, estimator=estimator)


def _fit_early_stopped_xgboost(
    spec: dict[str, Any],
    features: pd.DataFrame,
    target: np.ndarray,
    engines: np.ndarray,
    config: ModelingConfig,
    *,
    seed: int,
) -> tuple[XGBRegressor, FoldPreprocessor, int]:
    fit_rows, stop_rows = inner_engine_split(
        engines,
        fraction=config.inner_validation_fraction,
        seed=seed,
    )
    preprocessor = _new_preprocessor(spec).fit(features.iloc[fit_rows])
    estimator = _new_xgboost(
        spec["parameters"],
        config,
        seed=seed,
        n_estimators=config.xgboost_max_trees,
        early_stopping=True,
    )
    estimator.fit(
        preprocessor.transform(features.iloc[fit_rows]),
        target[fit_rows],
        sample_weight=engine_equal_weights(engines[fit_rows]),
        eval_set=[
            (preprocessor.transform(features.iloc[stop_rows]), target[stop_rows])
        ],
        sample_weight_eval_set=[engine_equal_weights(engines[stop_rows])],
        verbose=False,
    )
    return estimator, preprocessor, int(estimator.best_iteration) + 1


def _fit_fixed_xgboost(
    spec: dict[str, Any],
    features: pd.DataFrame,
    target: np.ndarray,
    engines: np.ndarray,
    config: ModelingConfig,
    *,
    seed: int,
) -> tuple[XGBRegressor, FoldPreprocessor]:
    preprocessor = _new_preprocessor(spec).fit(features)
    estimator = _new_xgboost(
        spec["parameters"],
        config,
        seed=seed,
        n_estimators=int(spec["tree_count"]),
        early_stopping=False,
    )
    estimator.fit(
        preprocessor.transform(features),
        target,
        sample_weight=engine_equal_weights(engines),
        verbose=False,
    )
    return estimator, preprocessor


@dataclass
class _SelectionData:
    """Development engines shared by every candidate in one selection run."""

    frame: pd.DataFrame
    features: pd.DataFrame
    folds: list[tuple[np.ndarray, np.ndarray]]
    plan: np.ndarray
    config: ModelingConfig

    @property
    def truth(self) -> np.ndarray:
        return self.frame["rul_true"].to_numpy(dtype=np.float64)

    @property
    def engines(self) -> np.ndarray:
        return self.frame["unit_id"].to_numpy()


def _candidate_oof(
    spec: dict[str, Any],
    data: _SelectionData,
) -> tuple[np.ndarray, list[int]]:
    features = _feature_view(data.features, spec["feature_set"])
    target = _capped_target(data.frame, int(spec["cap"]))
    engines = data.engines
    oof = np.empty(len(features), dtype=np.float64)
    tree_counts: list[int] = []
    for fold_index, (train_index, validation_index) in enumerate(data.folds):
        train_features = features.iloc[train_index]
        validation_features = features.iloc[validation_index]
        if spec["family"] == "ridge":
            bundle = _fit_ridge(
                spec, train_features, target[train_index], engines[train_index]
            )
            oof[validation_index] = bundle.predict(validation_features)
            continue
        estimator, preprocessor, tree_count = _fit_early_stopped_xgboost(
            spec,
            train_features,
            target[train_index],
            engines[train_index],
            data.config,
            seed=data.config.seed + fold_index,
        )
        tree_counts.append(tree_count)
        oof[validation_index] = np.maximum(
            0.0, estimator.predict(preprocessor.transform(validation_features))
        )
    return oof, tree_counts


def _evaluate_candidate(spec: dict[str, Any], data: _SelectionData) -> dict[str, Any]:
    oof, tree_counts = _candidate_oof(spec, data)
    summary, rmse_by_draw = summarize_draws(
        data.truth,
        oof,
        data.engines,
        data.plan,
        critical_threshold=data.config.critical_rul_threshold,
    )
    record: dict[str, Any] = {
        "spec": dict(spec),
        "metrics": summary,
        "fold_tree_counts": tree_counts,
    }
    if tree_counts:
        record["spec"]["tree_count"] = int(np.median(tree_counts))
    # Kept out of the persisted comparison; used for pairing candidates.
    record["_oof"] = oof
    record["_rmse_by_draw"] = rmse_by_draw
    return record


def _rank(record: dict[str, Any]) -> tuple[float, float]:
    metrics = record["metrics"]
    return (
        metrics["rmse"]["median"],
        metrics["nasa_score_motor_normalized"]["median"],
    )


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _run_stage(
    current: dict[str, Any],
    alternatives: list[dict[str, Any]],
    data: _SelectionData,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Replace the simpler earlier choice only for a required RMSE improvement."""
    required = 1.0 - data.config.stage_min_improvement
    evaluated = [_evaluate_candidate(spec, data) for spec in alternatives]
    log: list[dict[str, Any]] = []
    baseline = current["metrics"]["rmse"]["median"]
    best = current
    for record in sorted(evaluated, key=_rank):
        improves = record["metrics"]["rmse"]["median"] <= baseline * required
        accepted = improves and best is current
        log.append(
            {
                "stage": stage,
                **_public_record(record),
                "baseline_rmse": baseline,
                "accepted": accepted,
            }
        )
        if accepted:
            best = record
    return best, log


def _select_family(
    candidates: list[dict[str, Any]],
    data: _SelectionData,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the three protocol stages for one model family."""
    records = [_evaluate_candidate(spec, data) for spec in candidates]
    best = min(records, key=_rank)
    stages: dict[str, Any] = {
        "models": [_public_record(record) for record in records],
    }
    feature_specs = [
        {**best["spec"], "feature_set": name}
        for name in FEATURE_SETS
        if name != best["spec"]["feature_set"]
    ]
    best, stages["features"] = _run_stage(best, feature_specs, data, "features")
    cap_specs = [
        {**best["spec"], "cap": cap}
        for cap in data.config.rul_cap_candidates
        if cap != best["spec"]["cap"]
    ]
    best, stages["cap"] = _run_stage(best, cap_specs, data, "cap")
    return best, stages


def _select_champion(
    ridge: dict[str, Any],
    xgboost: dict[str, Any],
    config: ModelingConfig,
) -> dict[str, Any]:
    ridge_rmse = ridge["metrics"]["rmse"]["median"]
    xgboost_rmse = xgboost["metrics"]["rmse"]["median"]
    improvement = (ridge_rmse - xgboost_rmse) / ridge_rmse
    nasa_not_worse = (
        xgboost["metrics"]["nasa_score_motor_normalized"]["median"]
        <= ridge["metrics"]["nasa_score_motor_normalized"]["median"]
    )
    qualifies = improvement >= config.champion_min_improvement and nasa_not_worse
    champion = "xgboost" if qualifies else "ridge"
    chosen = xgboost if qualifies else ridge
    return {
        "rule": (
            "XGBoost only if median development RMSE improves over Ridge by at "
            f"least {config.champion_min_improvement:.0%} and median NASA score "
            "does not worsen"
        ),
        "ridge": _public_record(ridge),
        "xgboost": _public_record(xgboost),
        "xgboost_rmse_improvement_fraction": improvement,
        "xgboost_nasa_not_worse": nasa_not_worse,
        "xgboost_qualifies": qualifies,
        "xgboost_win_rate_over_paired_draws": float(
            np.mean(xgboost["_rmse_by_draw"] < ridge["_rmse_by_draw"])
        ),
        "champion": champion,
        "rmse_target": config.rmse_target,
        "rmse_target_met": chosen["metrics"]["rmse"]["median"] <= config.rmse_target,
    }


def _window_sensitivity(
    ridge: dict[str, Any],
    xgboost: dict[str, Any],
    data: _SelectionData,
) -> list[dict[str, Any]]:
    """Recheck the champion under other eligibility windows, without acting on it."""
    config = data.config
    rows: list[dict[str, Any]] = []
    for window in config.sensitivity_windows:
        plan = draw_engine_rows(
            data.engines,
            eligible_rows(data.truth, window),
            draws=config.scoring_draws,
            seed=config.seed,
        )
        records = []
        for record in (ridge, xgboost):
            summary, rmse_by_draw = summarize_draws(
                data.truth,
                record["_oof"],
                data.engines,
                plan,
                critical_threshold=config.critical_rul_threshold,
            )
            records.append({"metrics": summary, "_rmse_by_draw": rmse_by_draw})
        decision = _select_champion(records[0], records[1], config)
        rows.append(
            {
                "window": window,
                "champion": decision["champion"],
                "ridge_median_rmse": records[0]["metrics"]["rmse"]["median"],
                "xgboost_median_rmse": records[1]["metrics"]["rmse"]["median"],
                "ridge_median_nasa": records[0]["metrics"][
                    "nasa_score_motor_normalized"
                ]["median"],
                "xgboost_median_nasa": records[1]["metrics"][
                    "nasa_score_motor_normalized"
                ]["median"],
            }
        )
    return rows


def _mean_baseline(data: _SelectionData) -> dict[str, Any]:
    target = _capped_target(data.frame, data.config.rul_cap_candidates[0])
    oof = np.empty(len(target), dtype=np.float64)
    for train_index, validation_index in data.folds:
        oof[validation_index] = np.average(
            target[train_index],
            weights=engine_equal_weights(data.engines[train_index]),
        )
    summary, _ = summarize_draws(
        data.truth,
        oof,
        data.engines,
        data.plan,
        critical_threshold=data.config.critical_rul_threshold,
    )
    return {"spec": {"family": "development_target_mean"}, "metrics": summary}


def fit_final_model(
    spec: dict[str, Any],
    features: pd.DataFrame,
    frame: pd.DataFrame,
    config: ModelingConfig,
    *,
    seed: int,
) -> tuple[Any, FoldPreprocessor]:
    """Fit locked settings on the given engines."""
    view = _feature_view(features, spec["feature_set"])
    target = _capped_target(frame, int(spec["cap"]))
    engines = frame["unit_id"].to_numpy()
    if spec["family"] == "ridge":
        bundle = _fit_ridge(spec, view, target, engines)
        return bundle, bundle.preprocessor
    return _fit_fixed_xgboost(spec, view, target, engines, config, seed=seed)


def _persist_model(
    spec: dict[str, Any],
    model: Any,
    preprocessor: FoldPreprocessor,
    run_dir: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    if spec["family"] == "ridge":
        model_path = run_dir / "model.joblib"
        joblib.dump(model, model_path)
        return (
            {"model.joblib": sha256_file(model_path)},
            {
                "kind": "ridge",
                "format": "trusted-local-joblib",
                "path": "model.joblib",
                "parameters": {"alpha": float(spec["alpha"])},
                "feature_set": spec["feature_set"],
                "cap": int(spec["cap"]),
            },
        )
    model_path = run_dir / "model.json"
    preprocessor_path = run_dir / "preprocessor.joblib"
    model.save_model(model_path)
    joblib.dump(preprocessor, preprocessor_path)
    return (
        {
            "model.json": sha256_file(model_path),
            "preprocessor.joblib": sha256_file(preprocessor_path),
        },
        {
            "kind": "xgboost",
            "format": "xgboost-json-with-trusted-local-joblib-preprocessor",
            "path": "model.json",
            "preprocessor_path": "preprocessor.joblib",
            "parameters": {
                **spec["parameters"],
                "n_estimators": int(spec["tree_count"]),
            },
            "feature_set": spec["feature_set"],
            "cap": int(spec["cap"]),
        },
    )


def _predict_model(
    model: Any,
    preprocessor: FoldPreprocessor,
    features: pd.DataFrame,
    champion: str,
) -> np.ndarray:
    view = features.loc[:, list(preprocessor.input_features)]
    if champion == "ridge":
        return model.predict(view)
    return np.maximum(0.0, model.predict(preprocessor.transform(view)))


def cross_fit_predictions(
    spec: dict[str, Any],
    features: pd.DataFrame,
    frame: pd.DataFrame,
    config: ModelingConfig,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return out-of-fold predictions and fold IDs over all training engines."""
    engines = frame["unit_id"].to_numpy()
    prediction = np.empty(len(frame), dtype=np.float64)
    fold_ids = np.empty(len(frame), dtype=np.int64)
    for fold_index, (train_index, validation_index) in enumerate(
        group_fold_plan(engines, folds=config.cross_fit_folds, seed=seed)
    ):
        model, preprocessor = fit_final_model(
            spec,
            features.iloc[train_index],
            frame.iloc[train_index],
            config,
            seed=seed + fold_index,
        )
        prediction[validation_index] = _predict_model(
            model,
            preprocessor,
            features.iloc[validation_index],
            spec["family"],
        )
        fold_ids[validation_index] = fold_index
    return prediction, fold_ids


def calibrate_interval(
    prediction: np.ndarray,
    fold_ids: np.ndarray,
    frame: pd.DataFrame,
    config: ModelingConfig,
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the Mondrian interval and choose the risk threshold from train engines."""
    truth = frame["rul_true"].to_numpy(dtype=np.float64)
    engines = frame["unit_id"].to_numpy()
    eligible = eligible_rows(truth, config.scoring_rul_window)
    interval = fit_mondrian_interval(
        prediction[eligible],
        truth[eligible],
        engines[eligible],
        coverage=config.nominal_interval_coverage,
        edges=config.interval_band_edges,
        min_engines=config.interval_min_band_engines,
    )
    low, high = leave_engines_out_interval(
        prediction,
        truth,
        engines,
        eligible,
        fold_ids,
        coverage=config.nominal_interval_coverage,
        edges=config.interval_band_edges,
        min_engines=config.interval_min_band_engines,
    )
    plan = draw_engine_rows(
        engines,
        eligible,
        draws=config.scoring_draws,
        seed=seed + 1,
    )
    table = risk_threshold_table(
        truth,
        low,
        plan,
        critical_threshold=config.critical_rul_threshold,
        thresholds=RISK_THRESHOLD_GRID,
    )
    selected = select_risk_threshold(table, recall_target=config.risk_recall_target)
    analysis = {
        "scope": "leave-engines-out over all training engines",
        "interval": summarize_interval_draws(
            truth,
            prediction,
            low,
            high,
            plan,
            edges=config.interval_band_edges,
        ),
        "risk_threshold_table": table,
        "selected_risk_threshold": selected,
    }
    half_widths = [(band["q_high"] - band["q_low"]) / 2 for band in interval["bands"]]
    calibration = {
        "method": "mondrian-signed-residual-cross-fit",
        "label": "nominal empirical interval; not a safety guarantee",
        "coverage": config.nominal_interval_coverage,
        "bands": interval["bands"],
        "engines": int(pd.Series(engines[eligible]).nunique()),
        "max_half_width": float(max(half_widths)),
        "risk_threshold": selected["threshold"],
        "elevated_threshold": selected["threshold"] + ELEVATED_BAND_MARGIN,
        "critical_rul_threshold": float(config.critical_rul_threshold),
    }
    return calibration, analysis


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
    dataset: str = "FD001",
) -> TrainResult:
    """Select on development engines, calibrate, and atomically lock a run."""
    root = project_root.resolve()
    if not run_id or Path(run_id).name != run_id:
        raise ModelingError("run_id must be one safe path component")
    active_config = config or load_modeling_config(root)
    test_status = load_test_status(root, dataset)
    train, data_manifest, split_manifest = _load_and_validate_training_inputs(
        root, dataset
    )
    runs_root = root / "runs"
    final_run_dir = runs_root / run_id
    if final_run_dir.exists():
        raise ModelingError(f"Run directory already exists: {final_run_dir}")
    runs_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=runs_root))

    try:
        all_frame = train.reset_index(drop=True)
        all_features = build_causal_features(all_frame)
        development_mask = all_frame["role"].eq("development").to_numpy()
        development = all_frame.loc[development_mask].reset_index(drop=True)
        development_features = all_features.loc[development_mask].reset_index(drop=True)
        engines = development["unit_id"].to_numpy()
        truth = development["rul_true"].to_numpy(dtype=np.float64)
        folds = group_fold_plan(
            engines, folds=active_config.group_folds, seed=active_config.seed
        )
        plan = draw_engine_rows(
            engines,
            eligible_rows(truth, active_config.scoring_rul_window),
            draws=active_config.scoring_draws,
            seed=active_config.seed,
        )
        data = _SelectionData(
            development, development_features, folds, plan, active_config
        )

        base = {
            "feature_set": "all",
            "cap": active_config.rul_cap_candidates[0],
        }
        best_ridge, ridge_stages = _select_family(
            [{**base, "family": "ridge", "alpha": alpha} for alpha in RIDGE_ALPHAS],
            data,
        )
        best_xgboost, xgboost_stages = _select_family(
            [
                {
                    **base,
                    "family": "xgboost",
                    "candidate": index,
                    "parameters": parameters,
                }
                for index, parameters in enumerate(
                    _xgboost_parameter_candidates(active_config), start=1
                )
            ],
            data,
        )
        decision = _select_champion(best_ridge, best_xgboost, active_config)
        decision["window_sensitivity"] = _window_sensitivity(
            best_ridge, best_xgboost, data
        )
        locked_spec = (
            best_xgboost if decision["champion"] == "xgboost" else best_ridge
        )["spec"]

        cross_fit, fold_ids = cross_fit_predictions(
            locked_spec,
            all_features,
            all_frame,
            active_config,
            seed=active_config.seed,
        )
        calibration, calibration_analysis = calibrate_interval(
            cross_fit,
            fold_ids,
            all_frame,
            active_config,
            seed=active_config.seed,
        )
        model, preprocessor = fit_final_model(
            locked_spec,
            all_features,
            all_frame,
            active_config,
            seed=active_config.seed,
        )
        model_hashes, model_spec = _persist_model(
            locked_spec, model, preprocessor, temporary
        )

        last_rows = (
            all_frame.groupby("unit_id", sort=False)["cycle"]
            .transform("max")
            .eq(all_frame["cycle"])
        )
        explanation_features = all_features.loc[
            last_rows, list(preprocessor.input_features)
        ].reset_index(drop=True)
        explanation = _build_explanation(
            model,
            preprocessor,
            explanation_features,
            all_frame.loc[last_rows].iloc[0],
            locked_spec["family"],
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
            "feature_set": locked_spec["feature_set"],
            "feature_count": len(preprocessor.input_features),
            "feature_order": list(preprocessor.input_features),
            "final_preprocessing": preprocessor.manifest(),
        }
        comparison = {
            "shared_protocol": {
                "selection_scope": SELECTION_SCOPE,
                "folds": active_config.group_folds,
                "seed": active_config.seed,
                "scoring": (
                    f"{active_config.scoring_draws} seeded draws of one eligible "
                    "row per engine, scored with the official metric function"
                ),
                "rul_window": active_config.scoring_rul_window,
                "early_stopping": "inner engine split inside each training fold",
            },
            "development_target_mean": _mean_baseline(data),
            "ridge_stages": ridge_stages,
            "xgboost_stages": xgboost_stages,
            "champion_decision": decision,
        }
        oof = pd.DataFrame(
            {
                "unit_id": development["unit_id"],
                "cycle": development["cycle"],
                "rul_true": development["rul_true"],
                "ridge_prediction": best_ridge["_oof"],
                "xgboost_prediction": best_xgboost["_oof"],
            }
        )
        cross_fit_output = pd.DataFrame(
            {
                "unit_id": all_frame["unit_id"],
                "cycle": all_frame["cycle"],
                "rul_true": all_frame["rul_true"],
                "role": all_frame["role"],
                "fold": fold_ids,
                "prediction": cross_fit,
            }
        )

        _write_json(temporary / "feature_manifest.json", feature_manifest)
        _write_json(temporary / "model_comparison.json", comparison)
        _write_json(temporary / "champion_decision.json", decision)
        _write_new_bytes(temporary / "oof_predictions.parquet", _parquet_bytes(oof))
        _write_new_bytes(
            temporary / "cross_fit_predictions.parquet",
            _parquet_bytes(cross_fit_output),
        )
        _write_json(temporary / "calibration_analysis.json", calibration_analysis)
        _write_json(temporary / "explanation.json", explanation)

        generated_hashes = {
            path.name: sha256_file(path)
            for path in temporary.iterdir()
            if path.is_file() and path.name != "model_lock.json"
        }
        generated_hashes.update(model_hashes)
        processed = _processed_dir(root, dataset)
        config_path = root / "configs" / "project.yaml"
        protocol_path = root / PROTOCOL_DOCUMENT
        environment = runtime_fingerprint(root)
        lock = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "protocol": {
                "version": 2,
                "document": PROTOCOL_DOCUMENT,
                "sha256": (
                    sha256_file(protocol_path) if protocol_path.is_file() else None
                ),
                "test_status": test_status,
                "label": TEST_STATUS_LABELS[test_status],
            },
            "run_id": run_id,
            "dataset": dataset,
            "seed": active_config.seed,
            "selection_scope": SELECTION_SCOPE,
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
                "set": locked_spec["feature_set"],
                "order": list(preprocessor.input_features),
                "feature_manifest_sha256": generated_hashes["feature_manifest.json"],
            },
            "champion": model_spec,
            "champion_rule_evidence": decision,
            "calibration": calibration,
            "environment": environment,
            "artifacts": generated_hashes,
            "official_test_evaluation": "not opened during training or calibration",
        }
        _write_json(temporary / "model_lock.json", lock)
        run_manifest = {
            "run_id": run_id,
            "dataset": dataset,
            "status": "model_locked",
            "seed": active_config.seed,
            "data_manifest_sha256": lock["data"]["manifest_sha256"],
            "config_sha256": lock["config"]["sha256"],
            "model_lock_sha256": sha256_file(temporary / "model_lock.json"),
            "champion": decision["champion"],
            "development_metrics": decision[decision["champion"]]["metrics"],
            "interval_max_half_width": calibration["max_half_width"],
            "environment": environment,
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
        interval_half_width=calibration["max_half_width"],
        model_lock=final_run_dir / "model_lock.json",
    )


def evaluate_locked(
    project_root: Path,
    *,
    run_id: str,
) -> EvaluationResult:
    """Compatibility wrapper for the separated official-evaluation module."""
    from aeromaintain.models.evaluation import evaluate_locked as _evaluate_locked

    return _evaluate_locked(project_root, run_id=run_id)
