"""Load a verified locked C-MAPSS model and predict with it.

Used by official evaluation. The lock must already be verified
(``models.evaluation.validate_lock``): these functions open files the lock lists, and
``joblib`` files run code when loaded, so they are for this project's own runs only.

``predict_model`` and ``predict_quantiles`` repeat two short functions in the frozen
``models/rul.py``. Training keeps using the frozen copies;
``tests/test_model_boundary.py`` checks that both give identical predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from aeromaintain.artifacts import ArtifactValidationError, resolve_within
from aeromaintain.evaluation.protocol import apply_mondrian_cqr, apply_mondrian_interval
from aeromaintain.features import (
    ConditionNormalizer,
    FoldPreprocessor,
    build_causal_features,
    build_condition_features,
)
from aeromaintain.models.rul import ModelingError, RidgeBundle


def resolve_locked_artifact(run_dir: Path, relative_path: str) -> Path:
    try:
        path = resolve_within(run_dir, relative_path, "Model lock")
    except ArtifactValidationError as exc:
        raise ModelingError(str(exc)) from exc
    if not path.is_file():
        raise ModelingError(f"Locked artifact is missing: {relative_path}")
    return path


def load_locked_model(
    run_dir: Path,
    lock: dict[str, Any],
) -> tuple[Any, FoldPreprocessor, str]:
    """Return the champion, its preprocessor and its kind (``ridge``/``xgboost``)."""
    champion = lock["champion"]
    kind = champion["kind"]
    if kind == "ridge":
        bundle = joblib.load(resolve_locked_artifact(run_dir, champion["path"]))
        if not isinstance(bundle, RidgeBundle):
            raise ModelingError("Trusted local Ridge artifact has wrong type")
        return bundle, bundle.preprocessor, kind
    if kind == "xgboost":
        preprocessor = joblib.load(
            resolve_locked_artifact(run_dir, champion["preprocessor_path"])
        )
        if not isinstance(preprocessor, FoldPreprocessor):
            raise ModelingError("Trusted local preprocessor has wrong type")
        members = []
        for path in champion.get("member_paths", [champion["path"]]):
            member = XGBRegressor()
            member.load_model(resolve_locked_artifact(run_dir, path))
            members.append(member)
        return (members[0] if len(members) == 1 else members), preprocessor, kind
    raise ModelingError(f"Unsupported locked champion kind: {kind}")


def locked_features(
    run_dir: Path,
    lock: dict[str, Any],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build features exactly as training did, with the locked normalizer if any."""
    path = lock["champion"].get("normalizer_path")
    if path is None:
        return build_causal_features(frame)
    normalizer = joblib.load(resolve_locked_artifact(run_dir, path))
    if not isinstance(normalizer, ConditionNormalizer):
        raise ModelingError("Trusted local normalizer has wrong type")
    return build_condition_features(frame, normalizer)


def predict_model(
    model: Any,
    preprocessor: FoldPreprocessor,
    features: pd.DataFrame,
    champion: str,
) -> np.ndarray:
    """Non-negative RUL; an XGBoost ensemble is the mean of its members."""
    view = features.loc[:, list(preprocessor.input_features)]
    if champion == "ridge":
        return model.predict(view)
    transformed = preprocessor.transform(view)
    members = model if isinstance(model, list) else [model]
    return np.mean(
        [np.maximum(0.0, member.predict(transformed)) for member in members],
        axis=0,
    )


def predict_quantiles(
    model: XGBRegressor,
    preprocessor: FoldPreprocessor,
    features: pd.DataFrame,
) -> np.ndarray:
    """Return an ``n x 2`` array of lower and upper quantile predictions."""
    view = features.loc[:, list(preprocessor.input_features)]
    return np.asarray(model.predict(preprocessor.transform(view)), dtype=np.float64)


def locked_interval(
    run_dir: Path,
    calibration: dict[str, Any],
    prediction: np.ndarray,
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the locked Mondrian or CQR interval to ``prediction``."""
    if "quantile_model_path" not in calibration:
        return apply_mondrian_interval(prediction, calibration)
    model = XGBRegressor()
    model.load_model(
        resolve_locked_artifact(run_dir, calibration["quantile_model_path"])
    )
    preprocessor = joblib.load(
        resolve_locked_artifact(run_dir, calibration["quantile_preprocessor_path"])
    )
    if not isinstance(preprocessor, FoldPreprocessor):
        raise ModelingError("Trusted local quantile preprocessor has wrong type")
    quantiles = predict_quantiles(model, preprocessor, features)
    return apply_mondrian_cqr(prediction, quantiles, calibration)
