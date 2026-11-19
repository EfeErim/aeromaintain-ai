"""The line between frozen experiment code and the modules that are maintained.

``models/rul.py`` is pinned by the v5 lock, so a few of its helpers are repeated in
``files.py`` and ``models/inference.py`` for active code. These tests keep both
copies equivalent, and keep stored model files loadable.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from aeromaintain import files
from aeromaintain.features import ConditionNormalizer, FoldPreprocessor
from aeromaintain.models import inference, rul

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("a", "b", "c", "flat")


def _features(rows: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(rows, 3)), columns=list(COLUMNS[:3]))
    frame["flat"] = 1.0
    frame.loc[3, "b"] = np.nan
    return frame


def _fitted(scale: bool) -> tuple[FoldPreprocessor, pd.DataFrame, np.ndarray]:
    features = _features()
    target = 50 + 10 * features["a"].to_numpy() - 5 * features["c"].to_numpy()
    return (
        FoldPreprocessor(scale=scale, input_features=COLUMNS).fit(features),
        features,
        target,
    )


def test_run_file_helpers_match_frozen_copies(tmp_path: Path) -> None:
    value = {"b": [1, 2.5], "a": {"z": None, "y": "text"}}
    assert files.json_bytes(value) == rul._json_bytes(value)
    frame = pd.DataFrame({"unit_id": [1, 2], "x": [0.5, np.nan]})
    assert files.parquet_bytes(frame) == rul._parquet_bytes(frame)
    assert files.processed_dir(tmp_path, "FD002") == rul._processed_dir(
        tmp_path, "FD002"
    )

    def write_once(path: Path, payload: bytes) -> None:
        files.write_once(path, payload, rul.ModelingError)

    for name, writer in (("files", write_once), ("rul", rul._write_new_bytes)):
        path = tmp_path / name / "out.bin"
        writer(path, b"one")
        writer(path, b"one")
        with pytest.raises(rul.ModelingError, match="Refusing to overwrite"):
            writer(path, b"two")
        assert path.read_bytes() == b"one"


def test_ridge_prediction_matches_frozen_copy() -> None:
    preprocessor, features, target = _fitted(scale=True)
    bundle = rul.RidgeBundle(
        preprocessor=preprocessor,
        estimator=Ridge(alpha=1.0).fit(preprocessor.transform(features), target),
    )
    shuffled = features.loc[:, ["c", "a", "flat", "b"]]
    expected = rul._predict_model(bundle, preprocessor, shuffled, "ridge")
    np.testing.assert_array_equal(
        inference.predict_model(bundle, preprocessor, shuffled, "ridge"), expected
    )


def test_xgboost_ensemble_and_quantiles_match_frozen_copy() -> None:
    preprocessor, features, target = _fitted(scale=False)
    x = preprocessor.transform(features)
    members = [
        XGBRegressor(n_estimators=8, max_depth=2, random_state=seed).fit(x, target - 60)
        for seed in (1, 2)
    ]
    for model in (members[0], members):
        np.testing.assert_array_equal(
            inference.predict_model(model, preprocessor, features, "xgboost"),
            rul._predict_model(model, preprocessor, features, "xgboost"),
        )
    # Members are clipped at zero before averaging, in both copies.
    assert (
        inference.predict_model(members, preprocessor, features, "xgboost") >= 0
    ).all()

    quantile = XGBRegressor(
        n_estimators=8,
        objective="reg:quantileerror",
        quantile_alpha=np.array([0.05, 0.95]),
    ).fit(x, target)
    np.testing.assert_array_equal(
        inference.predict_quantiles(quantile, preprocessor, features),
        rul.predict_quantiles(quantile, preprocessor, features),
    )


def test_ridge_bundle_round_trips_through_locked_loader(tmp_path: Path) -> None:
    preprocessor, features, target = _fitted(scale=True)
    bundle = rul.RidgeBundle(
        preprocessor=preprocessor,
        estimator=Ridge(alpha=1.0).fit(preprocessor.transform(features), target),
    )
    joblib.dump(bundle, tmp_path / "model.joblib")
    lock = {"champion": {"kind": "ridge", "path": "model.joblib"}}
    model, loaded_preprocessor, kind = inference.load_locked_model(tmp_path, lock)
    assert kind == "ridge"
    np.testing.assert_array_equal(
        inference.predict_model(model, loaded_preprocessor, features, kind),
        bundle.predict(features),
    )

    joblib.dump({"not": "a bundle"}, tmp_path / "model.joblib")
    with pytest.raises(rul.ModelingError, match="wrong type"):
        inference.load_locked_model(tmp_path, lock)
    with pytest.raises(rul.ModelingError, match="external path"):
        inference.load_locked_model(
            tmp_path, {"champion": {"kind": "ridge", "path": "../model.joblib"}}
        )


def test_pickled_classes_keep_their_import_paths() -> None:
    # Stored .joblib files name these classes by module path.
    assert rul.RidgeBundle.__module__ == "aeromaintain.models.rul"
    assert FoldPreprocessor.__module__ == "aeromaintain.features.causal"
    assert ConditionNormalizer.__module__ == "aeromaintain.features.causal"


def test_stored_run_models_still_load() -> None:
    files = sorted((ROOT / "runs").glob("*/*.joblib"))
    if not files:
        pytest.skip("runs/ holds NASA-derived files and is not committed")
    project_classes = (rul.RidgeBundle, FoldPreprocessor, ConditionNormalizer)
    for path in files:
        loaded = joblib.load(path)
        if type(loaded).__module__.startswith("aeromaintain"):
            assert isinstance(loaded, project_classes), path
