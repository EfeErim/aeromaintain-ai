"""Leakage-safe feature generation and fold-local preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SETTING_COLUMNS = tuple(f"setting_{index}" for index in range(1, 4))
SENSOR_COLUMNS = tuple(f"sensor_{index}" for index in range(1, 22))
ROLLING_WINDOWS = (5, 10, 20)
ROLLING_STATISTICS = ("mean", "std", "min", "max", "slope", "delta_mean")
CONDITION_FEATURE = "cycles_since_condition_change"


class FeatureContractError(ValueError):
    """Raised when input data or persisted feature order violates the contract."""


def feature_names() -> tuple[str, ...]:
    """Return the stable model feature order."""
    names = [*SETTING_COLUMNS, *SENSOR_COLUMNS, "engine_age"]
    for window in ROLLING_WINDOWS:
        for sensor in SENSOR_COLUMNS:
            names.extend(
                f"{sensor}_w{window}_{statistic}" for statistic in ROLLING_STATISTICS
            )
    return tuple(names)


FEATURE_NAMES = feature_names()
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "all": FEATURE_NAMES,
    "no_age": tuple(name for name in FEATURE_NAMES if name != "engine_age"),
    "no_settings": tuple(name for name in FEATURE_NAMES if name not in SETTING_COLUMNS),
    "compact": tuple(
        name
        for name in FEATURE_NAMES
        if name in SENSOR_COLUMNS
        or name.endswith(("_w20_mean", "_w20_slope"))
        or name == "engine_age"
    ),
}


def _validate_source(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"unit_id", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FeatureContractError(f"Missing feature-source columns: {missing}")
    if frame.duplicated(["unit_id", "cycle"]).any():
        raise FeatureContractError("Feature source contains duplicate engine cycles")
    ordered = frame.sort_values(["unit_id", "cycle"], kind="stable")
    if not ordered.index.equals(frame.index):
        raise FeatureContractError(
            "Feature source must be ordered by unit_id and cycle"
        )
    cycle_steps = frame.groupby("unit_id", sort=False)["cycle"].diff().dropna()
    if not cycle_steps.eq(1).all():
        raise FeatureContractError("Engine cycles must be contiguous and increasing")
    return frame


def build_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build current-and-past-only rolling features."""
    source = _validate_source(frame)
    columns: dict[str, pd.Series] = {
        name: source[name].astype("float64")
        for name in (*SETTING_COLUMNS, *SENSOR_COLUMNS)
    }
    columns["engine_age"] = source["cycle"].astype("float64")

    groups = source.groupby("unit_id", sort=False)
    cycle = source["cycle"].astype("float64")
    for window in ROLLING_WINDOWS:
        cycle_roll = groups["cycle"].rolling(window, min_periods=1)
        count = cycle_roll.count().reset_index(level=0, drop=True).astype("float64")
        sum_x = cycle_roll.sum().reset_index(level=0, drop=True).astype("float64")
        sum_x2 = (
            cycle.pow(2)
            .groupby(source["unit_id"], sort=False)
            .rolling(window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        denominator = count * sum_x2 - sum_x.pow(2)

        for sensor in SENSOR_COLUMNS:
            rolling = groups[sensor].rolling(window, min_periods=1)
            mean = rolling.mean().reset_index(level=0, drop=True)
            standard_deviation = (
                rolling.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)
            )
            minimum = rolling.min().reset_index(level=0, drop=True)
            maximum = rolling.max().reset_index(level=0, drop=True)
            sum_y = rolling.sum().reset_index(level=0, drop=True)
            xy = cycle * source[sensor]
            sum_xy = (
                xy.groupby(source["unit_id"], sort=False)
                .rolling(window, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            numerator = count * sum_xy - sum_x * sum_y
            slope = numerator.div(denominator.where(denominator.ne(0))).fillna(0.0)

            prefix = f"{sensor}_w{window}"
            columns[f"{prefix}_mean"] = mean
            columns[f"{prefix}_std"] = standard_deviation
            columns[f"{prefix}_min"] = minimum
            columns[f"{prefix}_max"] = maximum
            columns[f"{prefix}_slope"] = slope
            columns[f"{prefix}_delta_mean"] = source[sensor] - mean

    result = pd.DataFrame(columns, index=source.index).loc[:, FEATURE_NAMES]
    if not np.isfinite(result.to_numpy()).all():
        raise FeatureContractError("Generated features contain non-finite values")
    return result


@dataclass
class ConditionNormalizer:
    """Condition clusters and per-condition sensor scales from one partition."""

    conditions: int
    seed: int
    max_within_spread: float = 0.1
    setting_mean: np.ndarray | None = None
    setting_scale: np.ndarray | None = None
    centroids: np.ndarray | None = None
    sensor_mean: np.ndarray | None = None
    sensor_scale: np.ndarray | None = None
    within_spread: float | None = None

    def fit(self, frame: pd.DataFrame) -> ConditionNormalizer:
        """Fit clusters and scales, refusing settings that do not separate."""
        settings = frame.loc[:, list(SETTING_COLUMNS)].to_numpy(dtype=np.float64)
        self.setting_mean = settings.mean(axis=0)
        scale = settings.std(axis=0)
        self.setting_scale = np.where(scale > 0, scale, 1.0)
        scaled = (settings - self.setting_mean) / self.setting_scale
        model = KMeans(n_clusters=self.conditions, n_init=10, random_state=self.seed)
        labels = model.fit_predict(scaled)
        # Sorting makes condition numbers independent of KMeans initialization.
        order = np.lexsort(model.cluster_centers_.T[::-1])
        self.centroids = model.cluster_centers_[order]
        labels = self._nearest(scaled)
        spreads = [
            float(np.abs(scaled[labels == index] - self.centroids[index]).max())
            for index in range(self.conditions)
            if (labels == index).any()
        ]
        self.within_spread = max(spreads)
        if len(spreads) != self.conditions or self.within_spread > (
            self.max_within_spread
        ):
            raise FeatureContractError(
                "Operating settings do not separate into "
                f"{self.conditions} conditions: spread {self.within_spread:.3f}"
            )
        sensors = frame.loc[:, list(SENSOR_COLUMNS)].to_numpy(dtype=np.float64)
        self.sensor_mean = np.vstack(
            [sensors[labels == index].mean(axis=0) for index in range(self.conditions)]
        )
        deviation = np.vstack(
            [sensors[labels == index].std(axis=0) for index in range(self.conditions)]
        )
        self.sensor_scale = np.where(deviation > 0, deviation, 1.0)
        return self

    def _nearest(self, scaled: np.ndarray) -> np.ndarray:
        distance = np.linalg.norm(
            scaled[:, np.newaxis, :] - self.centroids[np.newaxis, :, :],
            axis=2,
        )
        return distance.argmin(axis=1)

    def labels(self, frame: pd.DataFrame) -> np.ndarray:
        """Assign each row to its nearest fitted condition."""
        if self.centroids is None:
            raise FeatureContractError("Normalizer must be fitted before transform")
        settings = frame.loc[:, list(SETTING_COLUMNS)].to_numpy(dtype=np.float64)
        return self._nearest((settings - self.setting_mean) / self.setting_scale)

    def normalized_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy whose sensors are standardized within their condition."""
        labels = self.labels(frame)
        sensors = frame.loc[:, list(SENSOR_COLUMNS)].to_numpy(dtype=np.float64)
        result = frame.copy()
        result[list(SENSOR_COLUMNS)] = (
            sensors - self.sensor_mean[labels]
        ) / self.sensor_scale[labels]
        return result


def build_condition_features(
    frame: pd.DataFrame,
    normalizer: ConditionNormalizer,
) -> pd.DataFrame:
    """Build causal features on condition-normalized sensors."""
    features = build_causal_features(normalizer.normalized_frame(frame))
    labels = pd.Series(normalizer.labels(frame), index=frame.index)
    engines = frame["unit_id"]
    changed = labels.ne(labels.groupby(engines, sort=False).shift()) | (
        engines.ne(engines.shift())
    )
    run = changed.cumsum()
    features[CONDITION_FEATURE] = labels.groupby(run).cumcount().astype("float64")
    return features


@dataclass
class FoldPreprocessor:
    """Median imputation, training-fold constant removal and optional scaling."""

    scale: bool
    input_features: tuple[str, ...] = FEATURE_NAMES
    imputer: SimpleImputer | None = None
    scaler: StandardScaler | None = None
    constant_mask: np.ndarray | None = None
    output_features: tuple[str, ...] | None = None

    def fit(self, frame: pd.DataFrame) -> FoldPreprocessor:
        """Fit all data-dependent preprocessing on one training partition."""
        checked = self._ordered(frame)
        self.imputer = SimpleImputer(strategy="median")
        imputed = self.imputer.fit_transform(checked)
        self.constant_mask = np.ptp(imputed, axis=0) > 0.0
        if not self.constant_mask.any():
            raise FeatureContractError(
                "Training partition has no non-constant features"
            )
        self.output_features = tuple(
            name
            for name, keep in zip(self.input_features, self.constant_mask, strict=True)
            if keep
        )
        selected = imputed[:, self.constant_mask]
        self.scaler = StandardScaler() if self.scale else None
        if self.scaler is not None:
            self.scaler.fit(selected)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply persisted training-partition decisions in the same feature order."""
        if (
            self.imputer is None
            or self.constant_mask is None
            or self.output_features is None
        ):
            raise FeatureContractError("Preprocessor must be fitted before transform")
        selected = self.imputer.transform(self._ordered(frame))[:, self.constant_mask]
        transformed = (
            self.scaler.transform(selected) if self.scaler is not None else selected
        )
        return np.asarray(transformed, dtype=np.float64)

    def _ordered(self, frame: pd.DataFrame) -> pd.DataFrame:
        observed = tuple(frame.columns)
        if observed != self.input_features:
            raise FeatureContractError(
                "Feature order mismatch: "
                f"expected {len(self.input_features)} persisted features"
            )
        return frame

    def manifest(self) -> dict[str, Any]:
        """Return the persisted feature/preprocessing contract."""
        if self.output_features is None:
            raise FeatureContractError("Preprocessor has not been fitted")
        return {
            "input_features": list(self.input_features),
            "output_features": list(self.output_features),
            "constant_columns_removed": sorted(
                set(self.input_features).difference(self.output_features)
            ),
            "imputation": "training-partition median",
            "scaling": "training-partition standardization" if self.scale else "none",
        }
