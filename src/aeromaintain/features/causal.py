"""Leakage-safe feature generation and fold-local preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SETTING_COLUMNS = tuple(f"setting_{index}" for index in range(1, 4))
SENSOR_COLUMNS = tuple(f"sensor_{index}" for index in range(1, 22))
ROLLING_WINDOWS = (5, 10, 20)
ROLLING_STATISTICS = ("mean", "std", "min", "max", "slope", "delta_mean")


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
    """Build current-and-past-only FD001 rolling features."""
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
