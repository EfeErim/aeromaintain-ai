"""Protocol v2 Amendment 1: LSTM and CQR challengers, on training engines only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from aeromaintain.evaluation.protocol import (
    _weighted_quantile,
    draw_engine_rows,
    eligible_rows,
    summarize_draws,
    summarize_interval_draws,
)
from aeromaintain.features import FEATURE_SETS, build_causal_features
from aeromaintain.features.causal import SENSOR_COLUMNS, SETTING_COLUMNS
from aeromaintain.models.evaluation import _validate_lock
from aeromaintain.models.robustness import _config_from_lock
from aeromaintain.models.rul import (
    ModelingConfig,
    ModelingError,
    _feature_view,
    _new_preprocessor,
    _processed_dir,
    engine_equal_weights,
    group_fold_plan,
    inner_engine_split,
)

SEQUENCE_LENGTH = 30
SEQUENCE_COLUMNS = (*SETTING_COLUMNS, *SENSOR_COLUMNS)
LSTM_CONFIGURATIONS = (
    {"hidden_size": 64, "layers": 1, "dropout": 0.0},
    {"hidden_size": 64, "layers": 2, "dropout": 0.2},
    {"hidden_size": 128, "layers": 2, "dropout": 0.2},
)
LSTM_MAX_EPOCHS = 40
LSTM_PATIENCE = 6
LSTM_BATCH_SIZE = 256
LSTM_LEARNING_RATE = 1e-3
CQR_MIN_COVERAGE = 0.89
CQR_MIN_WIDTH_REDUCTION = 0.05


def build_sequences(
    frame: pd.DataFrame,
    *,
    length: int = SEQUENCE_LENGTH,
) -> np.ndarray:
    """Return past-and-current-only windows, left-padded with an engine's first row."""
    values = frame.loc[:, list(SEQUENCE_COLUMNS)].to_numpy(dtype=np.float32)
    engines = frame["unit_id"].to_numpy()
    sequences = np.empty((len(frame), length, values.shape[1]), dtype=np.float32)
    positions = pd.Series(np.arange(len(frame))).groupby(engines, sort=False)
    for _, group in positions:
        rows = group.to_numpy()
        for offset, row in enumerate(rows):
            window = rows[max(0, offset - length + 1) : offset + 1]
            padding = length - len(window)
            sequences[row, :padding] = values[rows[0]]
            sequences[row, padding:] = values[window]
    return sequences


def _scale_sequences(
    sequences: np.ndarray,
    fit_rows: np.ndarray,
) -> np.ndarray:
    # Only the current cycle of fitting rows sets scales, so no other engine leaks.
    current = sequences[fit_rows, -1, :]
    mean = current.mean(axis=0)
    deviation = current.std(axis=0)
    keep = deviation > 0
    return (sequences[:, :, keep] - mean[keep]) / deviation[keep]


def _fit_predict_lstm(
    sequences: np.ndarray,
    target: np.ndarray,
    engines: np.ndarray,
    train_rows: np.ndarray,
    predict_rows: np.ndarray,
    configuration: dict[str, Any],
    *,
    cap: float,
    seed: int,
    fraction: float,
) -> tuple[np.ndarray, int]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    inner_fit, inner_stop = inner_engine_split(
        engines[train_rows], fraction=fraction, seed=seed
    )
    fit_rows = train_rows[inner_fit]
    stop_rows = train_rows[inner_stop]
    scaled = torch.from_numpy(_scale_sequences(sequences, fit_rows))
    scaled_target = torch.from_numpy((target / cap).astype(np.float32))
    weights = np.zeros(len(target), dtype=np.float32)
    weights[fit_rows] = engine_equal_weights(engines[fit_rows])
    weights[stop_rows] = engine_equal_weights(engines[stop_rows])
    sample_weight = torch.from_numpy(weights)

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                scaled.shape[2],
                configuration["hidden_size"],
                num_layers=configuration["layers"],
                dropout=configuration["dropout"],
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(configuration["hidden_size"], 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            output, _ = self.lstm(batch)
            return self.head(output[:, -1, :]).squeeze(-1)

    def predict(model: nn.Module, rows: np.ndarray) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            return torch.cat(
                [
                    model(scaled[rows[start : start + 4096]])
                    for start in range(0, len(rows), 4096)
                ]
            )

    model = Network()
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LEARNING_RATE)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, LSTM_MAX_EPOCHS + 1):
        model.train()
        order = generator.permutation(fit_rows)
        for start in range(0, len(order), LSTM_BATCH_SIZE):
            batch = order[start : start + LSTM_BATCH_SIZE]
            optimizer.zero_grad()
            error = model(scaled[batch]) - scaled_target[batch]
            loss = (sample_weight[batch] * error.pow(2)).mean()
            loss.backward()
            optimizer.step()
        stop_error = predict(model, stop_rows) - scaled_target[stop_rows]
        stop_loss = float((sample_weight[stop_rows] * stop_error.pow(2)).mean())
        if stop_loss < best_loss:
            best_loss = stop_loss
            best_state = {
                key: value.clone() for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= LSTM_PATIENCE:
                break
    if best_state is None:
        raise ModelingError("LSTM training produced no usable epoch")
    model.load_state_dict(best_state)
    prediction = predict(model, predict_rows).numpy().astype(np.float64) * cap
    return np.maximum(0.0, prediction), best_epoch


def _load_base(
    root: Path,
    base_run_id: str,
) -> tuple[Path, dict[str, Any], ModelingConfig, pd.DataFrame]:
    run_dir, lock = _validate_lock(root, base_run_id)
    frame = pd.read_parquet(
        _processed_dir(root, lock["dataset"]) / "train.parquet"
    ).reset_index(drop=True)
    return run_dir, lock, _config_from_lock(lock), frame


def run_lstm_study(
    project_root: Path,
    *,
    base_run_id: str,
    configurations: tuple[dict[str, Any], ...] = LSTM_CONFIGURATIONS,
) -> dict[str, Any]:
    """Score LSTM configurations on the base run's folds and draws."""
    root = project_root.resolve()
    run_dir, lock, config, frame = _load_base(root, base_run_id)
    development = frame.loc[frame["role"].eq("development")].reset_index(drop=True)
    truth = development["rul_true"].to_numpy(dtype=np.float64)
    engines = development["unit_id"].to_numpy()
    cap = float(config.rul_cap_candidates[0])
    target = np.minimum(truth, cap)
    folds = group_fold_plan(engines, folds=config.group_folds, seed=config.seed)
    plan = draw_engine_rows(
        engines,
        eligible_rows(truth, config.scoring_rul_window),
        draws=config.scoring_draws,
        seed=config.seed,
    )
    baseline = pd.read_parquet(run_dir / "oof_predictions.parquet")
    baseline_summary, baseline_rmse = summarize_draws(
        truth,
        baseline["xgboost_prediction"].to_numpy(),
        engines,
        plan,
        critical_threshold=config.critical_rul_threshold,
    )
    sequences = build_sequences(development)
    records: list[dict[str, Any]] = []
    for configuration in configurations:
        oof = np.empty(len(development), dtype=np.float64)
        epochs: list[int] = []
        for fold_index, (train_rows, validation_rows) in enumerate(folds):
            oof[validation_rows], epoch = _fit_predict_lstm(
                sequences,
                target,
                engines,
                train_rows,
                validation_rows,
                configuration,
                cap=cap,
                seed=config.seed + fold_index,
                fraction=config.inner_validation_fraction,
            )
            epochs.append(epoch)
        summary, rmse = summarize_draws(
            truth,
            oof,
            engines,
            plan,
            critical_threshold=config.critical_rul_threshold,
        )
        records.append(
            {
                "configuration": configuration,
                "fold_best_epochs": epochs,
                "metrics": summary,
                "win_rate_over_xgboost_paired_draws": float(
                    np.mean(rmse < baseline_rmse)
                ),
            }
        )
    best = min(
        records,
        key=lambda item: (
            item["metrics"]["rmse"]["median"],
            item["metrics"]["nasa_score_motor_normalized"]["median"],
        ),
    )
    base_rmse = baseline_summary["rmse"]["median"]
    improvement = (base_rmse - best["metrics"]["rmse"]["median"]) / base_rmse
    nasa_not_worse = (
        best["metrics"]["nasa_score_motor_normalized"]["median"]
        <= baseline_summary["nasa_score_motor_normalized"]["median"]
    )
    return {
        "schema_version": 1,
        "scope": "development engines only; same folds and draws as the base run",
        "dataset": lock["dataset"],
        "base_run_id": base_run_id,
        "baseline_xgboost": baseline_summary,
        "records": records,
        "best_configuration": best["configuration"],
        "rmse_improvement_fraction": improvement,
        "nasa_not_worse": nasa_not_worse,
        "rule": (
            "LSTM replaces XGBoost only with at least 5% lower median RMSE and a "
            "median NASA score that is not worse"
        ),
        "lstm_replaces_xgboost": bool(improvement >= 0.05 and nasa_not_worse),
    }


def _fit_quantiles(
    spec: dict[str, Any],
    features: pd.DataFrame,
    target: np.ndarray,
    engines: np.ndarray,
    config: ModelingConfig,
    *,
    tail: float,
    seed: int,
) -> tuple[XGBRegressor, Any]:
    preprocessor = _new_preprocessor(spec).fit(features)
    model = XGBRegressor(
        **spec["parameters"],
        objective="reg:quantileerror",
        quantile_alpha=[tail, 1.0 - tail],
        n_estimators=int(spec["tree_count"]),
        tree_method="hist",
        random_state=seed,
        n_jobs=config.xgboost_n_jobs,
    )
    model.fit(
        preprocessor.transform(features),
        target,
        sample_weight=engine_equal_weights(engines),
        verbose=False,
    )
    return model, preprocessor


def _cqr_margin(
    score: np.ndarray,
    engines: np.ndarray,
    coverage: float,
) -> float:
    counts = pd.Series(engines).map(pd.Series(engines).value_counts())
    engine_count = int(pd.Series(engines).nunique())
    level = min(1.0, coverage * (engine_count + 1) / engine_count)
    return _weighted_quantile(score, 1.0 / counts.to_numpy(dtype=np.float64), level)


def run_cqr_study(
    project_root: Path,
    *,
    base_run_id: str,
) -> dict[str, Any]:
    """Compare CQR with the base run's Mondrian interval, leaving engines out."""
    root = project_root.resolve()
    run_dir, lock, config, frame = _load_base(root, base_run_id)
    decision = lock["champion_rule_evidence"]
    spec = decision[decision["champion"]]["spec"]
    if spec["family"] != "xgboost":
        raise ModelingError("The CQR study applies to an XGBoost champion")
    features = _feature_view(build_causal_features(frame), spec)
    truth = frame["rul_true"].to_numpy(dtype=np.float64)
    engines = frame["unit_id"].to_numpy()
    target = np.minimum(truth, float(spec["cap"]))
    cross_fit = pd.read_parquet(run_dir / "cross_fit_predictions.parquet")
    fold_ids = cross_fit["fold"].to_numpy()
    tail = (1.0 - config.nominal_interval_coverage) / 2.0

    quantiles = np.empty((len(frame), 2), dtype=np.float64)
    for fold in np.unique(fold_ids):
        held_out = fold_ids == fold
        model, preprocessor = _fit_quantiles(
            spec,
            features.loc[~held_out],
            target[~held_out],
            engines[~held_out],
            config,
            tail=tail,
            seed=config.seed + int(fold),
        )
        quantiles[held_out] = model.predict(
            preprocessor.transform(features.loc[held_out])
        )
    lower_quantile = quantiles.min(axis=1)
    upper_quantile = quantiles.max(axis=1)
    score = np.maximum(lower_quantile - truth, truth - upper_quantile)
    eligible = eligible_rows(truth, config.scoring_rul_window)

    low = np.empty(len(frame), dtype=np.float64)
    high = np.empty(len(frame), dtype=np.float64)
    for fold in np.unique(fold_ids):
        held_out = fold_ids == fold
        calibrating = ~held_out & eligible
        margin = _cqr_margin(
            score[calibrating],
            engines[calibrating],
            config.nominal_interval_coverage,
        )
        low[held_out] = np.maximum(0.0, lower_quantile[held_out] - margin)
        high[held_out] = upper_quantile[held_out] + margin

    plan = draw_engine_rows(
        engines,
        eligible,
        draws=config.scoring_draws,
        seed=config.seed + 1,
    )
    report = summarize_interval_draws(
        truth,
        cross_fit["prediction"].to_numpy(),
        low,
        high,
        plan,
        edges=config.interval_band_edges,
    )
    mondrian = json.loads(
        (run_dir / "calibration_analysis.json").read_text(encoding="utf-8")
    )["interval"]
    reduction = 1.0 - report["median_mean_width"] / mondrian["median_mean_width"]
    return {
        "schema_version": 1,
        "scope": "all training engines; leave-engines-out; test labels not opened",
        "dataset": lock["dataset"],
        "base_run_id": base_run_id,
        "cqr": report,
        "mondrian": mondrian,
        "width_reduction_fraction": reduction,
        "rule": (
            f"CQR replaces Mondrian only with coverage >= {CQR_MIN_COVERAGE:.0%} and "
            f"mean width at least {CQR_MIN_WIDTH_REDUCTION:.0%} lower"
        ),
        "cqr_replaces_mondrian": bool(
            report["median_coverage"] >= CQR_MIN_COVERAGE
            and reduction >= CQR_MIN_WIDTH_REDUCTION
        ),
    }


__all__ = [
    "FEATURE_SETS",
    "build_sequences",
    "run_cqr_study",
    "run_lstm_study",
]
