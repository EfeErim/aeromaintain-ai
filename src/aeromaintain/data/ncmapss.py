"""Measured-only N-CMAPSS cycle summaries and locked Ridge simulation pilots."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeromaintain.data.pipeline import sha256_file
from aeromaintain.evaluation.ledger import claim_holdout
from aeromaintain.evaluation.protocol import draw_engine_rows
from aeromaintain.evaluation.v4 import interval_metrics, point_metrics
from aeromaintain.models.rul import engine_equal_weights
from aeromaintain.models.v4 import write_json

MEASURED_CHANNELS = (
    "T24",
    "T30",
    "T48",
    "T50",
    "P15",
    "P2",
    "P21",
    "P24",
    "Ps30",
    "P40",
    "P50",
    "Nf",
    "Nc",
    "Wf",
)
OPERATING_CHANNELS = ("alt", "Mach", "TRA", "T2")


def _names(dataset):
    return tuple(
        x.decode() if isinstance(x, bytes) else str(x)
        for x in np.asarray(dataset).ravel()
    )


def _h5py():
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("Install aeromaintain-ai[ncmapss] to read HDF5") from error
    return h5py


def cycle_features(path: Path, *, chunk_size=100_000) -> pd.DataFrame:
    """Read only A unit/cycle, W and X_s; never read Y, X_v, T, Fc or hs."""
    summaries = []
    with _h5py().File(path, "r") as handle:
        if (
            _names(handle["X_s_var"]) != MEASURED_CHANNELS
            or _names(handle["W_var"]) != OPERATING_CHANNELS
        ):
            raise ValueError("Measured channel names or order do not match protocol")
        auxiliary = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in np.asarray(handle["A_var"]).ravel()
        ]
        unit_col, cycle_col = auxiliary.index("unit"), auxiliary.index("cycle")
        for split in ("dev", "test"):
            xs, settings, aux = (
                handle[f"{name}_{split}"] for name in ("X_s", "W", "A")
            )
            if (
                xs.shape[1] != 14
                or settings.shape[1] != 4
                or len(xs) != len(aux)
                or len(xs) != len(settings)
            ):
                raise ValueError("N-CMAPSS measured-sensor schema mismatch")
            names = [f"sensor_{i}" for i in range(14)] + [
                f"setting_{i}" for i in range(4)
            ]
            chunks = []
            for start in range(0, len(xs), chunk_size):
                stop = min(len(xs), start + chunk_size)
                data = np.column_stack([xs[start:stop], settings[start:stop]])
                if not np.isfinite(data).all():
                    raise ValueError("Non-finite measured values")
                frame = pd.DataFrame(data, columns=names)
                frame["unit_id"] = aux[start:stop, unit_col].astype(int)
                frame["cycle"] = aux[start:stop, cycle_col].astype(int)
                grouped = frame.groupby(["unit_id", "cycle"])[names]
                stats = grouped.agg(["count", "sum", "min", "max"])
                squares = frame.copy()
                squares[names] = squares[names] ** 2
                squared_sums = squares.groupby(["unit_id", "cycle"])[names].sum()
                for name in names:
                    stats[(name, "squares")] = squared_sums[name]
                chunks.append(stats)
            combined = pd.concat(chunks)
            groups = combined.groupby(level=[0, 1])
            aggregated = pd.DataFrame(index=groups.size().index)
            for name in names:
                count = combined[(name, "count")].groupby(level=[0, 1]).sum()
                total = combined[(name, "sum")].groupby(level=[0, 1]).sum()
                squares = combined[(name, "squares")].groupby(level=[0, 1]).sum()
                aggregated[f"{name}_mean"] = total / count
                aggregated[f"{name}_std"] = np.sqrt(
                    np.maximum(0, squares / count - (total / count) ** 2)
                )
                aggregated[f"{name}_min"] = (
                    combined[(name, "min")].groupby(level=[0, 1]).min()
                )
                aggregated[f"{name}_max"] = (
                    combined[(name, "max")].groupby(level=[0, 1]).max()
                )
            aggregated = aggregated.reset_index()
            aggregated["source_split"] = split
            summaries.append(aggregated)
    if set(summaries[0].unit_id) & set(summaries[1].unit_id):
        raise ValueError("Engine IDs overlap across archive splits")
    return pd.concat(summaries, ignore_index=True)


def read_targets(path: Path, requested: pd.DataFrame) -> np.ndarray:
    """Read Y only at requested engine/cycle positions, after caller's holdout claim."""
    result = {}
    with _h5py().File(path, "r") as handle:
        auxiliary = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in np.asarray(handle["A_var"]).ravel()
        ]
        unit_col, cycle_col = auxiliary.index("unit"), auxiliary.index("cycle")
        for split, rows in requested.groupby("source_split"):
            aux = handle[f"A_{split}"]
            units = np.asarray(aux[:, unit_col]).astype(int)
            cycles = np.asarray(aux[:, cycle_col]).astype(int)
            positions_by_cycle = (
                pd.DataFrame(
                    {"unit": units, "cycle": cycles, "position": np.arange(len(units))}
                )
                .groupby(["unit", "cycle"])["position"]
                .agg(["min", "max"])
            )
            for row in rows.itertuples():
                key = (row.unit_id, row.cycle)
                if key not in positions_by_cycle.index:
                    raise ValueError("Requested cycle has no HDF5 rows")
                positions = positions_by_cycle.loc[key]
                # One RUL label per flight cycle; check both endpoints.
                target = float(
                    np.asarray(handle[f"Y_{split}"][int(positions["min"])]).item()
                )
                last = float(
                    np.asarray(handle[f"Y_{split}"][int(positions["max"])]).item()
                )
                if target != last or not np.isfinite(target) or target < 0:
                    raise ValueError("Invalid or nonconstant cycle RUL")
                result[(split, row.unit_id, row.cycle)] = target
    return np.array(
        [result[(r.source_split, r.unit_id, r.cycle)] for r in requested.itertuples()]
    )


def read_multiple_targets(paths, requested):
    """Resolve namespaced engines back to each immutable archive member."""
    result = pd.Series(index=requested.index, dtype=float)
    for path in paths:
        selected = requested[requested.source_file == path.name].copy()
        if selected.empty:
            continue
        selected["unit_id"] = selected.archive_unit_id
        result.loc[selected.index] = read_targets(path, selected)
    if result.isna().any():
        raise ValueError("Missing source file for requested engine")
    return result.to_numpy()


def run_pilot(paths: list[Path], output: Path, root: Path, *, confirmatory=False):
    """Lock fit/calibration evidence before opening any evaluation engine labels."""
    output.mkdir(parents=True, exist_ok=False)
    frames = []
    for number, path in enumerate(sorted(paths)):
        print(f"Preparing measured cycle features: {path.name}", flush=True)
        member = cycle_features(path)
        member["source_file"] = path.name
        member["archive_unit_id"] = member.unit_id
        member["unit_id"] = number * 1000 + member.unit_id
        frames.append(member)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(output / "cycle_features.parquet", index=False)
    if confirmatory:
        fit_units, cal_units, eval_units = [], [], []
        for _, member in frame.groupby("source_file", sort=True):
            units = np.sort(member.unit_id.unique())
            if len(units) < 10:
                raise ValueError("Confirmatory subset requires at least ten engines")
            permuted = np.random.default_rng(42).permutation(units)
            eval_units.extend(permuted[:4])
            cal_units.extend(permuted[4:7])
            fit_units.extend(permuted[7:])
        fit_units, cal_units, eval_units = map(
            np.asarray, (fit_units, cal_units, eval_units)
        )
        if len(eval_units) != 20 or len(cal_units) != 15:
            raise ValueError("Frozen mixed-subset protocol requires five subsets")
    else:
        fit_units = frame.loc[frame.source_split == "dev", "unit_id"].unique()
        cal_units = np.array([], dtype=int)
        eval_units = frame.loc[frame.source_split == "test", "unit_id"].unique()
    fit = frame[frame.unit_id.isin(fit_units)].copy()
    ev = frame[frame.unit_id.isin(eval_units)].copy()
    columns = [
        name for name in frame.columns if name.startswith(("sensor_", "setting_"))
    ]
    target = read_multiple_targets(paths, fit)
    weights = engine_equal_weights(fit.unit_id.to_numpy())
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1)
    )
    model.fit(fit[columns], target, ridge__sample_weight=weights)
    baseline = float(np.average(target, weights=weights))
    margin = None
    if confirmatory:
        cal = frame[frame.unit_id.isin(cal_units)].reset_index(drop=True)
        cuts = draw_engine_rows(
            cal.unit_id.to_numpy(), np.ones(len(cal), dtype=bool), draws=1, seed=45
        )[0]
        sampled = cal.iloc[cuts]
        residual = np.abs(
            read_multiple_targets(paths, sampled)
            - np.maximum(0, model.predict(sampled[columns]))
        )
        margin = float(np.sort(residual)[math.ceil((len(residual) + 1) * 0.9) - 1])
    joblib.dump(model, output / "model.joblib")
    lock = {
        "protocol_sha256": sha256_file(root / "docs/protocol_v4.md"),
        "data_sha256": {path.name: sha256_file(path) for path in paths},
        "data_filenames": [path.name for path in paths],
        "model_sha256": sha256_file(output / "model.joblib"),
        "features": columns,
        "fit_engines": sorted(fit_units.tolist()),
        "calibration_engines": sorted(cal_units.tolist()),
        "evaluation_engines": sorted(eval_units.tolist()),
        "split_margin": margin,
        "baseline": baseline,
        "scope": "independent simulation validation"
        if confirmatory
        else "DS02 descriptive simulation pilot",
    }
    write_json(output / "model_lock.json", lock)
    published = root / "docs/ncmapss_locks"
    published.mkdir(exist_ok=True)
    published_lock = published / ("confirmatory.json" if confirmatory else "ds02.json")
    with published_lock.open("xb") as handle:
        handle.write((output / "model_lock.json").read_bytes())
    claim_holdout(
        root / "runs/ncmapss_holdout_ledger.json",
        "DS03-DS07" if confirmatory else "DS02",
        {
            "model_lock_sha256": sha256_file(output / "model_lock.json"),
            "protocol_sha256": lock["protocol_sha256"],
            "data_sha256": lock["data_sha256"],
        },
    )
    truth = read_multiple_targets(paths, ev)
    prediction = np.maximum(0, model.predict(ev[columns]))
    plan = draw_engine_rows(
        ev.unit_id.to_numpy(), np.ones(len(ev), dtype=bool), draws=200, seed=46
    )
    report = {
        "lock": lock,
        "point": point_metrics(truth[plan], prediction[plan]),
        "baseline": point_metrics(truth[plan], np.full_like(truth[plan], baseline)),
    }
    ev["rul_true"], ev["prediction"] = truth, prediction
    if margin is not None:
        low, high = np.maximum(0, prediction - margin), prediction + margin
        cuts = plan.ravel()
        report["interval"] = interval_metrics(
            truth[cuts], low[cuts], high[cuts], ev.unit_id.to_numpy()[cuts]
        )
        # Engine-column bootstrap, all cuts for each resampled engine retained.
        rng = np.random.default_rng(44)
        columns_boot = rng.integers(0, len(eval_units), size=(2000, len(eval_units)))
        squared = ((prediction[plan] - truth[plan]) ** 2).mean(axis=0)
        covered = ((truth[plan] >= low[plan]) & (truth[plan] <= high[plan])).mean(
            axis=0
        )
        report["intervals_95"] = {
            "rmse": np.percentile(
                np.sqrt(squared[columns_boot].mean(axis=1)), [2.5, 97.5]
            ).tolist(),
            "coverage": np.percentile(
                covered[columns_boot].mean(axis=1), [2.5, 97.5]
            ).tolist(),
        }
        report["gates"] = {
            "coverage_at_least_85_percent": report["interval"]["coverage"] >= 0.85,
            "rmse_not_worse_than_mean": report["point"]["rmse"]
            <= report["baseline"]["rmse"],
        }
        ev["interval_low"], ev["interval_high"] = low, high
    ev.to_parquet(output / "evaluation.parquet", index=False)
    write_json(output / "evidence.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--confirmatory", action="store_true")
    args = parser.parse_args()
    run_pilot(
        args.source, args.output, args.project_root, confirmatory=args.confirmatory
    )
