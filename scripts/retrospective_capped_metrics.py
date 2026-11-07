"""Score stored v3 official-test predictions against capped true RUL.

Retrospective only: every C-MAPSS test label was already opened. No model is fitted or
changed; the stored predictions are re-scored so they can be compared with papers that
cap the test truth at 125 or 130 cycles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aeromaintain.evaluation.metrics import nasa_score

RUNS = {
    "FD001": ("fd001-v3", "test previously seen"),
    "FD002": ("fd002-v3-clean", "clean single-open test"),
    "FD003": ("fd003-v3", "test previously seen"),
    "FD004": ("fd004-v3-clean", "clean single-open test"),
}
CAPS = (None, 125, 130)
RESAMPLES = 2000
SEED = 42


def score(truth, prediction, engines):
    error = prediction - truth
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "nasa_score": nasa_score(truth, prediction, engines),
    }


def bootstrap(truth, prediction, engines):
    rng = np.random.default_rng(SEED)
    rmse, nasa = [], []
    for _ in range(RESAMPLES):
        pick = rng.integers(0, len(truth), len(truth))
        error = prediction[pick] - truth[pick]
        rmse.append(np.sqrt(np.mean(error**2)))
        nasa.append(nasa_score(truth[pick], prediction[pick], np.arange(len(pick))))
    return {
        "rmse": [float(v) for v in np.percentile(rmse, [2.5, 97.5])],
        "nasa_score": [float(v) for v in np.percentile(nasa, [2.5, 97.5])],
    }


def evaluate(root: Path):
    out = {}
    for dataset, (run, label) in RUNS.items():
        path = root / "runs" / run / "official_test" / "predictions.parquet"
        frame = pd.read_parquet(path)
        if frame["unit_id"].duplicated().any():
            raise ValueError(f"{path} must hold one row per engine")
        published = json.loads(
            (path.parent / "metrics.json").read_text(encoding="utf-8")
        )
        truth_raw = frame["rul_true"].to_numpy(dtype=np.float64)
        prediction = frame["prediction"].to_numpy(dtype=np.float64)
        engines = frame["unit_id"].to_numpy()
        entry = {
            "run": run,
            "label": label,
            "engines": len(frame),
            "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "truth": {},
        }
        for cap in CAPS:
            truth = truth_raw if cap is None else np.minimum(truth_raw, cap)
            record = score(truth, prediction, engines)
            record["bootstrap_95"] = bootstrap(truth, prediction, engines)
            record["engines_with_truth_above_cap"] = (
                0 if cap is None else int((truth_raw > cap).sum())
            )
            entry["truth"]["uncapped" if cap is None else f"capped_{cap}"] = record
        if not np.isclose(entry["truth"]["uncapped"]["rmse"], published["rmse"]):
            raise ValueError(
                f"{dataset}: uncapped RMSE does not reproduce metrics.json"
            )
        out[dataset] = entry
    return {
        "status": (
            "retrospective; all C-MAPSS test labels were opened before this analysis"
        ),
        "model_training_cap": 125,
        "method": {
            "prediction": "stored v3 official-test prediction, unchanged",
            "truth": "official RUL_FD00x truth, optionally min(truth, cap)",
            "nasa_score": "sum over engines of the asymmetric penalty (a1=13, a2=10)",
            "bootstrap": f"engine-level percentile, {RESAMPLES} resamples, seed {SEED}",
        },
        "datasets": out,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path())
    parser.add_argument(
        "--output", type=Path, default=Path("docs/retrospective_capped_evidence.json")
    )
    args = parser.parse_args()
    evidence = evaluate(args.root)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for dataset, entry in evidence["datasets"].items():
        cells = ", ".join(
            f"{name} RMSE {rec['rmse']:.2f} NASA {rec['nasa_score']:.0f}"
            for name, rec in entry["truth"].items()
        )
        print(dataset, cells)


if __name__ == "__main__":
    main()
