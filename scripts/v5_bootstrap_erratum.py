"""Recompute the v5 final-test bootstrap intervals with every engine resampled.

The frozen ``models.v5.engine_bootstrap`` draws only engines 0 and 1, so the
published 95% ranges in ``docs/protocol_v5_evidence.json`` are wrong. This script
reads the stored v5 evaluation (no labels are opened again and no model is refit),
rebuilds the same 200 cuts per engine with the same seed, checks that the point
metrics reproduce the published evidence, and writes corrected ranges.

Needs ``runs/v5/final``, which holds NASA rows and is therefore not committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aeromaintain.evaluation.bootstrap import engine_bootstrap
from aeromaintain.evaluation.protocol import draw_engine_rows
from aeromaintain.models import v5

# The same values v5.evaluate used, so the cuts and resamples match the original run.
CUT_SEED = 46
BOOTSTRAP_SEED = 44
EVIDENCE = "docs/protocol_v5_evidence.json"


def recompute(root: Path, run_dir: Path) -> dict:
    config = json.loads((root / v5.CONFIG).read_text(encoding="utf-8"))
    lock = json.loads((run_dir / "model_lock.json").read_text(encoding="utf-8"))
    published = json.loads((root / EVIDENCE).read_text(encoding="utf-8"))
    published = published["part_b_final_test"]
    path = run_dir / "evaluation.parquet"
    frame = pd.read_parquet(path)

    plan = draw_engine_rows(
        frame.unit_id.to_numpy(),
        np.ones(len(frame), dtype=bool),
        draws=config["draws"],
        seed=CUT_SEED,
    )
    truth = frame.rul_true.to_numpy(dtype=float)[plan]
    prediction = frame.prediction.to_numpy(dtype=float)[plan]
    low = frame.mondrian_low.to_numpy(dtype=float)[plan]
    high = frame.mondrian_high.to_numpy(dtype=float)[plan]

    squared = ((prediction - truth) ** 2).mean(axis=0)
    squared_baseline = ((lock["baseline"] - truth) ** 2).mean(axis=0)
    covered = ((truth >= low) & (truth <= high)).mean(axis=0)
    point = {
        "rmse": float(np.sqrt(squared.mean())),
        "baseline_rmse": float(np.sqrt(squared_baseline.mean())),
        "mondrian_coverage": float(covered.mean()),
    }
    expected = {
        "rmse": published["point"]["rmse"],
        "baseline_rmse": published["baseline"]["rmse"],
        "mondrian_coverage": published["intervals"]["mondrian"]["overall"]["coverage"],
    }
    for name, value in point.items():
        if not np.isclose(value, expected[name]):
            raise ValueError(f"{name} {value} does not reproduce the evidence")

    corrected = engine_bootstrap(
        {
            "rmse": (squared, np.sqrt),
            "baseline_rmse": (squared_baseline, np.sqrt),
            "mondrian_coverage": (covered, lambda v: v),
        },
        config["bootstrap_resamples"],
        seed=BOOTSTRAP_SEED,
    )
    resampled = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(squared), size=(config["bootstrap_resamples"], len(squared))
    )
    model_beats_baseline = float(
        np.mean(
            squared[resampled].mean(axis=1) < squared_baseline[resampled].mean(axis=1)
        )
    )
    return {
        "status": "erratum; recomputed from stored v5 evaluation, labels not reopened",
        "defect": (
            "models.v5.engine_bootstrap took the engine count from the "
            "(values, transform) tuple, so it resampled engines 0 and 1 only"
        ),
        "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "engines": len(squared),
        "method": (
            f"engine-level percentile bootstrap, {config['bootstrap_resamples']} "
            f"resamples, seed {BOOTSTRAP_SEED}; cuts seed {CUT_SEED}"
        ),
        "point": point,
        "published_bootstrap_95": published["bootstrap_95"],
        "corrected_bootstrap_95": corrected,
        "resamples_where_model_beats_baseline": model_beats_baseline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path())
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v5/final"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/v5_bootstrap_erratum.json")
    )
    args = parser.parse_args()
    report = recompute(args.root, args.run_dir)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for name, (low, high) in report["corrected_bootstrap_95"].items():
        old_low, old_high = report["published_bootstrap_95"][name]
        print(
            f"{name}: {report['point'][name]:.4f}  "
            f"published {old_low:.4f}-{old_high:.4f}  corrected {low:.4f}-{high:.4f}"
        )


if __name__ == "__main__":
    main()
