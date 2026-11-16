"""Sensitivity checks on the v5 final test that the frozen protocol did not report.

Reads the stored v5 evaluation in ``runs/v5/final`` (no labels are opened again, no
model is refit) and writes ``docs/v5_sensitivity.json`` with:

- a Wilson 95% interval for the G3 miss rate;
- a stricter alarm reading: G3 counts an engine as alerted whenever a confirmed
  alarm is still active inside the 10-30 cycle window, even if it was confirmed
  earlier and has no false-alarm limit. Here the alarm must be newly confirmed
  inside the window, and separately, alerted with no false event;
- file-level sensitivity: leave-one-file-out point metrics and a two-stage
  bootstrap (resample files, then engines within each file), because the 35
  engines come from three files with opposite bias.

None of this changes a published gate decision; the gates were fixed in advance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aeromaintain.evaluation.metrics import wilson_interval
from aeromaintain.evaluation.protocol import draw_engine_rows
from aeromaintain.evaluation.v4 import replay_alarms
from aeromaintain.models import v5

# The same seeds v5.evaluate and the bootstrap erratum used.
CUT_SEED = 46
BOOTSTRAP_SEED = 44
EVIDENCE = "docs/protocol_v5_evidence.json"


def alarm_events(
    history: pd.DataFrame, values: np.ndarray, persistence: int, threshold: float
):
    """Yield the true RUL at each cycle where an alarm is newly confirmed."""
    streak, previous = 0, None
    for cycle, rul, value in zip(
        history.cycle.to_numpy(), history.rul_true.to_numpy(), values, strict=True
    ):
        if previous is not None and cycle != previous + 1:
            streak = 0
        streak = streak + 1 if value <= threshold else 0
        previous = cycle
        if streak == persistence:
            yield float(rul)


def strict_alarm(frame: pd.DataFrame, config: dict) -> dict:
    persistence = config["persistence_cycles"]
    threshold = config["alarm_threshold"]
    low = config["timely_min_rul"]
    in_window, clean = 0, 0
    per_engine = []
    ordered = frame.sort_values(["unit_id", "cycle"])
    for unit, history in ordered.groupby("unit_id", sort=True):
        events = list(
            alarm_events(history, history.prediction.to_numpy(), persistence, threshold)
        )
        fresh = any(low <= rul <= threshold for rul in events)
        false = sum(rul > threshold for rul in events)
        in_window += fresh
        clean += fresh and false == 0
        per_engine.append(
            {"unit_id": int(unit), "confirmations": events, "fresh_in_window": fresh}
        )
    engines = len(per_engine)
    return {
        "engines": engines,
        "newly_confirmed_in_window": in_window,
        "newly_confirmed_in_window_without_false_event": clean,
        "missed_if_alarm_must_start_in_window": engines - in_window,
        "missed_if_false_events_also_count": engines - clean,
        "records": per_engine,
    }


def per_engine_errors(frame: pd.DataFrame, baseline: float, draws: int):
    plan = draw_engine_rows(
        frame.unit_id.to_numpy(),
        np.ones(len(frame), dtype=bool),
        draws=draws,
        seed=CUT_SEED,
    )
    truth = frame.rul_true.to_numpy(dtype=float)[plan]
    prediction = frame.prediction.to_numpy(dtype=float)[plan]
    low = frame.mondrian_low.to_numpy(dtype=float)[plan]
    high = frame.mondrian_high.to_numpy(dtype=float)[plan]
    engines = np.sort(frame.unit_id.unique())
    files = frame.groupby("unit_id").source_file.first().loc[engines].to_numpy()
    return {
        "engines": engines,
        "files": files,
        "squared": ((prediction - truth) ** 2).mean(axis=0),
        "squared_baseline": ((baseline - truth) ** 2).mean(axis=0),
        "covered": ((truth >= low) & (truth <= high)).mean(axis=0),
    }


def summarize(errors: dict, index: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(errors["squared"][index].mean())),
        "baseline_rmse": float(np.sqrt(errors["squared_baseline"][index].mean())),
        "mondrian_coverage": float(errors["covered"][index].mean()),
    }


def two_stage_bootstrap(errors: dict, resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    names = np.unique(errors["files"])
    members = {name: np.flatnonzero(errors["files"] == name) for name in names}
    draws = {"rmse": [], "baseline_rmse": [], "mondrian_coverage": []}
    beats = 0
    for _ in range(resamples):
        chosen = rng.choice(names, size=len(names), replace=True)
        index = np.concatenate(
            [rng.choice(members[n], size=len(members[n]), replace=True) for n in chosen]
        )
        values = summarize(errors, index)
        for key, value in values.items():
            draws[key].append(value)
        beats += values["rmse"] < values["baseline_rmse"]
    return {
        "ranges_95": {
            key: np.percentile(values, [2.5, 97.5]).astype(float).tolist()
            for key, values in draws.items()
        },
        "resamples_where_model_beats_baseline": beats / resamples,
    }


def recompute(root: Path, run_dir: Path) -> dict:
    config = json.loads((root / v5.CONFIG).read_text(encoding="utf-8"))
    lock = json.loads((run_dir / "model_lock.json").read_text(encoding="utf-8"))
    published = json.loads((root / EVIDENCE).read_text(encoding="utf-8"))
    published = published["part_b_final_test"]
    path = run_dir / "evaluation.parquet"
    frame = pd.read_parquet(path)

    replay = replay_alarms(
        frame[["unit_id", "cycle", "rul_true"]],
        frame.prediction.to_numpy(dtype=float),
        persistence=config["persistence_cycles"],
        threshold=config["alarm_threshold"],
        min_rul=config["timely_min_rul"],
    )
    if (
        replay["missed_engines"]
        != published["replay"]["point_timely"]["missed_engines"]
    ):
        raise ValueError("Stored evaluation does not reproduce the published G3")
    missed = replay["missed_engines"]
    engines = replay["engines"]

    errors = per_engine_errors(frame, lock["baseline"], config["draws"])
    overall = summarize(errors, np.arange(len(errors["engines"])))
    if not np.isclose(overall["rmse"], published["point"]["rmse"]):
        raise ValueError("Stored evaluation does not reproduce the published RMSE")

    leave_one_out = {
        name: summarize(errors, np.flatnonzero(errors["files"] != name))
        for name in np.unique(errors["files"])
    }
    return {
        "status": (
            "sensitivity; recomputed from stored v5 evaluation, labels not reopened"
        ),
        "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "g3_alarm": {
            "published_rule": (
                "alerted if a confirmed point alarm is active at any cycle with true "
                "RUL in [10, 30], whenever it was confirmed; no false-alarm limit"
            ),
            "missed_engines": missed,
            "engines": engines,
            "missed_fraction_wilson_95": wilson_interval(missed, engines),
            "false_events_per_engine": replay["false_events_per_engine"],
            "stricter_readings": strict_alarm(frame, config),
        },
        "file_sensitivity": {
            "method": (
                f"two-stage bootstrap: resample the {len(leave_one_out)} files, then "
                f"engines within each, {config['bootstrap_resamples']} resamples, "
                f"seed {BOOTSTRAP_SEED}; cuts seed {CUT_SEED}"
            ),
            "point": overall,
            "leave_one_file_out": leave_one_out,
            **two_stage_bootstrap(
                errors, config["bootstrap_resamples"], seed=BOOTSTRAP_SEED
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path())
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v5/final"))
    parser.add_argument("--output", type=Path, default=Path("docs/v5_sensitivity.json"))
    args = parser.parse_args()
    report = recompute(args.root, args.run_dir)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    g3 = report["g3_alarm"]
    strict = g3["stricter_readings"]
    print(
        f"G3 missed {g3['missed_engines']}/{g3['engines']}, Wilson 95% "
        f"{g3['missed_fraction_wilson_95']}"
    )
    print(
        f"  alarm must start in window: missed "
        f"{strict['missed_if_alarm_must_start_in_window']}"
    )
    clean_missed = strict["missed_if_false_events_also_count"]
    print(f"  ... and no false event: missed {clean_missed}")
    files = report["file_sensitivity"]
    print("leave one file out:", json.dumps(files["leave_one_file_out"], indent=1))
    print(
        "two-stage 95%:",
        files["ranges_95"],
        "beats baseline in",
        files["resamples_where_model_beats_baseline"],
    )


if __name__ == "__main__":
    main()
