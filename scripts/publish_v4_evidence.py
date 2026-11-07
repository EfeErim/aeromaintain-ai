"""Publish aggregate v4 evidence without redistributing sensor histories or models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aeromaintain.data.pipeline import sha256_file
from aeromaintain.models.v4 import write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(value, prefix=""):
    """Yield every leaf of a nested JSON value with its path."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, f"{prefix}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten(item, f"{prefix}/{index}")
    else:
        yield prefix, value


def compare_reproduction(original, repeat):
    """Compare recorded results from two runs, ignoring only their provenance."""
    keys = ("selection", "chosen", "evaluation")
    first = dict(flatten({key: original[key] for key in keys}))
    second = dict(flatten({key: repeat[key] for key in keys}))
    if first.keys() != second.keys():
        raise ValueError("Reproduction has a different evidence structure")
    floating = [
        path
        for path in first
        if isinstance(first[path], float) or isinstance(second[path], float)
    ]
    return {
        "compared_values": len(first),
        "floating_values": len(floating),
        "max_absolute_difference": max(
            (abs(first[path] - second[path]) for path in floating), default=0.0
        ),
        "non_floating_values_equal": all(
            first[path] == second[path] for path in first if path not in set(floating)
        ),
        "recorded_source_sha256": original["provenance"]["source_sha256"],
        "reproduction_source_sha256": repeat["provenance"]["source_sha256"],
    }


def nasa_concentration(root, dataset):
    """Show how much of the reference NASA mean comes from its worst engines."""
    folder = root / "runs/v4" / dataset
    frame = pd.read_parquet(folder / "oof.parquet")
    plan = np.load(folder / "cuts.npy")
    error = frame["reference"].to_numpy()[plan] - frame["rul_true"].to_numpy()[plan]
    penalty = np.where(error < 0, np.exp(-error / 13) - 1, np.exp(error / 10) - 1)
    per_engine = np.sort(penalty.mean(axis=0))[::-1]
    return {
        "median_per_engine": float(np.median(per_engine)),
        "worst_engine_share": float(per_engine[0] / per_engine.sum()),
        "worst_three_engines_share": float(per_engine[:3].sum() / per_engine.sum()),
    }


def check_claims(evidence):
    """Stop publication if fixed prose below no longer matches the evidence."""
    if evidence["chosen"] != "reference":
        raise ValueError("Results text assumes the capped reference was retained")
    for records in evidence["selection"].values():
        for name, record in records.items():
            if name == "reference":
                continue
            checks = record["gate"]["checks"]
            if checks["critical_misses_acceptable"] or not all(
                passed
                for check, passed in checks.items()
                if check != "critical_misses_acceptable"
            ):
                raise ValueError("Results text assumes only the miss gate failed")
    for result in evidence["evaluation"].values():
        split = result["intervals"]["split_conformal"]["groups"]["overall"]
        mondrian = result["intervals"]["mondrian"]["groups"]["overall"]
        if (
            mondrian["coverage"] <= split["coverage"]
            or mondrian["interval_score"] >= split["interval_score"]
        ):
            raise ValueError("Results text assumes Mondrian was observed to be better")


def publish(root=Path(".")):
    evidence = read(root / "runs/v4/evidence.json")
    check_claims(evidence)
    evidence["reproduction"] = compare_reproduction(
        evidence, read(root / "runs/v4-repro/evidence.json")
    )
    evidence["reference_nasa_concentration"] = {
        dataset: nasa_concentration(root, dataset) for dataset in evidence["selection"]
    }
    concentration = evidence["reference_nasa_concentration"]
    reproduction = evidence["reproduction"]
    if (
        reproduction["max_absolute_difference"] != 0
        or not reproduction["non_floating_values_equal"]
    ):
        raise ValueError("Results text assumes an exact C-MAPSS reproduction")
    evidence["scope"] = (
        "C-MAPSS development-only full-range comparison. Official test labels were "
        "not used in v4. These training engines and v3 hyperparameters were previously "
        "used, so this is not an independent clean C-MAPSS test."
    )
    evidence["cmapss_locks"] = {
        dataset: read(root / f"runs/v4/{dataset}/model_lock.json")
        for dataset in ("FD002", "FD004")
    }
    evidence["independent_simulation"] = {
        name: read(root / f"runs/{run}/evidence.json")
        for name, run in [
            ("DS02", "ncmapss-ds02"),
            ("DS03-DS07", "ncmapss-confirmatory"),
        ]
    }
    evidence["holdout_ledger"] = read(root / "runs/ncmapss_holdout_ledger.json")
    evidence["data_source"] = read(root / "data/raw/ncmapss/source.json")
    evidence["protocol_snapshots"] = {
        path.name: sha256_file(path)
        for path in [
            root / f"docs/{name}.md"
            for name in ("protocol_v4_initial", "protocol_v4_amendment1", "protocol_v4")
        ]
    }
    evidence["current_source_hashes"] = {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in [
            root / name
            for name in (
                "src/aeromaintain/models/v4.py",
                "src/aeromaintain/evaluation/v4.py",
                "src/aeromaintain/data/ncmapss.py",
                "src/aeromaintain/evaluation/ledger.py",
                "configs/protocol_v4.json",
            )
        ]
    }
    write_json(root / "docs/protocol_v4_evidence.json", evidence)
    lines = [
        "# Protocol v4 results",
        "",
        "The capped v3 specification is retained: every uncapped candidate improves "
        "long-RUL error but fails the predeclared critical-miss guardrail. "
        "This is a completed experiment with a negative model-adoption result.",
        "",
        "[Frozen rules and dated amendments](protocol_v4.md) · "
        "[Inspectable evidence, paired intervals, band counts and engine replay "
        "records]"
        "(protocol_v4_evidence.json)",
        "",
        "## C-MAPSS development comparison",
        "",
        evidence["scope"],
        "",
        "The five candidates share engine folds and 200 uniform cuts per engine, "
        "with uncapped truth and no RUL window. Selection has 156 FD002 and 149 "
        "FD004 engines; calibration/evaluation have 52/52 and 50/50 respectively. "
        "NASA is the mean penalty per engine, not the total over engines. "
        "Miss rate means a sampled true-RUL <=30 cycle has prediction >30. "
        "Miss counts below are average engines per draw, not unique engines over "
        "all draws.",
        "",
        "| Dataset | Candidate | RMSE | MAE | NASA/engine | >150 RMSE | >150 bias | "
        "Critical miss rate | Misses/draw |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, records in evidence["selection"].items():
        for name, record in records.items():
            m = record["metrics"]
            lines.append(
                f"| {dataset} | {name} | {m['rmse']:.2f} | {m['mae']:.2f} | "
                f"{m['nasa_per_engine']:.2f} | {m['long_rmse']:.2f} | "
                f"{m['long_bias']:.2f} | {m['critical_miss_rate']:.1%} | "
                f"{m['mean_critical_misses_per_draw']:.2f} |"
            )
    lines += [
        "",
        "Acceptance required improvement in >150 RMSE, no deterioration in "
        "overall RMSE or NASA, and no more than +2 percentage points in critical "
        "miss rate on **both** subsets. All four alternatives fail the last gate "
        "on both subsets. No threshold was relaxed after observing results.",
        "",
        "Read the NASA column as tail sensitivity, not typical engine cost. Its "
        "exponential penalty on uncapped truth is dominated by a few long-lived "
        "engines the capped reference underpredicts by hundreds of cycles. One engine "
        f"contributes {concentration['FD002']['worst_engine_share']:.0%} of the "
        f"FD002 reference mean and {concentration['FD004']['worst_engine_share']:.0%} "
        "of FD004; the reference's median engine penalties are "
        f"{concentration['FD002']['median_per_engine']:.0f} and "
        f"{concentration['FD004']['median_per_engine']:.0f}. The NASA gate therefore "
        "says little about most engines, though it did not decide this outcome.",
        "",
        "Paired bootstrap resamples whole engine columns (2000 resamples), "
        "retaining all cuts of each engine. Candidate-minus-reference 95% "
        "intervals for normalized uncapped XGBoost:",
        "",
        "| Dataset | RMSE difference | >150 RMSE difference | "
        "Critical miss-rate difference |",
        "|---|---:|---:|---:|",
    ]
    for dataset, records in evidence["selection"].items():
        ci = records["xgboost_normalized"]["paired_comparison"]["intervals_95"]
        values = [f"{ci[k][0]:.2f} to {ci[k][1]:.2f}" for k in ("rmse", "long_rmse")]
        values.append(
            f"{ci['critical_miss_rate'][0]:+.1%} to {ci['critical_miss_rate'][1]:+.1%}"
        )
        lines.append(f"| {dataset} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Intervals and persistent alarms",
        "",
        "Refitted reference point models use only selection engines. Calibration "
        "uses separate engines and all RULs. Evaluation engines are excluded from "
        "v4 selection and calibration. Intervals have nominal 90% coverage. "
        "The split-conformal rule was preferred in advance for engine-level "
        "calibration sampling; that preference does not imply better observed "
        "coverage. Mondrian has better observed coverage and interval score here.",
        "",
        "Alarms confirm after three consecutive cycles at or below 30. A false "
        "event confirms with true RUL >30; uninterrupted runs merge into one "
        "event. Lead time is measured at confirmation. The reported median "
        "excludes engines with no alarm, whose count is retained in the JSON. "
        "A point alarm carries the point prediction, while lower-bound alarms "
        "trade earlier warning for more false events. Missed engines here is a "
        "replay measure: no confirmed alert at any cycle with true RUL <=30. It is "
        "more lenient than the sampled-cut critical miss rate above, so zero missed "
        "engines does not mean every critical cycle was flagged.",
        "",
        "| Dataset | Interval | Alarm | Coverage | Width | Interval score | "
        "Median lead | Missed engines | False events/engine |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, result in evidence["evaluation"].items():
        for method, item in result["intervals"].items():
            m = item["groups"]["overall"]
            for label, field in [
                ("point", "point_alarm"),
                ("lower bound", "lower_bound_alarm"),
            ]:
                alarm = item[field]
                lines.append(
                    f"| {dataset} | {method} | {label} | {m['coverage']:.1%} | "
                    f"{m['mean_width']:.1f} | {m['interval_score']:.1f} | "
                    f"{alarm['median_first_confirmed_lead']:.1f} | "
                    f"{alarm['missed_engines']}/{alarm['engines']} | "
                    f"{alarm['false_events_per_engine']:.2f} |"
                )
    lines += [
        "",
        "Overall coverage hides how each method distributes its misses. Split "
        "conformal uses one width for every RUL, so it over-covers short RUL and "
        "under-covers long RUL. Mondrian stays near nominal in each band, but pays "
        "with very wide long-RUL intervals because the capped point model still "
        "underpredicts there.",
        "",
        "| Dataset | Interval | True RUL | Engines | Coverage | Width |",
        "|---|---|---|---:|---:|---:|",
    ]
    for dataset, result in evidence["evaluation"].items():
        for method, item in result["intervals"].items():
            for band, label in [
                ("rul_<=30", "<=30"),
                ("rul_31-150", "31-150"),
                ("rul_>150", ">150"),
            ]:
                group = item["groups"][band]
                lines.append(
                    f"| {dataset} | {method} | {label} | {group['engines']} | "
                    f"{group['coverage']:.1%} | {group['mean_width']:.1f} |"
                )
    lines += [
        "",
        "The JSON reports lower/upper misses, coverage, width and interval score "
        "for every true-RUL band and operating condition. It flags fewer than 20 "
        "distinct engines; 200 cuts from an engine are not 200 independent "
        "observations. Neither interval supports a safety guarantee. "
        "The public demo remains on its labeled v3 bundle.",
        "",
        "## Independent N-CMAPSS simulation evidence",
        "",
        "Measured sensors and operating settings only: 72 cycle-level mean/std/min/max "
        "features, Ridge alpha=1, engine-equal weights and fit-only preprocessing. "
        "No virtual sensors, simulator health variables, labels, unit IDs or cycle "
        "age enter the model. DS02 is a six-fit/three-test-engine descriptive pilot. "
        "The confirmatory mixture uses DS03-DS07, with four test and three calibration "
        "engines in each subset; remaining engines fit. Source file namespaces "
        "prevent unit-ID collisions. This is not cross-regime or real-fleet transfer.",
        "",
        "| Experiment | Test engines | Ridge RMSE | Mean baseline RMSE | "
        "Critical miss rate | Coverage | Width |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in evidence["independent_simulation"].items():
        interval = item.get("interval")
        coverage = f"{interval['coverage']:.1%}" if interval else "Not calibrated"
        width = f"{interval['mean_width']:.1f}" if interval else "—"
        lines.append(
            f"| {name} | {item['point']['engines']} | {item['point']['rmse']:.2f} | "
            f"{item['baseline']['rmse']:.2f} | "
            f"{item['point']['critical_miss_rate']:.1%} | {coverage} | {width} |"
        )
    independent = evidence["independent_simulation"]["DS03-DS07"]
    ci = independent["intervals_95"]
    lines += [
        "",
        f"Confirmatory 95% engine-bootstrap intervals: RMSE {ci['rmse'][0]:.2f} to "
        f"{ci['rmse'][1]:.2f}; coverage {ci['coverage'][0]:.1%} to "
        f"{ci['coverage'][1]:.1%}. "
        "Twenty evaluation engines, including only four per subset, still give limited "
        "precision. The maximum of 15 calibration residuals defines the 90% interval.",
        "",
        "Predeclared gates: "
        + "; ".join(
            f"`{name}`: **{'PASS' if passed else 'FAIL'}**"
            for name, passed in independent["gates"].items()
        )
        + ". These gates did not include critical misses: Ridge still predicts above "
        "30 on about half of sampled true-RUL <=30 cycles, so passing them shows "
        "better-than-baseline error and empirical coverage, not useful critical-engine "
        "detection.",
        "",
        "Before evaluation labels were read, protocol/model/data hashes and engine "
        "partitions were published locally in [DS02 lock](ncmapss_locks/ds02.json) "
        "and [confirmatory lock](ncmapss_locks/confirmatory.json), then claimed in "
        "the atomic interprocess ledger. This is a local preregistration record, "
        "not an independently timestamped public preregistration.",
        "",
        "The DS08 plan was superseded before fitting or reading N-CMAPSS labels: "
        "the downloaded archive did not match the 54-unit assumption and DS08d "
        "was truncated according to its HDF5 header. It was not repaired or used. "
        "[NASA source and download](https://www.nasa.gov/intelligent-systems-division/"
        "discovery-and-systems-health/pcoe/pcoe-data-set-repository/) · "
        "[Data paper](https://doi.org/10.3390/data6010005).",
        "",
        "## Reproduction and provenance",
        "",
        "Use Python 3.11 and the repository environment. Raw data, models and "
        "per-cycle predictions remain Git-ignored; aggregate evidence and locks "
        "are published in docs. Hashes identify every source HDF5 file, model, "
        "configuration and protocol snapshot.",
        "",
        "```powershell",
        'python -m pip install -e ".[dev,ncmapss]"',
        "python -m aeromaintain.models.v4 --project-root . --output runs/v4",
        "python -m aeromaintain.models.v4 --project-root . --output runs/v4-repro",
        "python scripts/fetch_ncmapss.py --subsets DS02 DS03 DS04 DS05 DS06 DS07",
        "$raw = 'data/raw/ncmapss/N-CMAPSS'",
        "python -m aeromaintain.data.ncmapss ${raw}_DS02-006.h5 "
        "--output runs/ncmapss-ds02",
        "python -m aeromaintain.data.ncmapss ${raw}_DS03-012.h5 ${raw}_DS04.h5 "
        "${raw}_DS05.h5 ${raw}_DS06.h5 ${raw}_DS07.h5 --confirmatory "
        "--output runs/ncmapss-confirmatory",
        "python scripts/publish_v4_evidence.py",
        "ruff check .",
        "ruff format --check .",
        "pytest --cov=src/aeromaintain --cov-report=term-missing --cov-fail-under=80",
        "```",
        "",
        "**Source changes during the runs.** `models/v4.py` was edited while the "
        "recorded C-MAPSS run was in progress, so that run's source hash "
        f"(`{reproduction['recorded_source_sha256'][:12]}`) differs from the published "
        f"file (`{reproduction['reproduction_source_sha256'][:12]}`), and the earlier "
        "revision was not kept. A full rerun of the published code in a new directory "
        f"matched all {reproduction['compared_values']} recorded selection and "
        "evaluation values exactly, including tree counts and interval margins. Only "
        "`model.joblib` bytes and the wording of a reason string differ, so the model "
        "hashes in the two runs are not interchangeable. "
        "`data/ncmapss.py` was also edited 12 seconds after the DS02 pilot was "
        "evaluated and before the confirmatory lock; `evaluation/ledger.py` also "
        "changed between those runs. DS02 cannot be rerun without "
        "reopening a claimed holdout, so its pilot numbers have no equivalent check; "
        "it is descriptive only. The confirmatory run used the published "
        "`data/ncmapss.py` by file timestamp, which is not a cryptographic record.",
        "",
        "Run directories are exclusive; a second model cannot reclaim an already "
        "opened holdout. Reproduction on a new workspace is a previously-seen "
        "benchmark, not another clean test. Inspect existing stored predictions "
        "and evidence without reopening labels. C-MAPSS began under "
        "[the initial protocol](protocol_v4_initial.md); interval locks reference "
        "[amendment 1](protocol_v4_amendment1.md). P0/P1 rules did not change. "
        "The historical amendment-1 assertion about DS08 was corrected by amendment 2.",
        "",
    ]
    (root / "docs/results_v4.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    publish()
