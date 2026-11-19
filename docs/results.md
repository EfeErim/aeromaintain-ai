# C-MAPSS results (protocols v1–v3)

Run files are Git-ignored; aggregate evidence is in
[`reference_evidence.json`](reference_evidence.json). Protocol v4 changed no model
([results](results_v4.md)), and the capped rescoring made after the labels opened is
[the last section](#retrospective-comparison-on-the-literature-scale).

## Protocol v3 runs

Produced on 2026-11-05 under the frozen [protocol v3](protocol_v3.md). The locks and
development evidence were published in commit `56bca9b` before any FD002 or FD004 label
opened. That commit was `94fd0ec` before the history rewrite noted in
[results v5](results_v5.md); the two trees are identical, and
`reference_evidence.json` and `protocol_v5.md` still name the old hashes (`94fd0ec`,
and `6dcbec0` for what is now `8c32449`). Development evidence is in
[`protocol_v3_development.json`](protocol_v3_development.json).

| Run | Dataset | Label | What it can support |
|---|---|---|---|
| `fd002-v3-clean` | FD002 | clean single-open test | Independent check under six conditions, one fault mode |
| `fd004-v3-clean` | FD004 | clean single-open test | Independent check under six conditions, two fault modes |
| `fd001-v3` | FD001 | test previously seen | Secondary result only |
| `fd003-v3` | FD003 | test previously seen | Secondary result only |

The ledger records one opening each, by model locks
`fa50040abf9731cf3fd4a3b47aeb82222792dfc7304dc137b10166615ba4cb37` (FD002) and
`f296bab85503ef145919402059fc63fc735764f2019b9e24893396bcabca41f7` (FD004).
`fd003-v2-clean` keeps its FD003 ledger entry.

### Official test

Ranges are 95% engine-level bootstrap intervals over 2,000 resamples. Clean tests get
bootstrap intervals only, because a seed sweep would score their labels again.

| Metric | FD002 clean | FD004 clean | FD001 seen | FD003 seen |
|---|---:|---:|---:|---:|
| Engines | 259 | 248 | 100 | 100 |
| RMSE | 27.362 (23.767–30.817) | 27.395 (24.532–30.228) | 15.796 (12.867–18.781) | 15.453 (12.914–17.949) |
| MAE | 18.459 (15.962–21.006) | 19.650 (17.299–21.986) | 11.154 (9.100–13.431) | 11.439 (9.432–13.524) |
| Motor-normalized NASA score | 10,387.9 (5,192.6–16,823.1) | 5,125.7 (3,627.0–6,816.5) | 398.31 (232.98–624.67) | 400.71 (251.39–584.38) |
| NASA score per engine | 40.11 | 20.67 | 3.98 | 4.01 |
| Signed bias | -8.981 (-12.122 to -5.711) | -9.337 (-12.625 to -6.309) | -1.593 (-4.608–1.280) | 0.099 (-2.865–2.965) |
| Interval coverage, nominal 90% | 0.826 (0.780–0.873) | 0.827 (0.778–0.871) | 0.94 (0.89–0.98) | 0.94 (0.89–0.98) |
| Mean interval width | 55.772 (52.906–58.657) | 58.526 (55.630–61.343) | 55.422 (51.190–59.443) | 54.300 (49.991–58.538) |

`fd001-v3` keeps v2's FD001 specification, since no v3 candidate cleared its rule, so
its results match `fd001-v2`.

### Critical-engine alert rules

A truly critical engine has `rul_true <= 30`: 61 on FD002, 53 on FD004, 25 on FD001, and
20 on FD003. Ranges are two-sided Wilson 95% intervals.

| Dataset | Rule | Recall | Precision | False-alarm rate | Missed | False alarms |
|---|---|---:|---:|---:|---:|---:|
| FD002 | `prediction <= 30` | 93.4% (84.3–97.4%) | 93.4% (84.3–97.4%) | 2.0% (0.8–5.1%) | 4 | 4 |
| FD002 | `interval_low <= 25` | 98.4% (91.3–99.7%) | 73.2% (62.7–81.6%) | 11.1% (7.5–16.2%) | 1 | 22 |
| FD004 | `prediction <= 30` | 84.9% (72.9–92.1%) | 90.0% (78.6–95.7%) | 2.6% (1.1–5.9%) | 8 | 5 |
| FD004 | `interval_low <= 20` | 96.2% (87.2–99.0%) | 76.1% (64.7–84.7%) | 8.2% (5.1–12.9%) | 2 | 16 |
| FD001 seen | `prediction <= 30` | 80.0% (60.9–91.1%) | 100% (83.9–100%) | 0.0% (0.0–4.9%) | 5 | 0 |
| FD001 seen | `interval_low <= 20` | 96.0% (80.5–99.3%) | 85.7% (68.5–94.3%) | 5.3% (2.1–12.9%) | 1 | 4 |
| FD003 seen | `prediction <= 30` | 85.0% (64.0–94.8%) | 94.4% (74.2–99.0%) | 1.25% (0.2–6.7%) | 3 | 1 |
| FD003 seen | `interval_low <= 25` | 100% (83.9–100%) | 74.1% (55.3–86.8%) | 8.75% (4.3–17.0%) | 0 | 7 |

### Interval by RUL band

Each cell gives test coverage and mean width. The predicted `>100` and true `>125` bands
are where the six-condition tests fall short.

| Band | FD002 | FD004 | FD001 seen | FD003 seen |
|---|---:|---:|---:|---:|
| Predicted `<=30` | 0.92, 16.9 | 0.96, 19.1 | 1.00, 19.5 | 1.00, 15.8 |
| Predicted `31-60` | 0.90, 43.0 | 0.83, 41.7 | 0.76, 40.1 | 0.94, 35.5 |
| Predicted `61-100` | 0.90, 81.3 | 0.89, 79.9 | 0.97, 74.8 | 0.93, 73.5 |
| Predicted `>100` | 0.67, 69.1 | 0.71, 70.7 | 0.97, 67.5 | 0.92, 68.1 |
| True `0-30` | 0.95, 18.5 | 0.96, 22.4 | 0.96, 23.6 | 1.00, 18.7 |
| True `31-60` | 0.95, 47.0 | 0.89, 47.7 | 1.00, 52.5 | 0.95, 45.4 |
| True `61-125` | 0.89, 73.6 | 0.92, 74.4 | 0.92, 68.9 | 0.91, 69.8 |
| True `>125` | 0.49, 70.6 | 0.55, 71.8 | 0.91, 70.1 | 0.93, 68.8 |

### Diagnosis made after opening

Nothing in this section was computed before the labels opened, and no result was
changed because of it. It explains the gap between development and test, and does not
replace the figures above.

| Engines by true RUL | FD002 engines | RMSE | Bias | Coverage | Share of squared error | FD004 engines | RMSE | Bias | Coverage | Share of squared error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| At most 125 | 202 | 14.69 | 1.36 | 0.92 | 22% | 181 | 15.41 | 2.69 | 0.93 | 23% |
| 126–150 | 21 | 27.90 | -24.95 | 0.90 | 8% | 25 | 27.75 | -24.42 | 0.96 | 10% |
| Above 150 | 36 | 61.00 | -57.71 | 0.25 | 69% | 42 | 54.31 | -52.19 | 0.31 | 67% |

- **The cap sets the error.** Training targets stop at 125, and the largest test
  prediction is 126.8 on FD002 and 127.0 on FD004, while true RUL reaches 194 and 195.
  The largest upper interval bound is about 167, so an engine beyond that cannot be
  covered.
- **The window hid it.** Selection, the interval check, and the risk threshold were
  judged on rows with true RUL of at most 150, so development never scored such engines.
  Within that range, test RMSE is 16.39 and 17.38 and coverage 0.92 and 0.93, close to
  the development figures.
- **FD001 and FD003 could not show it.** Their true test RUL stops at 145.

Any fix would change a clean test's result, so under rule 5 it needs a new protocol and
could not call these tests clean again. No clean C-MAPSS subset remains after this.

### Development selection

Metrics are medians over 200 seeded draws of one row per engine, on development engines
with `rul_true <= 150`. XGBoost lowered RMSE against Ridge by 11.6% on FD001, 11.6% on
FD002, 17.2% on FD003, and 13.3% on FD004, and won at least 99% of paired draws.

| Model | Dataset | RMSE | MAE | NASA | Point-rule recall |
|---|---|---:|---:|---:|---:|
| Ridge | FD002 | 21.41 | 16.72 | 1,924 | 0.73 |
| **XGBoost, compact, normalized, cap 125** | FD002 | **18.93** | 14.28 | 1,470 | 0.84 |
| Ridge | FD004 | 22.62 | 17.93 | 2,386 | 0.68 |
| **XGBoost, compact, normalized, cap 125** | FD004 | **19.61** | 14.60 | 1,764 | 0.84 |
| Ridge | FD001 | 21.01 | 16.75 | 626 | 0.70 |
| **XGBoost, all features, cap 125** | FD001 | **18.57** | 14.40 | 501 | 0.85 |
| Ridge | FD003 | 21.58 | 17.11 | 702 | 0.70 |
| **XGBoost, all features, cap 125, five seeds** | FD003 | **17.86** | 13.27 | 473 | 0.89 |

Each later stage had to lower median RMSE at least 1% without a worse median NASA score.

| Stage | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Starting point | 18.57, 501 | 21.19, 2,110 | 18.05, 493 | 21.82, 2,345 |
| Condition handling | not applicable | **18.93, 1,470, passed** | not applicable | **19.61, 1,764, passed** |
| `window_focus` emphasis | 18.92, 499 | 19.05, 1,415 | 18.15, 467 | 19.48, 1,569 |
| Five-seed ensemble | 18.80, 503 | 18.85, 1,430 | **17.86, 473, passed** | 19.43, 1,616 |

Each cell gives median RMSE, then median NASA score. Unmarked candidates failed the rule.
Targets are in the [protocol history](protocol_history.md#targets).

### Window sensitivity

Each cell gives median RMSE for Ridge and XGBoost, then the rule's champion.

| Window | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| 130 | 18.35 / 16.35, XGBoost | 18.59 / 16.52, XGBoost | 18.33 / 15.66, XGBoost | 20.11 / 17.45, XGBoost |
| **150 (used)** | 21.01 / 18.57, XGBoost | 21.41 / 18.93, XGBoost | 21.58 / 17.86, XGBoost | 22.62 / 19.61, XGBoost |
| 175 | 26.32 / 24.15, XGBoost | 26.66 / 23.95, XGBoost | 27.36 / 23.54, XGBoost | 28.61 / 25.37, XGBoost |
| None | 44.86 / 43.78, **Ridge** | 44.82 / 43.25, **Ridge** | 72.34 / 71.43, **Ridge** | 67.19 / 65.38, **Ridge** |

### Interval and risk threshold

Mondrian stayed on every subset. CQR, with tree counts tuned on pinball loss, ran on
FD001 and FD003 only and failed the rule: its width was 54–69% larger on FD001 and 69–80%
larger on FD003. [Amendment 1](protocol_v3.md#amendment-1) removed it from FD002 and
FD004 before any model on those subsets was fitted.

| Dataset | Leave-engines-out coverage | Mean width | `interval_low <= T` | Median recall | Median missed / false alarms |
|---|---:|---:|---:|---:|---:|
| FD001 | 0.91 | 55.7 | 20 | 0.96 | 1 / 5 |
| FD002 | 0.90 | 58.8 | 25 | 0.98 | 1 / 23 |
| FD003 | 0.91 | 53.2 | 25 | 1.00 | 0 / 8 |
| FD004 | 0.91 | 58.6 | 20 | 0.97 | 1.5 / 15 |

### Seed sensitivity

Ten seeds (42–51) refit the locked settings on FD001 and FD003 only, because a sweep
scores test labels. Selection is not rerun, so this spread excludes selection variance.
FD002 and FD004 get no sweep, so their point values have no seed spread beside them.

| Metric | FD001 mean, SD, range | FD003 mean, SD, range |
|---|---:|---:|
| RMSE | 16.034, 0.135, 15.796–16.205 | 15.412, 0.081, 15.227–15.535 |
| MAE | 11.623, 0.205, 11.154–11.902 | 11.410, 0.095, 11.271–11.601 |
| NASA score | 419.4, 13.3, 398.3–442.4 | 398.5, 7.1, 382.4–407.2 |
| Interval coverage | 0.942, 0.009, 0.93–0.95 | 0.945, 0.007, 0.93–0.95 |
| Mean interval width | 56.31, 0.65, 55.42–57.43 | 54.70, 0.40, 54.16–55.19 |
| Risk-rule recall | 0.960, 0.027, 0.92–1.00 | 1.000, 0.000, 1.00–1.00 |
| Risk-rule false alarms | 3.2, 0.8, 2–4 | 6.8, 0.4, 6–7 |

Produced by `aeromaintain robustness --base-run-id fd001-v3` and
`aeromaintain robustness --base-run-id fd003-v3` (default: ten seeds from 42), which
write `runs/<run>/robustness/seed_sweep.json`. The command refuses a run whose test is
still clean.

The risk threshold stayed 20 on FD001 and 25 on FD003 for every seed. Seed 42 has the lowest
RMSE of ten on FD001 and the second highest on FD003, at 15.453. The
five-seed FD003 ensemble spreads less than FD001's single model.

## Protocol v2 runs

Produced on 2026-11-04 under the frozen [protocol v2](protocol_v2.md). `fd003-v2-clean`
is the clean single-open test; its ledger records one opening, by model lock
`bcd2acda16f5a60497725543879d033eca0d83932a7f62525581e44172bfba33`. `fd001-v2` (test
previously seen) is identical to the `fd001-v3` column above.

### Official test

Ranges are 95% engine-level bootstrap intervals over 2,000 resamples (Wilson for alert
rates). FD003 has 20 truly critical engines.

| Metric | FD003 clean |
|---|---:|
| Engines | 100 |
| RMSE | 15.066 (12.607–17.503) |
| MAE | 11.227 (9.339–13.276) |
| Motor-normalized NASA score | 398.56 (239.62–619.97) |
| Signed bias | 0.384 (-2.547–3.284) |
| Interval coverage, nominal 90% | 0.92 (0.86–0.97) |
| Mean interval width | 53.285 (49.279–57.278) |
| `prediction <= 30`: recall, missed, false alarms | 80.0% (58.4–91.9%), 4, 1 |
| `interval_low <= 25`: recall, missed, false alarms | 100% (83.9–100%), 0, 8 |

### Development

Medians over 200 seeded draws on 80 development engines with `rul_true <= 150`. On
FD003, Ridge (alpha 100, no age) reached RMSE 21.58 and NASA 702; XGBoost (compact
features, cap 130) reached **17.62** and 515, 18.4% lower RMSE, winning 99% of paired
draws. The FD001 rows match v3. Neither reached the RMSE target of about 16.6. With
window 130, 175 or none, FD003 RMSE for Ridge / XGBoost was 18.33 / 15.59, 27.36 / 22.25
and 72.34 / 69.27; with no window Ridge wins on NASA score, and by rule that
disagreement is published without changing the decision.

The risk threshold is the smallest `interval_low <= T` with at least 95%
leave-engines-out recall: 20 on FD001 (1 missed, 5 false alarms) and 25 on FD003 (0
missed, 7 false alarms). Leave-engines-out interval coverage is 0.91 on both, with mean
width 55.7 and 52.0, against a target of 45.

### Challengers on FD001

[Amendment 1](protocol_v2.md#amendment-1) wrote these rules after the first FD001
result and before any challenger result. Neither earned a place.

| Challenger | Result | Rule outcome |
|---|---|---|
| LSTM, 64 units, one layer | Median RMSE 18.39 (0.95% lower), NASA 546 against 501 | XGBoost stays |
| CQR | Coverage 0.91, mean width 98.6, 77% wider than Mondrian | Mondrian stays |

The CQR trial reused a tree count tuned for squared error, so it shows only that this
bounded trial lost.

Both came from `aeromaintain escalation-study --base-run-id fd001-v2-selection --method
lstm` (and `--method cqr`), which score on training engines only and write
`runs/fd001-v2-selection/robustness/{lstm,cqr}_study.json`. Byte-identical copies are
[`fd001_lstm_study.json`](fd001_lstm_study.json) (`rmse_improvement_fraction`,
`nasa_not_worse`) and [`fd001_cqr_study.json`](fd001_cqr_study.json)
(`width_reduction_fraction` -0.771). The LSTM study needs
the `lstm` extra (`pip install -e ".[lstm]"`). `torch` is not in the pinned
constraints and the study file does not record its version, so the LSTM figures are
not pinned to an environment; CI skips the LSTM test.

### Reproducibility

`fd001-v2` and `fd001-v2-repeat` ran with identical inputs on one machine. Models,
predictions, metrics and analyses were byte-identical; only files that carry the run ID
or a hash of such a file differed
(`aeromaintain compare-runs --first fd001-v2 --second fd001-v2-repeat`, written to
`runs/fd001-v2/robustness/compare_fd001-v2-repeat.json`). This does not establish identical output across
machines or library versions.

## Protocol v1 reference

The run `m4-fd001-seed42-20260729` remains published as v1, reproducible from commit
`1b14801`; Ridge was its champion under all-rows selection. On FD001 it reached RMSE
19.622062, MAE 15.369728 and coverage 0.89 at mean width 60.623140. The interval rule
`interval_low <= 30` caught all 25 critical engines with 9 false alarms; the point rule
missed 13.

## Retrospective comparison on the literature scale

**Status: retrospective. Every C-MAPSS test label was opened before this analysis.**
Nothing here is a new clean result, and nothing here changes a model, a threshold or a
headline figure.

Produced on 2026-11-07 by
[`scripts/retrospective_capped_metrics.py`](../scripts/retrospective_capped_metrics.py).
Evidence, with prediction file hashes, is in
[`retrospective_capped_evidence.json`](retrospective_capped_evidence.json).

The clean v3 tests report RMSE 27.4 on FD002 and FD004 against NASA's uncapped truth.
Most papers score against truth capped at 125 or 130, so the stored v3 test predictions
(one per engine, unchanged) are rescored here against `min(truth, cap)`. Ranges are 95%
engine-level bootstrap intervals (2,000 resamples, seed 42); the script first checks that
the uncapped RMSE matches each run's published `metrics.json`.

### Rescored test results

| Dataset | Test status | Truth | RMSE (95% CI) | NASA score (95% CI) | Engines above cap |
|---|---|---|---:|---:|---:|
| FD002 | clean under v3 | uncapped | 27.36 (23.77–30.82) | 10,388 (5,193–16,823) | – |
| FD002 | clean under v3 | capped at 125 | **14.74** (13.28–16.15) | 893 (673–1,146) | 57 of 259 |
| FD002 | clean under v3 | capped at 130 | 15.73 (14.28–17.14) | 978 (755–1,230) | 55 of 259 |
| FD004 | clean under v3 | uncapped | 27.40 (24.53–30.23) | 5,126 (3,627–6,816) | – |
| FD004 | clean under v3 | capped at 125 | **15.33** (13.69–17.05) | 1,011 (702–1,391) | 67 of 248 |
| FD004 | clean under v3 | capped at 130 | 16.27 (14.58–18.00) | 1,107 (793–1,485) | 60 of 248 |
| FD001 | previously seen | capped at 125 | 14.39 (11.45–17.39) | 346 (191–568) | 11 of 100 |
| FD003 | previously seen | capped at 125 | 13.69 (11.32–15.91) | 332 (204–490) | 15 of 100 |

"Engines above cap" counts engines whose official truth exceeds the cap. In the capped
rows their error is measured against the cap, not against their real remaining life.

### Published results under the same last-cycle protocol

RMSE on the official test set, one prediction per engine, as reported by the authors.

| Method | Year | Cap | FD002 | FD004 | Source of the figures |
|---|---:|---:|---:|---:|---|
| CNN (Babu et al.) | 2016 | 130 | 30.29 | 29.16 | [Survey, Table 7](https://arxiv.org/abs/2010.03207) and [Bi-cLSTM, Table V](https://arxiv.org/abs/2603.00745) |
| LSTM (Zheng et al.) | 2017 | 130 | 24.49 | 28.17 | Same two sources |
| Deep CNN (Li, Ding and Sun) | 2018 | 125 | 22.36 | 23.31 | [Reliability Engineering & System Safety 172](https://econpapers.repec.org/RePEc:eee:reensy:v:172:y:2018:i:c:p:1-11), via the survey |
| TATFA-Transformer (Zhang et al.) | 2024 | not stated in the citing table | 15.07 | 18.81 | Reliability Engineering & System Safety 241, via Bi-cLSTM, Table V |
| Bi-cLSTM (Chowdhury et al.) | 2026 | 125 | 13.96 | 14.25 | [arXiv 2603.00745](https://arxiv.org/abs/2603.00745), Table V |
| **This project, v3 XGBoost** | 2026 | 125 | **14.74** | **15.33** | This section, retrospective |

Sources disagree on the last digit for some older rows: Babu FD004 appears as both 29.15
and 29.16. A 2024 DFormer row was left out, because the citation attached to it in the
citing table does not match its name.

### How to read the comparison

On the literature's own scale the v3 model is competitive, but the comparison still
favours the papers: our models were locked before their labels opened, while most papers
tune against the same test set; preprocessing differs, and per-condition normalization
alone moves FD002/FD004 RMSE by several points
([Gupta, engrXiv](https://engrxiv.org/preprint/view/8145)); and gaps such as 14.74
against 13.96 fall inside our bootstrap range.

Capping the truth also hides the main failure: the 57 FD002 and 67 FD004 engines above
125 cycles are the ones the model underpredicts most. The uncapped 27.4 stays the
headline; the capped figure is only for comparing with papers that cap their truth.
