# Retrospective comparison on the literature scale

**Status: retrospective. Every C-MAPSS test label was opened before this analysis.**
Nothing here is a new clean result, and nothing here changes a model, a threshold or a
headline figure.

Produced on 2026-11-07 by
[`scripts/retrospective_capped_metrics.py`](../scripts/retrospective_capped_metrics.py).
Evidence, with prediction file hashes, is in
[`retrospective_capped_evidence.json`](retrospective_capped_evidence.json).

## Why this exists

The clean v3 tests report RMSE 27.4 on FD002 and FD004 against the official, uncapped
truth. Most C-MAPSS papers score against truth capped at 125 or 130 cycles, the same
cap they train with. The two numbers are on different scales, so comparing them
directly says nothing about the model.

## Method

- **Predictions:** the stored v3 official-test predictions (one per engine, last
  cycle), unchanged. The v3 model was trained with a cap of 125.
- **Truth:** the official `RUL_FD00x` values, either as published or as
  `min(truth, cap)` for caps of 125 and 130.
- **NASA score:** the sum over engines of the asymmetric penalty (13 for early
  predictions, 10 for late ones).
- **Ranges:** 95% engine-level percentile bootstrap, 2,000 resamples, seed 42.

As a check, the uncapped RMSE has to match each run's published `metrics.json`
before anything is written.

## Results

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

## Published results under the same last-cycle protocol

RMSE on the official test set, one prediction per engine, as reported by the authors.

| Method | Year | Cap | FD002 | FD004 | Source of the figures |
|---|---:|---:|---:|---:|---|
| CNN (Babu et al.) | 2016 | 130 | 30.29 | 29.16 | [Survey, Table 7](https://arxiv.org/abs/2010.03207) and [Bi-cLSTM, Table V](https://arxiv.org/abs/2603.00745) |
| LSTM (Zheng et al.) | 2017 | 130 | 24.49 | 28.17 | Same two sources |
| Deep CNN (Li, Ding and Sun) | 2018 | 125 | 22.36 | 23.31 | [Reliability Engineering & System Safety 172](https://econpapers.repec.org/RePEc:eee:reensy:v:172:y:2018:i:c:p:1-11), via the survey |
| TATFA-Transformer (Zhang et al.) | 2024 | not stated in the citing table | 15.07 | 18.81 | Reliability Engineering & System Safety 241, via Bi-cLSTM, Table V |
| Bi-cLSTM (Chowdhury et al.) | 2026 | 125 | 13.96 | 14.25 | [arXiv 2603.00745](https://arxiv.org/abs/2603.00745), Table V |
| **This project, v3 XGBoost** | 2026 | 125 | **14.74** | **15.33** | This page, retrospective |

Sources disagree on the last digit for some older rows: Babu FD004 appears as both 29.15
and 29.16. A 2024 DFormer row was left out, because the citation attached to it in the
citing table does not match its name.

## How to read this

**On the literature's own scale, the v3 model is competitive.** With truth capped at
125, its FD002 and FD004 RMSE sits well below the 2016–2018 deep models and near recent
transformer and recurrent results. The 27.4 headline was never evidence of a weak model
on the usual scale.

**The comparison still favours the papers.** Three things cut the other way:

1. Our FD002 and FD004 predictions come from models locked before their labels opened.
   Most papers tune architectures and epochs while repeatedly scoring the same official
   test set.
2. Preprocessing differs. Per-condition normalization alone moves FD002/FD004 RMSE by
   several points ([Gupta, engrXiv](https://engrxiv.org/preprint/view/8145)).
3. Small differences, such as 14.74 against 13.96, fall inside our bootstrap range. The
   same audit finds that pipeline choices such as which data the final model is refit
   on can move RMSE by about one cycle. Differences that small do not rank methods.

**Capping the truth hides the main failure.** The 57 FD002 and 67 FD004 engines above
125 cycles are the engines the model underpredicts most. Capped scoring treats any
prediction near 125 as nearly correct for them. The uncapped figure (27.4) remains this
project's headline, and the capped figure is the one to use only when comparing with
papers that cap their truth.
