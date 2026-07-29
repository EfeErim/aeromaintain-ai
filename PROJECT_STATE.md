# AeroMaintain AI Project State

Last updated: 2026-07-29

## Current position

- Active phase: Phase 2
- Active milestone: M2.1 — Causal feature generation
- Overall status: Phases 0 and 1 complete; Phase 2 ready to start
- Current blocker: None

## Milestone register

| ID | Milestone | Status | Evidence |
|---|---|---|---|
| M0.1 | Project skeleton | complete | Python 3.11 editable install; package import; doctor 5/5 |
| M0.2 | Research and data governance | complete | `docs/research.md`; `docs/data_card.md`; NASA source and redistribution boundary |
| M0.3 | Quality infrastructure | complete | Ruff, pytest, coverage and Python 3.11 CI pass |
| M1.1 | Safe data acquisition | complete | NASA ZIP size/hash verification; FD001-only safe extraction |
| M1.2 | Parser and data contract | complete | 26-column schema; expected train/test row and engine counts |
| M1.3 | RUL target and engine split | complete | Capped/uncapped RUL; deterministic 80/20 engine split; isolated labels |
| M1.4 | EDA and quality report | complete | Development-only EDA; stable processed hashes across repeated prepare |
| M2.1 | Causal feature generation | pending | — |
| M2.2 | Baselines and grouped CV | pending | — |
| M2.3 | XGBoost and champion selection | pending | — |
| M2.4 | Uncertainty, explanation and model lock | pending | — |
| M2.5 | Locked official test evaluation | pending | — |
| M3.1 | Synthetic maintenance scenario | pending | — |
| M3.2 | Baseline maintenance policies | pending | — |
| M3.3 | CP-SAT schedule | pending | — |
| M3.4 | Policy and capacity comparison | pending | — |
| M4.1 | Streamlit decision application | pending | — |
| M4.2 | End-to-end pipeline | pending | — |
| M4.3 | Documentation and release preparation | pending | — |
| M4.4 | Final release candidate | pending | — |

## Phase 0 closure evidence

- Clean Python 3.11 editable installation and package import passed.
- `aeromaintain doctor`, Ruff, format, pytest, coverage, and Python 3.11 CI
  passed.
- Research and data-card documents record simulation and redistribution limits.
- Generated data and run paths are Git-ignored.

## Phase 1 closure evidence

- `aeromaintain prepare` verified the NASA archive:
  `Bytes=12425978`,
  `SHA256=74BEF434A34DB25C7BF72E668EA4CD52AFE5F2CF8E44367C55A82BFD91A5A34F`.
- Safe extraction selected only train, test, and RUL FD001 members and rejected
  changed archives, traversal, duplicates, and conflicting raw files.
- Train contains `20,631` rows/100 engines; test contains `13,096` rows/100
  engines with unique ordered `(unit_id, cycle)` keys and finite values.
- Train has non-negative uncapped and capped-at-125 RUL targets.
- Seed-42 split contains 80 development and 20 disjoint calibration engines.
  Official test labels remain isolated under `evaluation/`.
- Development-only EDA and quality reports were generated.
- Two prepare runs produced the same hashes for all eight processed artefacts.
- Data tests, Ruff, format, `git diff --check`, and the 80% coverage gate passed.
- No raw, processed, artifact, or run file is tracked.

## Decisions and blockers

- Fixed assumptions live in `PROJECT_PLAN.md`.
- A milestone is complete only after every deliverable and acceptance check
  passes.
