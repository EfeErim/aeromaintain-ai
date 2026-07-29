# AeroMaintain AI Project State

Last updated: 2026-07-29

## Current position

- Active phase: Phase 1
- Active milestone: M1.1 — Safe data acquisition
- Overall status: Phase 0 complete; Phase 1 ready to start
- Current blocker: None

## Milestone register

| ID | Milestone | Status | Evidence |
|---|---|---|---|
| M0.1 | Project skeleton | complete | Python 3.11 editable install; package import; doctor 5/5 |
| M0.2 | Research and data governance | complete | `docs/research.md`; `docs/data_card.md`; NASA source and redistribution boundary |
| M0.3 | Quality infrastructure | complete | Ruff, pytest, coverage and Python 3.11 CI pass |
| M1.1 | Safe data acquisition | pending | — |
| M1.2 | Parser and data contract | pending | — |
| M1.3 | RUL target and engine split | pending | — |
| M1.4 | EDA and quality report | pending | — |
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
- `aeromaintain doctor` and `python -m aeromaintain doctor` passed all five
  environment checks.
- `docs/research.md` and `docs/data_card.md` record FD001 simulation status,
  primary method sources, label isolation, and raw-data redistribution limits.
- `ruff check .`, `ruff format --check .`, pytest and the 80% coverage gate
  passed.
- CI runs the same checks on Ubuntu and Python 3.11 without downloading NASA
  data.
- Raw, processed, artifact, and run paths are Git-ignored.

## Decisions and blockers

- Fixed assumptions live in `PROJECT_PLAN.md`.
- A milestone is complete only after every deliverable and acceptance check
  passes.
