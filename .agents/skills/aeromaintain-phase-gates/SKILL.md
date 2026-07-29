---
name: aeromaintain-phase-gates
description: Coordinate AeroMaintain AI milestones, phase gates, project status, and release readiness. Use when deciding the next project step, executing or closing a milestone, reporting progress, checking deliverables, updating PROJECT_STATE.md, or validating that a phase is complete.
---

# AeroMaintain Phase Gates

Keep project progress evidence-based and sequential.

## Establish the current milestone

1. Read `AGENTS.md`, `PROJECT_STATE.md` and the phase sections of
   `PROJECT_PLAN.md`.
2. Inspect the repository and existing artefacts. Do not trust status text
   without matching files and checks.
3. Select the earliest incomplete milestone unless the user explicitly names a
   different one.
4. If a requested milestone depends on an incomplete earlier gate, finish or
   report that dependency first.
5. Route domain work through the relevant AeroMaintain skill named in
   `AGENTS.md`.

## Execute a milestone

1. State the milestone ID, deliverables and acceptance checks before editing.
2. Limit changes to the selected milestone and required prerequisites.
3. Preserve unrelated user work and existing run artefacts.
4. Run focused validation before broad validation.
5. Compare observed results with every acceptance criterion in
   `PROJECT_PLAN.md`.

## Close or block a milestone

- Mark `complete` only when deliverables exist and every required check passes.
- Record exact commands and the smallest useful evidence in `PROJECT_STATE.md`.
- Keep a milestone `in_progress` when implementation exists but verification is
  incomplete.
- Mark `blocked` only with the exact failure, its cause if known, and the next
  action that can unblock it.
- Never infer success from a command that was not run.
- Never advance the active phase until all milestones in the current phase are
  complete.

## Release gate

For M4.4, require a clean Python 3.11 installation check, end-to-end pipeline,
Ruff, tests with the required coverage, Streamlit smoke test, artefact hash
validation and a raw-data Git exclusion check. Preserve negative or infeasible
results in the release evidence.
