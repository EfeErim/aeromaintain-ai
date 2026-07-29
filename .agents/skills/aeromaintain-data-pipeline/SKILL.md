---
name: aeromaintain-data-pipeline
description: Build and validate the AeroMaintain AI NASA C-MAPSS FD001 data pipeline. Use for data download, SHA-256 verification, safe ZIP extraction, 26-column parsing, RUL target creation, development/calibration engine splits, data quality checks, EDA inputs, or related tests and documentation.
---

# AeroMaintain Data Pipeline

Implement a deterministic, fail-closed FD001 preparation workflow.

## Load the contract

Read `AGENTS.md`, `PROJECT_STATE.md`, the data sections of `PROJECT_PLAN.md`,
and [references/data-contract.md](references/data-contract.md). Inspect current
code and tests before changing them.

## Implement in order

1. Download only from the configured NASA URL into `data/raw/`.
2. Verify the complete ZIP SHA-256 before extraction. Reject mismatches.
3. Extract with a path-traversal guard and select only FD001 train, test and RUL
   members.
4. Parse whitespace-separated rows into the fixed 26-column schema with
   explicit numeric types.
5. Validate row counts, engine counts, cycle ordering, key uniqueness, nulls and
   finite values.
6. Create uncapped train RUL and the capped modeling target
   `min(true_rul, 125)`.
7. Create the deterministic development/calibration engine split from engine
   lifetime quartiles using seed 42.
8. Write processed tables, split metadata, quality metrics and their hashes
   without overwriting an existing run.

## Preserve label isolation

- Data preparation may store the official test RUL file in a protected
  evaluation location.
- Training, feature selection and model-selection modules must not import or
  read that file.
- Expose official test labels only through the locked evaluation entrypoint.
- Never copy raw NASA members into Git-tracked fixtures. Build small synthetic
  fixtures for tests.

## Verify

Run focused data tests covering hash failure, malicious ZIP paths, schema
failure, expected FD001 dimensions, duplicate keys, negative RUL, split
disjointness and deterministic output hashes. Record evidence through
`$aeromaintain-phase-gates`.
