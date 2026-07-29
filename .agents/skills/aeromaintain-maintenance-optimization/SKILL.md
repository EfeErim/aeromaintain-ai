---
name: aeromaintain-maintenance-optimization
description: Build and validate AeroMaintain AI maintenance scenarios, baseline policies, and OR-Tools CP-SAT schedules. Use for synthetic fleet/resource generation, safe due dates, team/bay/parts/operating constraints, lexicographic objectives, solver statuses, capacity sensitivity, or true-RUL-isolated policy comparisons.
---

# AeroMaintain Maintenance Optimization

Turn locked predictions into an auditable schedule without exposing true test
RUL to the decision layer.

## Load the contract

Read `AGENTS.md`, `PROJECT_STATE.md`, the scenario and optimization sections of
`PROJECT_PLAN.md`, and
[references/optimization-contract.md](references/optimization-contract.md).
Require locked prediction and interval artefacts before using official test
engines.

## Generate the scenario

- Select the 20 engines with the lowest interval lower bounds; break ties by
  engine ID.
- Use the fixed 30-day horizon and seed 42.
- Generate cycles/day, maintenance duration, technician demand, kit type and
  quantity from the documented synthetic distributions.
- Create base resources: two bays, two teams of six technicians, starting
  inventory four per kit, replenishment three per kit at the start of days 10
  and 20, and 80% daily operating demand.
- Persist every synthetic value and its generator version.
- Assert that the optimizer input schema has no true-RUL field.

## Implement policies

- Evaluate reactive, fixed-90-cycle, predicted-RUL-30 and CP-SAT policies
  against the same scenario and evaluator.
- For rule policies, schedule triggered jobs on the earliest feasible day in
  ascending safe-due-day and engine-ID order; explicitly defer unschedulable
  jobs.
- Consume parts on the maintenance start day.
- Keep a job on the same team and bay for its full duration.

## Solve with CP-SAT

1. Use integer decision variables and integer cost units.
2. Compute `safe_due_day = floor(floor(interval_low) / cycles_per_day) - 1`,
   clamp negative values to day 0 and mark values beyond the horizon as not due
   in this horizon.
3. Enforce team capacity, bay no-overlap, daily parts balance, daily operating
   demand and completion within the horizon.
4. Solve stage 1 for due deferrals and late days.
5. Fix the best stage-1 score found, then solve stage 2 for planned cost, early
   cycles and low-risk deferral cost.
6. Set seed 42, one search worker and a 30-second limit per stage.
7. Preserve `OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, `MODEL_INVALID` or `UNKNOWN`.
   If stage 1 is not proven optimal, label the final schedule as feasible with
   unproven lexicographic optimality.
8. Never create a schedule for a no-solution status.

## Compare and verify

Run base, constrained and expanded capacity scenarios. Report failures, total
synthetic cost, early-cycle loss, deferrals, late days, utilization, operating
shortfall and solve time. Verify a hand-solvable 1–3-engine fixture and all
resource invariants. Use true RUL only in a separate retrospective evaluator.
Record evidence through `$aeromaintain-phase-gates`.
