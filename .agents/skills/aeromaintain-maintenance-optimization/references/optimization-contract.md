# Maintenance Optimization Contract

Checked: 2026-07-29

## Primary documentation

- OR-Tools CP-SAT:
  https://developers.google.com/optimization/cp/cp_solver
- Constraint programming and scheduling:
  https://developers.google.com/optimization/cp
- Solver time limits:
  https://developers.google.com/optimization/cp/cp_tasks

CP-SAT models operate on integer variables. The documented result states are
`OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, `MODEL_INVALID` and `UNKNOWN`. A time
limit prevents an interactive demo from waiting indefinitely, but a feasible
status does not prove optimality.

## Fixed scenario values

- Fleet: 20 lowest interval lower bounds
- Horizon: 30 days
- Seed: 42
- Cycles/day: integer 1–4
- Maintenance duration: integer 2–5 days
- Technician demand: integer 2–6
- Kit: `kit_A`, `kit_B` or `kit_C`
- Kit quantity: integer 1–2
- Planned maintenance cost: 100
- Emergency maintenance cost: 500
- Early-cycle cost: 1 per predicted unused cycle
- Low-risk horizon deferral cost: 150

Capacity scenarios:

| Scenario | Bays | Minimum operating demand |
|---|---:|---:|
| constrained | 1 | 90% |
| base | 2 | 80% |
| expanded | 3 | 70% |

Use two teams of six technicians in every scenario. Do not add overtime in V1.

## Truth boundary

The scenario generator, baseline policies and solver may use prediction,
interval, engine ID and synthetic fields only. The retrospective evaluator may
join true RUL after the schedule is frozen. Test this boundary by validating
the optimizer input schema and by passing a sentinel true-RUL column that must
be rejected.
