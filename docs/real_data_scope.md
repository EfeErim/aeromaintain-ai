# Real-data scope review

Checked: 2026-08-04

## Question

Can the repository replace its invented maintenance duration, technicians,
bays, parts, operating demand, and cost values with one defensible public
aviation dataset?

## Sources reviewed

### NASA C-MAPSS FD001

[NASA Open Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
describes C-MAPSS as simulated run-to-failure turbofan data. It supplies the
sensor histories and RUL benchmark used here, but no observed maintenance work,
staffing, bay, parts, demand, or cost records.

### NGAFID maintenance dataset

The [NGAFID dataset paper](https://arxiv.org/abs/2210.07317) documents real,
non-simulated general-aviation flight data: 28,935 flights, 31,177 flight hours,
2,111 unplanned maintenance events, and 36 issue types. The
[Zenodo record](https://zenodo.org/records/6624956) publishes an open CC BY 4.0
dataset, including a 1.1 GB two-day subset and a 4.3 GB full dataset. The
[companion repository](https://github.com/hyang0129/NGAFIDDATASET) provides the
download and preprocessing code.

This is a credible real-data candidate for a separate predictive-maintenance
study. It does not provide the coherent technicians, bay capacity, parts
inventory, operating-demand, work duration, and cost schema required by the
retired scheduling problem. Its published task is also a different target:
whether an unplanned maintenance event is more than two days away, rather than
FD001 cycle-level RUL regression.

### FAA Service Difficulty Reports

The [FAA Aviation Information portal](https://www.faa.gov/av-info) provides
Service Difficulty Reports about aircraft malfunctions, defects, and maintenance
findings. These are useful event records, but not a resource-constrained work
schedule with the operational fields required here.

### US BTS maintenance costs

The [Bureau of Transportation Statistics maintenance-cost view](https://www.bts.gov/data-spotlight/aircraft-maintenance)
reports aggregate quarterly maintenance cost per flight hour by aircraft type.
It does not link event-level sensor history to job duration, people, bays,
parts, capacity, and cost.

## Decision

No reviewed source supports the complete operational scheduling schema without
inventing or joining major fields under unverified assumptions. Mixing these
sources would create a plausible-looking table, not an observed operational
dataset.

The active repository therefore keeps only the FD001 RUL evaluation and removes:

- synthetic maintenance scenario generation;
- maintenance policy and synthetic cost comparison;
- CP-SAT resource scheduling and capacity what-if analysis;
- related configuration, UI pages, public evidence, and dependencies.

NGAFID remains a possible new project, subject to a fresh target definition,
data contract, compute budget, and evaluation protocol. It is not silently
substituted into this repository.
