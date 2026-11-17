# Documentation map

Start with the top-level [README](../README.md). This page says which document answers
which question, and which ones are kept only as a record.

The protocol files cannot be merged or moved: model locks record their paths and
SHA-256 hashes, so they stay exactly where and as they were when each test ran.

## Read these

| Question | Document |
|---|---|
| What was run, in what order, and how did each test turn out? | [Protocol history](protocol_history.md) |
| Final N-CMAPSS test, including the erratum and sensitivity checks | [Results v5](results_v5.md) |
| C-MAPSS clean tests (FD003, FD002, FD004) and challengers | [Results v2 and v3](results.md) |
| How the system is built, and what "opened once" does and does not prove | [Architecture](architecture.md) |
| What the model is for and where it fails | [Model card](model_card.md) |
| Where the data comes from and what is published | [Data card](data_card.md) |
| Why these methods | [Research notes](research.md) |
| What a fresh clone prints | [Demo output](demo.md) |

## Kept as a record

| Document | Why it exists |
|---|---|
| [Protocol v2](protocol_v2.md), [v3](protocol_v3.md), [v5](protocol_v5.md) | Frozen protocols; hashed into their locks |
| [Protocol v4](protocol_v4.md), [initial](protocol_v4_initial.md), [amendment 1](protocol_v4_amendment1.md) | v4 as finally run, as first written, and the amendment between them |
| [Results v4](results_v4.md) | Negative result: removing the RUL cap; development evidence only |
| [Retrospective](results_retrospective.md) | v3 rescored on the literature's capped scale after labels were open |
| `*.json` next to these pages | Machine-readable evidence that the pages cite, with hashes |
| `ncmapss_locks/` | Published locks and the holdout reservation |
