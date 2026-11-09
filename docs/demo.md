# Demo: captured output

Every block below is real output, captured on 2026-11-09 from a fresh `git clone` of
this repository on Windows 11 with Python 3.11.9, after
`pip install -c constraints/python311-tested.txt -e ".[dev,ncmapss]"`. Paths are
shortened to `.`; nothing else is edited. The NASA archive was passed with `--archive`
instead of downloaded, which skips the download but not the size and SHA-256 check.

## Command line

```text
$ aeromaintain --help
                                                                               
 Usage: aeromaintain [OPTIONS] COMMAND [ARGS]...                               
                                                                               
 AeroMaintain AI educational RUL evaluation prototype.                         
                                                                               
+- Options -------------------------------------------------------------------+
| --help          Show this message and exit.                                 |
+-----------------------------------------------------------------------------+
+- Commands ------------------------------------------------------------------+
| prepare           Prepare one frozen NASA C-MAPSS subset.                   |
| train             Select, fit, calibrate, explain, persist, and lock an RUL |
|                   model.                                                    |
| evaluate          Evaluate one verified model lock against isolated         |
|                   official test labels.                                     |
| robustness        Repeat a locked specification across seeds on already     |
|                   seen test labels.                                         |
| escalation-study  Score a protocol amendment challenger without opening     |
|                   test labels.                                              |
| compare-runs      Report which files of two runs repeat byte for byte.      |
| pipeline          Run prepare, train/lock, evaluate, and report.            |
| finish            Open official labels for an existing lock and write its   |
|                   report.                                                   |
| export-demo       Export aggregate metrics and predictions, without NASA    |
|                   rows or true RUL.                                         |
| app               Open the local Streamlit application for one verified     |
|                   run.                                                      |
+-----------------------------------------------------------------------------+
```

## Tests

`torch` is an optional extra, so its one test is skipped.

```text
$ pytest -q
SKIPPED [1] tests\test_protocol_v2.py:441: could not import 'torch': No module named 'torch'
96 passed, 1 skipped in 53.05s
```

## End-to-end FD001 run

This is the README quickstart. It prepares the data, selects and locks a model, opens
the FD001 test labels and writes the report. It took about 11 minutes on CPU.
FD001 labels were seen before protocol v2, so this run shows that the pipeline works,
not a clean test result.

```text
$ time aeromaintain pipeline --run-id fd001-demo --archive data/raw/CMAPSSData.zip
AeroMaintain end-to-end pipeline complete
Run: fd001-demo
Manifest: .\runs\fd001-demo\manifest.json
Report: .\runs\fd001-demo\report.html
real	10m52.731s
```

The report summary was read with this snippet:

```python
import json
r = json.load(open('runs/fd001-demo/report.json'))
t = r['official_test']; b = t['bootstrap']['intervals_95']
print('dataset', r['dataset'], 'champion', r['model']['champion'], 'engines', t['engines'])
print('rmse %.2f (%.2f-%.2f)' % (t['rmse'], *b['rmse']))
print('mae %.2f (%.2f-%.2f)' % (t['mae'], *b['mae']))
print('interval coverage (%.2f-%.2f)' % tuple(b['interval_coverage']))
p = t['alert_rules']['point_prediction']
print('point alert rule: recall %.2f, precision %.2f, false alarms %d' % (p['recall'], p['precision'], p['false_positive']))
```

```text
$ python summarize_report.py   # script shown above
dataset FD001 champion xgboost engines 100
rmse 15.80 (12.87-18.78)
mae 11.15 (9.10-13.43)
interval coverage (0.89-0.98)
point alert rule: recall 0.80, precision 1.00, false alarms 0
```

These numbers match the FD001 row in the [model card](model_card.md): the run repeats.

## v5 bootstrap erratum

Recomputes the v5 ranges from the stored evaluation; see
[results_v5.md](results_v5.md#erratum-bootstrap-ranges-2026-11-09). It needs
`runs/v5/final`, which holds NASA rows and is not committed.

```text
$ python scripts/v5_bootstrap_erratum.py --output /tmp/erratum.json
rmse: 13.0568  published 8.9163-13.1450  corrected 11.1296-15.0029
baseline_rmse: 22.7036  published 20.4651-29.3352  corrected 21.4035-23.9861
mondrian_coverage: 0.9250  published 0.8800-0.9800  corrected 0.8836-0.9594
```
