# Demo: captured output

Every block below is real output, captured on 2026-11-09 from a fresh `git clone` of
this repository on Windows 11 with Python 3.11.9, after
`pip install -c constraints/python311-tested.txt -e ".[dev,ncmapss]"`. Paths are
shortened to `.`; nothing else is edited. The NASA archive was passed with `--archive`
instead of downloaded, which skips the download but not the size and SHA-256 check.

## End-to-end FD001 run

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

r = json.load(open("runs/fd001-demo/report.json"))
t = r["official_test"]
b = t["bootstrap"]["intervals_95"]
print(
    "dataset", r["dataset"], "champion", r["model"]["champion"], "engines", t["engines"]
)
print("rmse %.2f (%.2f-%.2f)" % (t["rmse"], *b["rmse"]))
print("mae %.2f (%.2f-%.2f)" % (t["mae"], *b["mae"]))
print("interval coverage (%.2f-%.2f)" % tuple(b["interval_coverage"]))
p = t["alert_rules"]["point_prediction"]
print(
    "point alert rule: recall %.2f, precision %.2f, false alarms %d"
    % (p["recall"], p["precision"], p["false_positive"])
)
```

```text
$ python summarize_report.py   # script shown above
dataset FD001 champion xgboost engines 100
rmse 15.80 (12.87-18.78)
mae 11.15 (9.10-13.43)
interval coverage (0.89-0.98)
point alert rule: recall 0.80, precision 1.00, false alarms 0
```

These match the `fd001-v3` column in [results](results.md#official-test): the run repeats on
the Windows machine that made it ([reproducibility](results.md#reproducibility)).
