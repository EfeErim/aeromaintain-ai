# Interface design notes

## Goal

The application is a compact experiment-review surface, not an invented fleet
control room. It should make model quality, uncertainty, run identity, and the
simulated-data boundary easy to find.

## Applied decisions

- Use two direct page names: `Overview` and `Engine risk`.
- Lead with the weak result as well as headline error metrics: critical recall
  remains visible beside MAE, RMSE, and interval coverage.
- Keep FD001's simulated-benchmark status visible.
- Use tables for engine priority and run details; charts support rather than
  replace exact values.
- Limit alert colors to risk meaning and use standard Streamlit components.
- Require an explicit verified run and expose only `risk_ranking.csv` for
  download; official true RUL is not part of the review table.

The tracked screenshots are captured from the verified reference run after the
page finishes rendering.
