# Interface design notes

## Goal

The application is a compact experiment-review surface, not an invented fleet
control room. It should make model quality, uncertainty, run identity, and the
simulated-data boundary easy to find.

## Applied decisions

- Use two direct page names: `Overview` and `Engine risk`.
- Show both alert rules side by side with their ranges, so the cost of the interval
  rule stays visible beside its recall.
- Keep each run's dataset, simulated-benchmark status, and protocol label visible, so
  a previously seen test cannot pass as a clean one.
- Use tables for engine priority and run details; charts support rather than
  replace exact values.
- Limit alert colors to risk meaning and use standard Streamlit components.
- Require an explicit verified run and expose only `risk_ranking.csv` for
  download; official true RUL is not part of the review table.

The tracked screenshots are captured from the verified `fd004-v3-clean` run after
the page finishes rendering. The hosted demo reads the same run and hides sensor history because NASA's
dataset page specifies no license.
