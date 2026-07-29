"""Streamlit entrypoint for the verified AeroMaintain decision application."""

from __future__ import annotations

import os
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from aeromaintain.app.artifacts import (
    ArtifactValidationError,
    load_verified_run,
    run_capacity_what_if,
)

APP_CSS = """
<style>
    :root {
        --ink: #17211f;
        --muted: #64706d;
        --line: #d9ddda;
        --paper: #f5f6f2;
        --panel: #ffffff;
        --petrol: #123f3c;
        --teal: #167c73;
        --amber: #b56a21;
        --red: #9a3f3f;
    }

    [data-testid="stAppViewContainer"] {
        background: var(--paper);
        color: var(--ink);
    }
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    footer {
        display: none;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1240px;
        padding: 2.1rem 2.5rem 4rem;
    }
    [data-testid="stSidebar"] {
        background: var(--petrol);
        border-right: 0;
        min-width: 250px;
    }
    [data-testid="stSidebar"] * {
        color: #eef4f1;
    }
    [data-testid="stSidebarContent"] {
        padding: 1.75rem 1rem;
    }
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.25rem;
    }
    [data-testid="stSidebar"] label[data-baseweb="radio"] {
        min-height: 2.55rem;
        padding: 0.62rem 0.7rem;
        border-radius: 4px;
        transition: background 120ms ease;
    }
    [data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
        background: rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {
        background: #f4f2e9;
    }
    [data-testid="stSidebar"]
      label[data-baseweb="radio"]:has(input:checked) p {
        color: var(--petrol) !important;
        font-weight: 650;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: inherit;
    }

    h1, h2, h3 {
        color: var(--ink);
        font-family: "Segoe UI", Arial, sans-serif;
        letter-spacing: -0.025em;
    }
    h2 {
        font-size: 2rem !important;
        font-weight: 650 !important;
        margin: 0.15rem 0 0.25rem !important;
    }
    h3 {
        font-size: 1.08rem !important;
        font-weight: 650 !important;
        margin-top: 0.35rem !important;
    }
    p, label, button, input {
        font-family: "Segoe UI", Arial, sans-serif;
    }
    [data-testid="stCaptionContainer"] {
        color: var(--muted);
        font-size: 0.9rem;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 3px;
        overflow: hidden;
    }
    [data-testid="stExpander"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 3px;
    }
    [data-testid="stDownloadButton"] button,
    [data-testid="stFormSubmitButton"] button {
        border: 1px solid var(--petrol);
        border-radius: 3px;
        background: transparent;
        color: var(--petrol);
        font-weight: 650;
    }
    [data-testid="stDownloadButton"] button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        border-color: var(--petrol);
        background: var(--petrol);
        color: white;
    }

    .side-brand {
        border-bottom: 1px solid rgba(255,255,255,0.18);
        margin: 0 0 1.25rem;
        padding: 0 0.4rem 1.2rem;
    }
    .side-brand__mark {
        align-items: center;
        border: 1px solid rgba(255,255,255,0.55);
        display: inline-flex;
        font-family: Consolas, monospace;
        font-size: 0.78rem;
        height: 2rem;
        justify-content: center;
        letter-spacing: 0.08em;
        margin-bottom: 0.8rem;
        width: 2rem;
    }
    .side-brand strong {
        display: block;
        font-size: 1rem;
        letter-spacing: 0.01em;
    }
    .side-brand small {
        color: #b8cbc5 !important;
        display: block;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        margin-top: 0.18rem;
        text-transform: uppercase;
    }
    .side-run {
        border-top: 1px solid rgba(255,255,255,0.18);
        color: #b8cbc5 !important;
        font-family: Consolas, monospace;
        font-size: 0.67rem;
        line-height: 1.55;
        margin: 1.2rem 0.4rem 0;
        overflow-wrap: anywhere;
        padding-top: 1rem;
    }
    .side-run b {
        color: #ffffff !important;
        display: block;
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 0.67rem;
        letter-spacing: 0.1em;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
    }

    .masthead {
        align-items: center;
        border-bottom: 1px solid var(--line);
        display: flex;
        justify-content: space-between;
        margin-bottom: 1.4rem;
        padding-bottom: 0.75rem;
    }
    .masthead__product {
        color: var(--petrol);
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .masthead__status {
        align-items: center;
        color: var(--muted);
        display: flex;
        font-size: 0.73rem;
        gap: 0.45rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .masthead__status::before {
        background: var(--teal);
        border-radius: 50%;
        content: "";
        height: 7px;
        width: 7px;
    }
    .page-deck {
        color: var(--muted);
        font-size: 0.98rem;
        margin: -0.25rem 0 1.4rem;
        max-width: 760px;
    }

    .metric-strip {
        background: var(--panel);
        border: 1px solid var(--line);
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        margin: 0.5rem 0 1.5rem;
    }
    .metric-strip__item {
        border-right: 1px solid var(--line);
        min-height: 104px;
        padding: 1.05rem 1.1rem 0.9rem;
    }
    .metric-strip__item:last-child {
        border-right: 0;
    }
    .metric-strip__label {
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .metric-strip__value {
        color: var(--ink);
        font-family: Consolas, "Segoe UI", monospace;
        font-size: 1.75rem;
        line-height: 1.2;
        margin-top: 0.42rem;
    }
    .metric-strip__note {
        color: var(--muted);
        font-size: 0.72rem;
        margin-top: 0.18rem;
    }
    .metric-strip__value--warn {
        color: var(--amber);
    }
    .metric-strip__value--bad {
        color: var(--red);
    }

    .section-label {
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.11em;
        margin: 0.4rem 0 0.55rem;
        text-transform: uppercase;
    }
    .scope-note {
        border-left: 3px solid var(--amber);
        color: #5f4b37;
        font-size: 0.82rem;
        line-height: 1.5;
        margin: 1.1rem 0 1.4rem;
        padding: 0.55rem 0.8rem;
    }
    .status-line {
        align-items: center;
        background: var(--panel);
        border: 1px solid var(--line);
        display: flex;
        gap: 0.8rem;
        margin: 0.4rem 0 1.25rem;
        padding: 0.75rem 0.9rem;
    }
    .status-line__badge {
        border: 1px solid currentColor;
        color: var(--amber);
        font-family: Consolas, monospace;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        padding: 0.22rem 0.42rem;
    }
    .status-line__copy {
        color: var(--muted);
        font-size: 0.8rem;
    }
    .provenance {
        color: var(--muted);
        font-family: Consolas, monospace;
        font-size: 0.72rem;
        line-height: 1.7;
    }

    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.4rem 1rem 3rem;
        }
        .metric-strip {
            grid-template-columns: repeat(2, 1fr);
        }
        .metric-strip__item:nth-child(2) {
            border-right: 0;
        }
        .metric-strip__item:nth-child(-n+2) {
            border-bottom: 1px solid var(--line);
        }
    }
</style>
"""

PLOT_COLORS = ["#167c73", "#b56a21", "#637a75", "#9a3f3f"]


def _plot_style(figure, *, height: int = 330):
    figure.update_layout(
        template="plotly_white",
        colorway=PLOT_COLORS,
        font={"family": "Segoe UI, Arial, sans-serif", "color": "#28312f"},
        height=height,
        margin={"l": 12, "r": 12, "t": 26, "b": 12},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend_title_text="",
        hoverlabel={"font_family": "Segoe UI, Arial, sans-serif"},
    )
    figure.update_xaxes(gridcolor="#eaede9", zerolinecolor="#d9ddda")
    figure.update_yaxes(gridcolor="#eaede9", zerolinecolor="#d9ddda")
    return figure


def _page_header(title: str, deck: str) -> None:
    st.markdown(
        """
        <div class="masthead">
          <span class="masthead__product">AeroMaintain / Decision Console</span>
          <span class="masthead__status">Verified local evidence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.header(title, anchor=False)
    st.markdown(f'<div class="page-deck">{escape(deck)}</div>', unsafe_allow_html=True)


def _metric_strip(items: tuple[tuple[str, str, str, str], ...]) -> None:
    cells = "".join(
        (
            '<div class="metric-strip__item">'
            f'<div class="metric-strip__label">{escape(label)}</div>'
            f'<div class="metric-strip__value {escape(tone)}">{escape(value)}</div>'
            f'<div class="metric-strip__note">{escape(note)}</div>'
            "</div>"
        )
        for label, value, note, tone in items
    )
    st.markdown(f'<div class="metric-strip">{cells}</div>', unsafe_allow_html=True)


def _section_label(text: str) -> None:
    st.markdown(
        f'<div class="section-label">{escape(text)}</div>',
        unsafe_allow_html=True,
    )


def _download_buttons(artifacts, names: tuple[str, ...]) -> None:
    columns = st.columns(len(names))
    for column, name in zip(columns, names, strict=True):
        column.download_button(
            "Export CSV",
            artifacts.downloads[name],
            file_name=name,
            mime="text/csv",
            key=f"download-{name}",
            width="stretch",
        )


def _overview(artifacts) -> None:
    _page_header(
        "Fleet brief",
        "Locked model evidence and the current 30-day synthetic maintenance position.",
    )
    metrics = artifacts.metrics
    cp_sat = artifacts.policy_comparison.loc[
        artifacts.policy_comparison["policy"].eq("cp_sat")
    ].iloc[0]
    interval = metrics["nominal_empirical_interval"]
    _metric_strip(
        (
            ("Official-test RMSE", f"{metrics['rmse']:.2f}", "100 FD001 engines", ""),
            (
                "Interval coverage",
                f"{interval['observed_official_test_coverage']:.0%}",
                "90% nominal target",
                "metric-strip__value--warn",
            ),
            (
                "Maintenance placed",
                str(int(cp_sat["scheduled_maintenance"])),
                "20-engine synthetic fleet",
                "",
            ),
            (
                "Due deferrals",
                str(int(cp_sat["due_deferrals"])),
                "Visible planning exceptions",
                "metric-strip__value--bad",
            ),
        )
    )

    left, right = st.columns((1.15, 0.85), gap="large")
    risk_counts = (
        artifacts.risk_ranking["risk_band"]
        .value_counts()
        .reindex(["critical", "elevated", "routine"], fill_value=0)
        .rename_axis("Risk band")
        .reset_index(name="Engines")
    )
    risk_counts["Risk band"] = risk_counts["Risk band"].str.title()
    with left:
        _section_label("Official test fleet / interval-lower-bound risk")
        risk_chart = px.bar(
            risk_counts,
            x="Engines",
            y="Risk band",
            orientation="h",
            color="Risk band",
            color_discrete_map={
                "Critical": "#9a3f3f",
                "Elevated": "#b56a21",
                "Routine": "#167c73",
            },
            text="Engines",
        )
        risk_chart.update_layout(showlegend=False)
        risk_chart.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(_plot_style(risk_chart, height=270), width="stretch")
    with right:
        _section_label("Base plan / decision record")
        plan_rows = pd.DataFrame(
            [
                ("Solver state", str(cp_sat["solver_status"])),
                ("Optimality", str(cp_sat["lexicographic_optimality"])),
                ("Scheduled", int(cp_sat["scheduled_maintenance"])),
                ("Deferred", int(cp_sat.get("deferred_maintenance", 0))),
                ("Late days", int(cp_sat.get("late_days", 0))),
                (
                    "Synthetic total cost",
                    f"{int(cp_sat.get('total_synthetic_cost_units', 0)):,}",
                ),
            ],
            columns=["Measure", "Value"],
        )
        plan_rows["Value"] = plan_rows["Value"].astype(str)
        st.dataframe(
            plan_rows,
            hide_index=True,
            width="stretch",
            height=270,
            column_config={
                "Measure": st.column_config.TextColumn("Measure", width="medium"),
                "Value": st.column_config.TextColumn("Value", width="small"),
            },
        )

    st.markdown(
        """
        <div class="scope-note">
          Scope: NASA C-MAPSS FD001 is simulated. Staffing, bays, parts,
          duration and cost are synthetic planning assumptions. This console
          is not an airworthiness or maintenance-approval system.
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Release provenance"):
        lock_hash = artifacts.run_manifest["model_lock_sha256"]
        st.markdown(
            (
                '<div class="provenance">'
                f"RUN&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{escape(artifacts.run_id)}<br>"
                f"DATASET&nbsp;&nbsp;{escape(artifacts.model_lock['dataset'])}<br>"
                "MODEL&nbsp;&nbsp;&nbsp;&nbsp;"
                f"{escape(artifacts.model_lock['champion']['kind'])}<br>"
                f"SEED&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{artifacts.model_lock['seed']}<br>"
                f"LOCK&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{escape(lock_hash)}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def _engine_health(artifacts) -> None:
    _page_header(
        "Engine condition",
        "Risk ranking from the locked point estimate and nominal interval. "
        "Official true RUL is not exposed in this decision view.",
    )
    ranking = artifacts.risk_ranking.copy()
    counts = ranking["risk_band"].value_counts()
    _metric_strip(
        (
            ("Engines", str(len(ranking)), "Official FD001 test fleet", ""),
            (
                "Critical",
                str(int(counts.get("critical", 0))),
                "Interval lower bound ≤ 30",
                "metric-strip__value--bad",
            ),
            (
                "Elevated",
                str(int(counts.get("elevated", 0))),
                "Interval lower bound 31-60",
                "metric-strip__value--warn",
            ),
            (
                "Routine",
                str(int(counts.get("routine", 0))),
                "Interval lower bound > 60",
                "",
            ),
        )
    )

    _section_label("Priority queue")
    risk_view = ranking.rename(
        columns={
            "unit_id": "Engine",
            "cycle": "Observed cycle",
            "prediction": "Point RUL",
            "interval_low": "Interval low",
            "interval_high": "Interval high",
            "risk_band": "Risk band",
        }
    )
    st.dataframe(
        risk_view,
        hide_index=True,
        width="stretch",
        height=330,
        column_config={
            "Engine": st.column_config.NumberColumn(format="%d"),
            "Observed cycle": st.column_config.NumberColumn(format="%d"),
            "Point RUL": st.column_config.NumberColumn(format="%.1f"),
            "Interval low": st.column_config.ProgressColumn(
                format="%.1f", min_value=0, max_value=150
            ),
            "Interval high": st.column_config.NumberColumn(format="%.1f"),
            "Risk band": st.column_config.TextColumn(),
        },
    )
    _download_buttons(artifacts, ("risk_ranking.csv",))

    _section_label("Engine trace")
    selector, detail = st.columns((0.34, 0.66), gap="large")
    with selector:
        engine_id = st.selectbox(
            "Engine ID",
            ranking["unit_id"].astype(int).tolist(),
        )
        selected = ranking.loc[ranking["unit_id"].eq(engine_id)].iloc[0]
        selected_band = escape(str(selected["risk_band"]).upper())
        st.markdown(
            (
                '<div class="status-line">'
                f'<span class="status-line__badge">{selected_band}</span>'
                '<span class="status-line__copy">'
                f"Point RUL {selected['prediction']:.1f}<br>"
                f"Interval {selected['interval_low']:.1f}-"
                f"{selected['interval_high']:.1f}"
                "</span></div>"
            ),
            unsafe_allow_html=True,
        )
        sensors = st.multiselect(
            "Sensor channels",
            [
                column
                for column in artifacts.sensor_history.columns
                if column.startswith("sensor_")
            ],
            default=["sensor_2", "sensor_7", "sensor_12"],
            max_selections=5,
        )
    history = artifacts.sensor_history.loc[
        artifacts.sensor_history["unit_id"].eq(engine_id)
    ]
    with detail:
        if sensors:
            chart_data = history.loc[:, ["cycle", *sensors]].melt(
                id_vars="cycle", var_name="Sensor", value_name="Reading"
            )
            sensor_chart = px.line(
                chart_data,
                x="cycle",
                y="Reading",
                color="Sensor",
            )
            sensor_chart.update_layout(xaxis_title="Cycle", yaxis_title="Reading")
            st.plotly_chart(_plot_style(sensor_chart, height=360), width="stretch")

    importance = pd.DataFrame(artifacts.explanation["global_importance"]).head(12)
    _section_label("Locked model behavior / global coefficient magnitude")
    explanation_chart = px.bar(
        importance.sort_values("coefficient"),
        x="coefficient",
        y="feature",
        orientation="h",
        color="coefficient",
        color_continuous_scale=["#9a3f3f", "#e3e7e3", "#167c73"],
    )
    explanation_chart.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Standardized coefficient",
        yaxis_title="",
    )
    st.plotly_chart(_plot_style(explanation_chart, height=420), width="stretch")
    st.caption(
        "Coefficients describe model behavior, not physical causality or diagnosis."
    )


def _schedule(artifacts) -> None:
    _page_header(
        "30-day work plan",
        "The base synthetic schedule, resource assignment and unresolved exceptions.",
    )
    cp_sat = artifacts.policy_comparison.loc[
        artifacts.policy_comparison["policy"].eq("cp_sat")
    ].iloc[0]
    _metric_strip(
        (
            (
                "Solver",
                str(cp_sat["solver_status"]),
                "Two-stage CP-SAT",
                "metric-strip__value--warn",
            ),
            (
                "Scheduled",
                str(int(cp_sat["scheduled_maintenance"])),
                "Maintenance jobs placed",
                "",
            ),
            (
                "Due deferrals",
                str(int(cp_sat["due_deferrals"])),
                "Require operator review",
                "metric-strip__value--bad",
            ),
            (
                "Capacity shortfall",
                str(int(cp_sat.get("operating_capacity_shortfall_cycles", 0))),
                "Synthetic cycles",
                "",
            ),
        )
    )
    solver_status = escape(str(cp_sat["solver_status"]))
    optimality = escape(str(cp_sat["lexicographic_optimality"]))
    st.markdown(
        (
            '<div class="status-line">'
            f'<span class="status-line__badge">{solver_status}</span>'
            '<span class="status-line__copy">'
            f"Lexicographic optimality is {optimality}. "
            "A feasible result is not presented as optimal."
            "</span></div>"
        ),
        unsafe_allow_html=True,
    )

    scheduled = artifacts.schedule.loc[
        artifacts.schedule["status"].eq("scheduled")
    ].copy()
    deferred = artifacts.schedule.loc[
        artifacts.schedule["status"].eq("deferred")
    ].copy()
    if not scheduled.empty:
        scheduled["start"] = pd.to_datetime(
            scheduled["start_day"], unit="D", origin="2026-01-01"
        )
        scheduled["finish"] = pd.to_datetime(
            scheduled["end_day"], unit="D", origin="2026-01-01"
        )
        scheduled["engine"] = "Engine " + scheduled["engine_id"].astype(str)
        _section_label("Assignment timeline")
        timeline = px.timeline(
            scheduled,
            x_start="start",
            x_end="finish",
            y="engine",
            color="team_id",
            hover_data=["bay_id"],
            color_discrete_map={"team_A": "#167c73", "team_B": "#b56a21"},
        )
        timeline.update_yaxes(autorange="reversed", title="")
        timeline.update_xaxes(title="Planning day")
        st.plotly_chart(_plot_style(timeline, height=500), width="stretch")

    left, right = st.columns((0.42, 0.58), gap="large")
    with left:
        _section_label("Unresolved exceptions")
        if deferred.empty:
            st.caption("No deferred maintenance.")
        else:
            st.dataframe(
                deferred.loc[:, ["engine_id", "reason"]].rename(
                    columns={"engine_id": "Engine", "reason": "Reason"}
                ),
                hide_index=True,
                width="stretch",
            )
    with right:
        _section_label("Resource position")
        resource = pd.DataFrame(
            [
                ("Team utilization", f"{cp_sat.get('team_utilization', 0):.0%}"),
                ("Bay utilization", f"{cp_sat.get('bay_utilization', 0):.0%}"),
                ("Late days", int(cp_sat.get("late_days", 0))),
                ("Early cycles lost", int(cp_sat.get("early_cycles_lost", 0))),
            ],
            columns=["Measure", "Value"],
        )
        resource["Value"] = resource["Value"].astype(str)
        st.dataframe(resource, hide_index=True, width="stretch")

    with st.expander("Full assignment ledger"):
        st.dataframe(artifacts.schedule, hide_index=True, width="stretch")
        _download_buttons(artifacts, ("cp_sat_schedule.csv",))


def _comparison(artifacts) -> None:
    _page_header(
        "Scenario lab",
        "Policy outcomes, capacity sensitivity and a bounded truth-free replan.",
    )
    policy = artifacts.policy_comparison.copy()
    capacity = artifacts.capacity_comparison.copy()
    for column in (
        "scheduled_maintenance",
        "due_deferrals",
        "total_synthetic_cost_units",
        "unplanned_failures",
        "solve_time_seconds",
    ):
        if column not in policy:
            policy[column] = 0
    for column in (
        "scheduled_maintenance",
        "due_deferrals",
        "unplanned_failures",
    ):
        if column not in capacity:
            capacity[column] = 0
    cp_sat = policy.loc[policy["policy"].eq("cp_sat")].iloc[0]
    _metric_strip(
        (
            (
                "CP-SAT cost",
                f"{int(cp_sat['total_synthetic_cost_units']):,}",
                "Synthetic cost units",
                "",
            ),
            (
                "Simulated failures",
                str(int(cp_sat["unplanned_failures"])),
                "Retrospective only",
                "metric-strip__value--bad",
            ),
            (
                "Base due deferrals",
                str(int(cp_sat["due_deferrals"])),
                "2 bays / 80% demand",
                "metric-strip__value--warn",
            ),
            (
                "Solve time",
                f"{cp_sat['solve_time_seconds']:.1f}s",
                str(cp_sat["solver_status"]),
                "",
            ),
        )
    )

    policy_labels = {
        "reactive": "Reactive",
        "fixed_90": "Fixed 90",
        "predicted_rul_30": "Predicted RUL 30",
        "cp_sat": "CP-SAT",
    }
    policy["Policy"] = policy["policy"].map(policy_labels)
    left, right = st.columns(2, gap="large")
    with left:
        _section_label("Synthetic total cost by policy")
        cost_chart = px.bar(
            policy,
            x="Policy",
            y="total_synthetic_cost_units",
            color="Policy",
            color_discrete_sequence=PLOT_COLORS,
            text="total_synthetic_cost_units",
        )
        cost_chart.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Synthetic cost units",
        )
        st.plotly_chart(_plot_style(cost_chart), width="stretch")
    with right:
        _section_label("Maintenance placement and deferral")
        decision_chart_data = policy.melt(
            id_vars="Policy",
            value_vars=["scheduled_maintenance", "due_deferrals"],
            var_name="Decision",
            value_name="Engines",
        )
        decision_chart_data["Decision"] = decision_chart_data["Decision"].map(
            {
                "scheduled_maintenance": "Scheduled",
                "due_deferrals": "Due deferrals",
            }
        )
        decision_chart = px.bar(
            decision_chart_data,
            x="Policy",
            y="Engines",
            color="Decision",
            barmode="group",
            color_discrete_map={
                "Scheduled": "#167c73",
                "Due deferrals": "#9a3f3f",
            },
        )
        decision_chart.update_layout(xaxis_title="", yaxis_title="Engines")
        st.plotly_chart(_plot_style(decision_chart), width="stretch")

    _section_label("Capacity sensitivity")
    capacity_order = ["constrained", "base", "expanded"]
    capacity["capacity_scenario"] = pd.Categorical(
        capacity["capacity_scenario"],
        categories=capacity_order,
        ordered=True,
    )
    capacity_plot = capacity.sort_values("capacity_scenario").melt(
        id_vars="capacity_scenario",
        value_vars=["scheduled_maintenance", "due_deferrals", "unplanned_failures"],
        var_name="Measure",
        value_name="Engines",
    )
    capacity_plot["Measure"] = capacity_plot["Measure"].map(
        {
            "scheduled_maintenance": "Scheduled",
            "due_deferrals": "Due deferrals",
            "unplanned_failures": "Simulated failures",
        }
    )
    capacity_chart = px.line(
        capacity_plot,
        x="capacity_scenario",
        y="Engines",
        color="Measure",
        markers=True,
        color_discrete_map={
            "Scheduled": "#167c73",
            "Due deferrals": "#9a3f3f",
            "Simulated failures": "#b56a21",
        },
    )
    capacity_chart.update_layout(
        xaxis_title="Capacity scenario",
        yaxis_title="Engines",
    )
    st.plotly_chart(_plot_style(capacity_chart, height=350), width="stretch")

    with st.expander("Underlying comparison tables"):
        st.dataframe(policy.drop(columns="Policy"), hide_index=True, width="stretch")
        st.dataframe(capacity, hide_index=True, width="stretch")
        _download_buttons(
            artifacts,
            ("policy_comparison.csv", "capacity_comparison.csv"),
        )

    _section_label("Bounded capacity replan")
    st.caption(
        "The run remains immutable. This solve uses the verified truth-free "
        "scenario in memory and does not write into the release run."
    )
    with st.form("capacity-what-if", border=True):
        control_left, control_right = st.columns(2)
        bays = control_left.slider("Maintenance bays", 1, 3, 2)
        demand_percent = control_right.slider(
            "Minimum operating demand (%)", 70, 90, 80
        )
        submitted = st.form_submit_button(
            "Run capacity replan",
            width="stretch",
        )
    if submitted:
        try:
            schedule, result = run_capacity_what_if(
                artifacts,
                bays=bays,
                operating_demand_fraction=demand_percent / 100,
            )
        except ArtifactValidationError as exc:
            st.error(str(exc))
        else:
            _metric_strip(
                (
                    (
                        "Solver",
                        str(result["solver_status"]),
                        "Truth-free what-if",
                        "metric-strip__value--warn",
                    ),
                    (
                        "Scheduled",
                        str(int(result.get("scheduled_maintenance", 0))),
                        "Maintenance jobs",
                        "",
                    ),
                    (
                        "Due deferrals",
                        str(int(result.get("due_deferrals", 0))),
                        "Planning exceptions",
                        "metric-strip__value--bad",
                    ),
                    (
                        "Shortfall",
                        str(int(result.get("operating_capacity_shortfall_cycles", 0))),
                        "Synthetic cycles",
                        "",
                    ),
                )
            )
            with st.expander("What-if assignment ledger"):
                st.dataframe(
                    pd.DataFrame(schedule.get("jobs", [])),
                    hide_index=True,
                    width="stretch",
                )


def main() -> None:
    """Render the four-page decision application."""
    st.set_page_config(
        page_title="AeroMaintain | Decision Console",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    run_id = os.environ.get("AEROMAINTAIN_RUN_ID", "")
    project_root = Path(os.environ.get("AEROMAINTAIN_PROJECT_ROOT", Path.cwd()))
    try:
        artifacts = load_verified_run(project_root, run_id)
    except ArtifactValidationError as exc:
        st.error(f"Verified run could not be loaded: {exc}")
        st.stop()

    st.sidebar.markdown(
        """
        <div class="side-brand">
          <span class="side-brand__mark">AM</span>
          <strong>AeroMaintain</strong>
          <small>Maintenance decision console</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pages = (
        "Fleet brief",
        "Engine condition",
        "30-day work plan",
        "Scenario lab",
    )
    page = st.sidebar.radio(
        "Workspace",
        pages,
        label_visibility="collapsed",
    )
    st.sidebar.markdown(
        (f'<div class="side-run"><b>Verified run</b>{escape(artifacts.run_id)}</div>'),
        unsafe_allow_html=True,
    )
    if page == "Fleet brief":
        _overview(artifacts)
    elif page == "Engine condition":
        _engine_health(artifacts)
    elif page == "30-day work plan":
        _schedule(artifacts)
    else:
        _comparison(artifacts)


if __name__ == "__main__":
    main()
