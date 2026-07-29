"""Streamlit entrypoint for the verified AeroMaintain decision application."""

from __future__ import annotations

import os
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
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid #d9dee3;
    }
    [data-testid="stSidebarContent"] {
        padding-top: 1.5rem;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #d9dee3;
    }
    h1, h2, h3 {
        letter-spacing: -0.01em;
    }
    h2 {
        font-size: 1.8rem !important;
    }
    h3 {
        font-size: 1.1rem !important;
        margin-top: 1.25rem !important;
    }
</style>
"""

PLOT_COLORS = ["#2F5D8A", "#6F7F8F", "#B35C44", "#7A6F9B"]
RISK_COLORS = {
    "Critical": "#B42318",
    "Elevated": "#B26A00",
    "Routine": "#667085",
}


def _plot_style(figure, *, height: int = 330):
    figure.update_layout(
        template="plotly_white",
        colorway=PLOT_COLORS,
        font={"family": "Arial, sans-serif", "color": "#27313B"},
        height=height,
        margin={"l": 16, "r": 16, "t": 20, "b": 16},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend_title_text="",
        hoverlabel={"font_family": "Arial, sans-serif"},
    )
    figure.update_xaxes(gridcolor="#E8EBEE", zeroline=False)
    figure.update_yaxes(gridcolor="#E8EBEE", zeroline=False)
    return figure


def _page_header(title: str, summary: str) -> None:
    st.header(title, anchor=False)
    st.caption(summary)


def _metric_row(items: tuple[tuple[str, str, str], ...]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items, strict=True):
        column.metric(label, value, help=help_text)


def _download_buttons(artifacts, names: tuple[str, ...]) -> None:
    columns = st.columns(len(names))
    for column, name in zip(columns, names, strict=True):
        column.download_button(
            f"Download {name}",
            artifacts.downloads[name],
            file_name=name,
            mime="text/csv",
            key=f"download-{name}",
            width="stretch",
        )


def _overview(artifacts) -> None:
    _page_header(
        "Overview",
        "Locked model results and the current 30-day synthetic maintenance plan.",
    )
    metrics = artifacts.metrics
    interval = metrics["nominal_empirical_interval"]
    cp_sat = artifacts.policy_comparison.loc[
        artifacts.policy_comparison["policy"].eq("cp_sat")
    ].iloc[0]

    _metric_row(
        (
            (
                "Official-test RMSE",
                f"{metrics['rmse']:.2f}",
                "Measured on 100 FD001 test engines after model lock.",
            ),
            (
                "Interval coverage",
                f"{interval['observed_official_test_coverage']:.0%}",
                "Observed coverage for the nominal 90% empirical interval.",
            ),
            (
                "Maintenance scheduled",
                str(int(cp_sat["scheduled_maintenance"])),
                "Jobs placed in the 30-day synthetic plan.",
            ),
            (
                "Due jobs deferred",
                str(int(cp_sat["due_deferrals"])),
                "Due jobs that could not be placed under base capacity.",
            ),
        )
    )

    st.info(
        "NASA C-MAPSS FD001 is simulated. Staffing, bays, parts, duration and "
        "cost are synthetic assumptions. This prototype is not an airworthiness "
        "or maintenance-approval system."
    )

    risk_counts = (
        artifacts.risk_ranking["risk_band"]
        .value_counts()
        .reindex(["critical", "elevated", "routine"], fill_value=0)
        .rename_axis("Risk band")
        .reset_index(name="Engines")
    )
    risk_counts["Risk band"] = risk_counts["Risk band"].str.title()

    left, right = st.columns((1.1, 0.9), gap="large")
    with left:
        st.subheader("Fleet risk distribution")
        risk_chart = px.bar(
            risk_counts,
            x="Engines",
            y="Risk band",
            orientation="h",
            color="Risk band",
            color_discrete_map=RISK_COLORS,
            text="Engines",
        )
        risk_chart.update_layout(showlegend=False)
        risk_chart.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(_plot_style(risk_chart, height=270), width="stretch")

    with right:
        st.subheader("Base plan summary")
        plan_rows = pd.DataFrame(
            [
                ("Solver status", str(cp_sat["solver_status"])),
                ("Optimality", str(cp_sat["lexicographic_optimality"])),
                ("Scheduled jobs", int(cp_sat["scheduled_maintenance"])),
                ("Deferred jobs", int(cp_sat.get("deferred_maintenance", 0))),
                ("Late days", int(cp_sat.get("late_days", 0))),
                (
                    "Total synthetic cost",
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

    with st.expander("Run details"):
        identity = pd.DataFrame(
            [
                ("Run", artifacts.run_id),
                ("Dataset", artifacts.model_lock["dataset"]),
                ("Champion model", artifacts.model_lock["champion"]["kind"]),
                ("Seed", artifacts.model_lock["seed"]),
                (
                    "Interval",
                    artifacts.model_lock["calibration"]["label"],
                ),
                (
                    "Model lock SHA-256",
                    artifacts.run_manifest["model_lock_sha256"],
                ),
            ],
            columns=["Field", "Value"],
        )
        identity["Value"] = identity["Value"].astype(str)
        st.dataframe(identity, hide_index=True, width="stretch")


def _engine_risk(artifacts) -> None:
    _page_header(
        "Engine risk",
        "Prioritization uses the locked prediction and nominal interval. "
        "Official true RUL is not available in this decision view.",
    )
    ranking = artifacts.risk_ranking.copy()
    counts = ranking["risk_band"].value_counts()
    _metric_row(
        (
            ("Engines", str(len(ranking)), "Official FD001 test fleet."),
            (
                "Critical",
                str(int(counts.get("critical", 0))),
                "Interval lower bound at or below 30 cycles.",
            ),
            (
                "Elevated",
                str(int(counts.get("elevated", 0))),
                "Interval lower bound from 31 to 60 cycles.",
            ),
            (
                "Routine",
                str(int(counts.get("routine", 0))),
                "Interval lower bound above 60 cycles.",
            ),
        )
    )

    st.subheader("Priority list")
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
            "Interval low": st.column_config.NumberColumn(format="%.1f"),
            "Interval high": st.column_config.NumberColumn(format="%.1f"),
            "Risk band": st.column_config.TextColumn(),
        },
    )
    _download_buttons(artifacts, ("risk_ranking.csv",))

    st.subheader("Selected engine")
    selector, chart_column = st.columns((0.32, 0.68), gap="large")
    with selector:
        engine_id = st.selectbox(
            "Engine ID",
            ranking["unit_id"].astype(int).tolist(),
        )
        selected = ranking.loc[ranking["unit_id"].eq(engine_id)].iloc[0]
        st.metric("Point RUL", f"{selected['prediction']:.1f} cycles")
        low, high = st.columns(2)
        low.metric("Interval low", f"{selected['interval_low']:.1f}")
        high.metric("Interval high", f"{selected['interval_high']:.1f}")
        st.caption(f"Risk band: {str(selected['risk_band']).title()}")
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
    with chart_column:
        if sensors:
            chart_data = history.loc[:, ["cycle", *sensors]].melt(
                id_vars="cycle",
                var_name="Sensor",
                value_name="Reading",
            )
            sensor_chart = px.line(
                chart_data,
                x="cycle",
                y="Reading",
                color="Sensor",
            )
            sensor_chart.update_layout(
                xaxis_title="Cycle",
                yaxis_title="Reading",
            )
            st.plotly_chart(
                _plot_style(sensor_chart, height=350),
                width="stretch",
            )

    st.subheader("Model coefficients")
    st.caption(
        "Standardized coefficients describe locked model behavior, not physical "
        "causality or diagnosis."
    )
    importance = pd.DataFrame(artifacts.explanation["global_importance"]).head(12)
    coefficient_chart = px.bar(
        importance.sort_values("coefficient"),
        x="coefficient",
        y="feature",
        orientation="h",
    )
    coefficient_chart.update_traces(marker_color="#2F5D8A")
    coefficient_chart.update_layout(
        xaxis_title="Standardized coefficient",
        yaxis_title="",
    )
    st.plotly_chart(
        _plot_style(coefficient_chart, height=420),
        width="stretch",
    )


def _maintenance_plan(artifacts) -> None:
    _page_header(
        "Maintenance plan",
        "Base synthetic schedule, resource assignment and unresolved work.",
    )
    cp_sat = artifacts.policy_comparison.loc[
        artifacts.policy_comparison["policy"].eq("cp_sat")
    ].iloc[0]
    _metric_row(
        (
            (
                "Solver status",
                str(cp_sat["solver_status"]),
                "Two-stage CP-SAT result.",
            ),
            (
                "Scheduled jobs",
                str(int(cp_sat["scheduled_maintenance"])),
                "Maintenance jobs placed.",
            ),
            (
                "Due jobs deferred",
                str(int(cp_sat["due_deferrals"])),
                "Planning exceptions requiring review.",
            ),
            (
                "Capacity shortfall",
                str(int(cp_sat.get("operating_capacity_shortfall_cycles", 0))),
                "Synthetic operating cycles below demand.",
            ),
        )
    )

    st.warning(
        f"Solver status is {cp_sat['solver_status']}; lexicographic optimality "
        f"is {cp_sat['lexicographic_optimality']}. A feasible schedule is not "
        "presented as proven optimal."
    )

    scheduled = artifacts.schedule.loc[
        artifacts.schedule["status"].eq("scheduled")
    ].copy()
    deferred = artifacts.schedule.loc[
        artifacts.schedule["status"].eq("deferred")
    ].copy()
    if not scheduled.empty:
        scheduled["start"] = pd.to_datetime(
            scheduled["start_day"],
            unit="D",
            origin="2026-01-01",
        )
        scheduled["finish"] = pd.to_datetime(
            scheduled["end_day"],
            unit="D",
            origin="2026-01-01",
        )
        scheduled["engine"] = "Engine " + scheduled["engine_id"].astype(str)
        st.subheader("Assignment timeline")
        timeline = px.timeline(
            scheduled,
            x_start="start",
            x_end="finish",
            y="engine",
            color="team_id",
            hover_data=["bay_id"],
            color_discrete_map={
                "team_A": "#2F5D8A",
                "team_B": "#7A6F9B",
            },
        )
        timeline.update_yaxes(autorange="reversed", title="")
        timeline.update_xaxes(title="Planning day")
        st.plotly_chart(_plot_style(timeline, height=500), width="stretch")

    left, right = st.columns((0.45, 0.55), gap="large")
    with left:
        st.subheader("Deferred work")
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
        st.subheader("Resource summary")
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

    with st.expander("Full assignment table"):
        st.dataframe(artifacts.schedule, hide_index=True, width="stretch")
        _download_buttons(artifacts, ("cp_sat_schedule.csv",))


def _policy_analysis(artifacts) -> None:
    _page_header(
        "Policy analysis",
        "Policy results, capacity sensitivity and a bounded truth-free replan.",
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
    _metric_row(
        (
            (
                "CP-SAT cost",
                f"{int(cp_sat['total_synthetic_cost_units']):,}",
                "Synthetic cost units, not currency.",
            ),
            (
                "Simulated failures",
                str(int(cp_sat["unplanned_failures"])),
                "Retrospective evaluation only.",
            ),
            (
                "Due jobs deferred",
                str(int(cp_sat["due_deferrals"])),
                "Base case: two bays and 80% operating demand.",
            ),
            (
                "Solve time",
                f"{cp_sat['solve_time_seconds']:.1f}s",
                f"Solver status: {cp_sat['solver_status']}.",
            ),
        )
    )
    st.caption("All costs are synthetic cost units, not real currency.")

    policy_labels = {
        "reactive": "Reactive",
        "fixed_90": "Fixed 90",
        "predicted_rul_30": "Predicted RUL 30",
        "cp_sat": "CP-SAT",
    }
    policy["Policy"] = policy["policy"].map(policy_labels)
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Total cost by policy")
        cost_chart = px.bar(
            policy,
            x="Policy",
            y="total_synthetic_cost_units",
            text="total_synthetic_cost_units",
        )
        cost_chart.update_traces(marker_color="#2F5D8A")
        cost_chart.update_layout(
            xaxis_title="",
            yaxis_title="Synthetic cost units",
        )
        st.plotly_chart(_plot_style(cost_chart), width="stretch")

    with right:
        st.subheader("Scheduled and deferred work")
        decision_chart_data = policy.melt(
            id_vars="Policy",
            value_vars=["scheduled_maintenance", "due_deferrals"],
            var_name="Decision",
            value_name="Engines",
        )
        decision_chart_data["Decision"] = decision_chart_data["Decision"].map(
            {
                "scheduled_maintenance": "Scheduled",
                "due_deferrals": "Due jobs deferred",
            }
        )
        decision_chart = px.bar(
            decision_chart_data,
            x="Policy",
            y="Engines",
            color="Decision",
            barmode="group",
            color_discrete_map={
                "Scheduled": "#2F5D8A",
                "Due jobs deferred": "#B42318",
            },
        )
        decision_chart.update_layout(xaxis_title="", yaxis_title="Engines")
        st.plotly_chart(_plot_style(decision_chart), width="stretch")

    st.subheader("Capacity sensitivity")
    capacity_order = ["constrained", "base", "expanded"]
    capacity["capacity_scenario"] = pd.Categorical(
        capacity["capacity_scenario"],
        categories=capacity_order,
        ordered=True,
    )
    capacity_plot = capacity.sort_values("capacity_scenario").melt(
        id_vars="capacity_scenario",
        value_vars=[
            "scheduled_maintenance",
            "due_deferrals",
            "unplanned_failures",
        ],
        var_name="Measure",
        value_name="Engines",
    )
    capacity_plot["Measure"] = capacity_plot["Measure"].map(
        {
            "scheduled_maintenance": "Scheduled",
            "due_deferrals": "Due jobs deferred",
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
            "Scheduled": "#2F5D8A",
            "Due jobs deferred": "#B42318",
            "Simulated failures": "#B26A00",
        },
    )
    capacity_chart.update_layout(
        xaxis_title="Capacity scenario",
        yaxis_title="Engines",
    )
    st.plotly_chart(_plot_style(capacity_chart, height=350), width="stretch")

    with st.expander("Comparison tables"):
        st.dataframe(
            policy.drop(columns="Policy"),
            hide_index=True,
            width="stretch",
        )
        st.dataframe(capacity, hide_index=True, width="stretch")
        _download_buttons(
            artifacts,
            ("policy_comparison.csv", "capacity_comparison.csv"),
        )

    st.subheader("Capacity replan")
    st.caption(
        "This calculation uses the verified truth-free scenario in memory and "
        "does not write into the release run."
    )
    with st.form("capacity-what-if", border=True):
        control_left, control_right = st.columns(2)
        bays = control_left.slider("Maintenance bays", 1, 3, 2)
        demand_percent = control_right.slider(
            "Minimum operating demand (%)",
            70,
            90,
            80,
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
            _metric_row(
                (
                    (
                        "Solver status",
                        str(result["solver_status"]),
                        "Truth-free what-if result.",
                    ),
                    (
                        "Scheduled jobs",
                        str(int(result.get("scheduled_maintenance", 0))),
                        "Maintenance jobs placed.",
                    ),
                    (
                        "Due jobs deferred",
                        str(int(result.get("due_deferrals", 0))),
                        "Planning exceptions.",
                    ),
                    (
                        "Capacity shortfall",
                        str(
                            int(
                                result.get(
                                    "operating_capacity_shortfall_cycles",
                                    0,
                                )
                            )
                        ),
                        "Synthetic operating cycles below demand.",
                    ),
                )
            )
            with st.expander("What-if assignment table"):
                st.dataframe(
                    pd.DataFrame(schedule.get("jobs", [])),
                    hide_index=True,
                    width="stretch",
                )


def main() -> None:
    """Render the four-page decision application."""
    st.set_page_config(
        page_title="AeroMaintain",
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

    st.sidebar.title("AeroMaintain")
    st.sidebar.caption("Maintenance planning prototype")
    pages = (
        "Overview",
        "Engine risk",
        "Maintenance plan",
        "Policy analysis",
    )
    page = st.sidebar.radio("View", pages)
    st.sidebar.divider()
    st.sidebar.caption("Verified run")
    st.sidebar.code(artifacts.run_id, language=None)
    st.sidebar.caption("NASA C-MAPSS FD001 · simulated data")

    if page == "Overview":
        _overview(artifacts)
    elif page == "Engine risk":
        _engine_risk(artifacts)
    elif page == "Maintenance plan":
        _maintenance_plan(artifacts)
    else:
        _policy_analysis(artifacts)


if __name__ == "__main__":
    main()
