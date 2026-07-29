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


def _download_buttons(artifacts, names: tuple[str, ...]) -> None:
    for name in names:
        st.download_button(
            f"Download {name}",
            artifacts.downloads[name],
            file_name=name,
            mime="text/csv",
            key=f"download-{name}",
        )


def _overview(artifacts) -> None:
    st.header("Overview")
    metrics = artifacts.metrics
    cp_sat = artifacts.policy_comparison.loc[
        artifacts.policy_comparison["policy"].eq("cp_sat")
    ].iloc[0]
    left, middle, right, fourth = st.columns(4)
    left.metric("Official-test RMSE", f"{metrics['rmse']:.2f}")
    middle.metric("Official-test MAE", f"{metrics['mae']:.2f}")
    right.metric("CP-SAT scheduled", int(cp_sat["scheduled_maintenance"]))
    fourth.metric("Due deferrals", int(cp_sat["due_deferrals"]))
    st.info(
        "NASA C-MAPSS FD001 is simulated. All operational, resource, duration, "
        "and cost fields are synthetic. This is not an airworthiness system."
    )
    identity = pd.DataFrame(
        [
            ("Run", artifacts.run_id),
            ("Dataset", artifacts.model_lock["dataset"]),
            ("Champion", artifacts.model_lock["champion"]["kind"]),
            ("Seed", artifacts.model_lock["seed"]),
            ("Interval", artifacts.model_lock["calibration"]["label"]),
            ("Model lock SHA-256", artifacts.run_manifest["model_lock_sha256"]),
        ],
        columns=["Identity", "Value"],
    )
    identity["Value"] = identity["Value"].astype(str)
    st.dataframe(identity, hide_index=True, width="stretch")


def _engine_health(artifacts) -> None:
    st.header("Engine Health")
    st.caption(
        "Risk bands use the nominal empirical interval lower bound. Official true "
        "RUL is intentionally unavailable in this decision view."
    )
    st.dataframe(artifacts.risk_ranking, hide_index=True, width="stretch")
    _download_buttons(artifacts, ("risk_ranking.csv",))
    engine_id = st.selectbox(
        "Engine",
        artifacts.risk_ranking["unit_id"].astype(int).tolist(),
    )
    history = artifacts.sensor_history.loc[
        artifacts.sensor_history["unit_id"].eq(engine_id)
    ]
    sensors = st.multiselect(
        "Sensor history",
        [column for column in history.columns if column.startswith("sensor_")],
        default=["sensor_2", "sensor_7", "sensor_12"],
        max_selections=5,
    )
    if sensors:
        chart_data = history.loc[:, ["cycle", *sensors]].melt(
            id_vars="cycle", var_name="sensor", value_name="value"
        )
        st.plotly_chart(
            px.line(chart_data, x="cycle", y="value", color="sensor"),
            width="stretch",
        )
    importance = pd.DataFrame(artifacts.explanation["global_importance"]).head(15)
    st.subheader("Global model-behavior explanation")
    st.caption("Feature importance describes model behavior, not physical causality.")
    st.plotly_chart(
        px.bar(
            importance.sort_values("coefficient"),
            x="coefficient",
            y="feature",
            orientation="h",
        ),
        width="stretch",
    )


def _schedule(artifacts) -> None:
    st.header("Maintenance Schedule")
    cp_sat = artifacts.policy_comparison.loc[
        artifacts.policy_comparison["policy"].eq("cp_sat")
    ].iloc[0]
    st.warning(
        f"Solver status: {cp_sat['solver_status']}; lexicographic optimality: "
        f"{cp_sat['lexicographic_optimality']}. Negative and unproven results "
        "remain visible."
    )
    st.dataframe(artifacts.schedule, hide_index=True, width="stretch")
    _download_buttons(artifacts, ("cp_sat_schedule.csv",))
    scheduled = artifacts.schedule.loc[
        artifacts.schedule["status"].eq("scheduled")
    ].copy()
    if not scheduled.empty:
        scheduled["start"] = pd.to_datetime(
            scheduled["start_day"], unit="D", origin="2026-01-01"
        )
        scheduled["finish"] = pd.to_datetime(
            scheduled["end_day"], unit="D", origin="2026-01-01"
        )
        scheduled["engine"] = "Engine " + scheduled["engine_id"].astype(str)
        st.plotly_chart(
            px.timeline(
                scheduled,
                x_start="start",
                x_end="finish",
                y="engine",
                color="team_id",
                hover_data=["bay_id"],
            ),
            width="stretch",
        )


def _comparison(artifacts) -> None:
    st.header("Policy Comparison & What-if")
    st.caption("Every cost is a synthetic cost_units value, not real currency.")
    st.subheader("Four-policy comparison")
    st.dataframe(artifacts.policy_comparison, hide_index=True, width="stretch")
    st.subheader("Capacity sensitivity")
    st.dataframe(artifacts.capacity_comparison, hide_index=True, width="stretch")
    _download_buttons(
        artifacts,
        ("policy_comparison.csv", "capacity_comparison.csv"),
    )
    st.subheader("Validated capacity what-if")
    bays = st.slider("Maintenance bays", 1, 3, 2)
    demand_percent = st.slider("Minimum operating demand (%)", 70, 90, 80)
    if st.button("Run truth-free optimization"):
        try:
            schedule, metrics = run_capacity_what_if(
                artifacts,
                bays=bays,
                operating_demand_fraction=demand_percent / 100,
            )
        except ArtifactValidationError as exc:
            st.error(str(exc))
        else:
            st.json(metrics)
            st.dataframe(
                pd.DataFrame(schedule.get("jobs", [])),
                hide_index=True,
                width="stretch",
            )


def main() -> None:
    """Render the four-page decision application."""
    st.set_page_config(page_title="AeroMaintain AI", layout="wide")
    st.title("AeroMaintain AI")
    run_id = os.environ.get("AEROMAINTAIN_RUN_ID", "")
    project_root = Path(os.environ.get("AEROMAINTAIN_PROJECT_ROOT", Path.cwd()))
    try:
        artifacts = load_verified_run(project_root, run_id)
    except ArtifactValidationError as exc:
        st.error(f"Verified run could not be loaded: {exc}")
        st.stop()

    page = st.sidebar.radio(
        "Page",
        (
            "Overview",
            "Engine Health",
            "Maintenance Schedule",
            "Policy Comparison & What-if",
        ),
    )
    st.sidebar.caption(f"Explicit run: {artifacts.run_id}")
    if page == "Overview":
        _overview(artifacts)
    elif page == "Engine Health":
        _engine_health(artifacts)
    elif page == "Maintenance Schedule":
        _schedule(artifacts)
    else:
        _comparison(artifacts)


if __name__ == "__main__":
    main()
