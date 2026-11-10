"""Streamlit entrypoint for the verified AeroMaintain RUL application."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from aeromaintain.app.artifacts import (
    ArtifactValidationError,
    load_public_bundle,
    load_verified_run,
)

APP_CSS = """
<style>
    :root {
        --am-ink: #202322;
        --am-muted: #68716d;
        --am-line: #dedfd9;
        --am-soft: #f6f6f3;
        --am-green: #315c4c;
    }
    [data-testid="stAppViewContainer"] { background: #ffffff; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background: var(--am-soft);
        border-right: 1px solid var(--am-line);
        min-width: 260px !important;
        max-width: 260px !important;
    }
    [data-testid="stSidebarContent"] {
        padding-top: 1.5rem;
        width: 260px !important;
    }
    [data-testid="stMetric"] {
        padding: 0.7rem 0 0.9rem;
        border-top: 1px solid var(--am-line);
    }
    [data-testid="stMetricLabel"] { color: var(--am-muted); }
    [data-testid="stMetricValue"] {
        color: var(--am-ink);
        font-variant-numeric: tabular-nums;
    }
    [data-testid="stDataFrame"], [data-testid="stExpander"] {
        border-color: var(--am-line);
        border-radius: 3px;
    }
    [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
    h1, h2, h3 {
        color: var(--am-ink);
        letter-spacing: -0.02em;
    }
    h1 { font-size: 2.15rem !important; }
    h3 { margin-top: 1rem !important; }
</style>
"""

PROJECT_URL = "https://github.com/EfeErim/aeromaintain-ai"
PLOT_COLORS = ["#315C4C", "#748078", "#A67C52", "#786D7D"]
RISK_COLORS = {
    "Critical": "#B24C3D",
    "Elevated": "#C38B3A",
    "Routine": "#748078",
}


def _plot_style(figure, *, height: int = 330):
    figure.update_layout(
        template="plotly_white",
        colorway=PLOT_COLORS,
        font={"family": "Segoe UI, Arial, sans-serif", "color": "#3D4541"},
        height=height,
        margin={"l": 20, "r": 20, "t": 24, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        hoverlabel={"font_family": "Segoe UI, Arial, sans-serif"},
    )
    figure.update_xaxes(gridcolor="#E8E8E3", zeroline=False, linecolor="#CDCEC8")
    figure.update_yaxes(gridcolor="#E8E8E3", zeroline=False, linecolor="#CDCEC8")
    return figure


def _page_header(title: str) -> None:
    st.header(title, anchor=False)
    st.divider()


def _metric_row(items: tuple[tuple[str, str, str], ...]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items, strict=True):
        column.metric(label, value, help=help_text)


def _rate_with_interval(rule: dict, name: str) -> str:
    interval = rule["wilson_95"][name]
    if interval is None:
        return "n/a"
    return f"{rule[name]:.1%} ({interval['low']:.1%} to {interval['high']:.1%})"


def _alert_rule_table(point_rule: dict, risk_rule: dict) -> pd.DataFrame:
    rows = []
    for label, column, rule in (
        ("Point prediction", "prediction", point_rule),
        ("Interval lower bound", "interval_low", risk_rule),
    ):
        rows.append(
            {
                "Rule": f"{label} ({column} <= {rule['alert_threshold']:.0f})",
                "Recall": _rate_with_interval(rule, "recall"),
                "Precision": _rate_with_interval(rule, "precision"),
                "False-alarm rate": _rate_with_interval(rule, "false_alarm_rate"),
                "Missed": rule["false_negative"],
                "False alarms": rule["false_positive"],
            }
        )
    return pd.DataFrame(rows)


def _overview(artifacts) -> None:
    _page_header("Overview")
    metrics = artifacts.metrics
    interval = metrics["nominal_empirical_interval"]
    point_rule = metrics["alert_rules"]["point_prediction"]
    risk_rule = metrics["alert_rules"]["interval_low"]
    dataset = artifacts.model_lock["dataset"]
    _metric_row(
        (
            (
                "Test MAE",
                f"{metrics['mae']:.2f}",
                f"Mean absolute error over official {dataset} test engines.",
            ),
            (
                "Test RMSE",
                f"{metrics['rmse']:.2f}",
                "Root mean squared error after model lock.",
            ),
            (
                "Coverage",
                f"{interval['observed_official_test_coverage']:.0%}",
                "Observed coverage for the nominal 90% empirical interval.",
            ),
            (
                "Interval-low recall",
                f"{risk_rule['recall']:.0%}",
                "Share of truly critical engines flagged by the interface risk rule.",
            ),
        )
    )
    st.info(
        f"{dataset} is a simulated NASA benchmark. Metrics shown here come from "
        "the locked official-test evaluation."
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
        risk_chart.update_traces(
            textposition="outside",
            cliponaxis=False,
            marker_line_width=0,
        )
        st.plotly_chart(_plot_style(risk_chart, height=270), width="stretch")

    with right:
        st.subheader("Evaluation summary")
        summary = pd.DataFrame(
            [
                ("Test engines", int(metrics["engines"])),
                ("NASA score", f"{metrics['nasa_score_motor_normalized']:.2f}"),
                (
                    "RMSE 95% range",
                    "{:.2f} to {:.2f}".format(
                        *metrics["bootstrap"]["intervals_95"]["rmse"]
                    ),
                ),
                ("Mean interval width", f"{interval['mean_width']:.1f}"),
                (
                    "Interval target",
                    f"{interval['nominal_coverage']:.0%}",
                ),
                (
                    "Interval observed",
                    f"{interval['observed_official_test_coverage']:.0%}",
                ),
            ],
            columns=["Measure", "Value"],
        )
        summary["Value"] = summary["Value"].astype(str)
        st.dataframe(summary, hide_index=True, width="stretch")

    st.subheader("Critical-engine alert rules")
    st.dataframe(
        _alert_rule_table(point_rule, risk_rule),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "A truly critical engine has at most "
        f"{risk_rule['critical_threshold']:.0f} remaining cycles. Ranges are "
        "two-sided Wilson 95% intervals over one row per engine."
    )

    with st.expander("Run details"):
        identity = pd.DataFrame(
            [
                ("Run", artifacts.run_id),
                ("Dataset", artifacts.model_lock["dataset"]),
                ("Protocol", artifacts.model_lock["protocol"]["label"]),
                ("Champion model", artifacts.model_lock["champion"]["kind"]),
                ("Seed", artifacts.model_lock["seed"]),
                ("Interval", artifacts.model_lock["calibration"]["label"]),
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
    _page_header("Engine risk")
    ranking = artifacts.risk_ranking.copy()
    counts = ranking["risk_band"].value_counts()
    calibration = artifacts.model_lock["calibration"]
    risk_threshold = float(calibration["risk_threshold"])
    elevated_threshold = float(calibration["elevated_threshold"])
    _metric_row(
        (
            (
                "Engines",
                str(len(ranking)),
                f"Official {artifacts.model_lock['dataset']} test engines.",
            ),
            (
                "Critical",
                str(int(counts.get("critical", 0))),
                f"Interval lower bound at or below {risk_threshold:.0f} cycles.",
            ),
            (
                "Elevated",
                str(int(counts.get("elevated", 0))),
                f"Interval lower bound up to {elevated_threshold:.0f} cycles.",
            ),
            (
                "Routine",
                str(int(counts.get("routine", 0))),
                f"Interval lower bound above {elevated_threshold:.0f} cycles.",
            ),
        )
    )

    risk_view = ranking.rename(
        columns={
            "unit_id": "Engine",
            "cycle": "Observed cycle",
            "prediction": "Point RUL",
            "interval_low": "Interval low",
            "interval_high": "Interval high",
            "risk_band": "Band",
        }
    )
    list_column, detail_column = st.columns((0.53, 0.47), gap="large")
    with list_column:
        st.subheader("Priority list")
        st.dataframe(
            risk_view.loc[:, ["Engine", "Point RUL", "Interval low", "Band"]],
            hide_index=True,
            width="stretch",
            height=355,
            column_config={
                "Engine": st.column_config.NumberColumn(format="%d"),
                "Point RUL": st.column_config.NumberColumn(format="%.1f"),
                "Interval low": st.column_config.NumberColumn(format="%.1f"),
                "Band": st.column_config.TextColumn(),
            },
        )
        st.download_button(
            "Download risk_ranking.csv",
            artifacts.risk_ranking.to_csv(index=False).encode("utf-8"),
            file_name="risk_ranking.csv",
            mime="text/csv",
            width="stretch",
        )

    with detail_column:
        st.subheader("Engine review")
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
        if artifacts.sensor_history.empty:
            st.caption(
                "Sensor history is local-only: NASA's dataset page specifies no "
                "license, so this demo does not redistribute its rows."
            )
            sensors = []
        else:
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
        if sensors:
            history = artifacts.sensor_history.loc[
                artifacts.sensor_history["unit_id"].eq(engine_id)
            ]
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
            sensor_chart.update_layout(xaxis_title="Cycle", yaxis_title="Reading")
            st.plotly_chart(
                _plot_style(sensor_chart, height=260),
                width="stretch",
            )

    importance = pd.DataFrame(artifacts.explanation["global_importance"]).head(12)
    if "coefficient" in importance.columns:
        value_column = "coefficient"
        section_title = "Model coefficient review"
        caption = "Model coefficients describe model behavior, not causal effects."
        axis_title = "Standardized coefficient"
    else:
        value_column = "mean_abs_shap"
        section_title = "Model SHAP review"
        caption = "SHAP values describe model behavior, not causal effects."
        axis_title = "Mean absolute SHAP value"

    with st.expander(section_title):
        st.caption(caption)
        importance_chart = px.bar(
            importance.sort_values(value_column),
            x=value_column,
            y="feature",
            orientation="h",
        )
        importance_chart.update_traces(marker_color="#315C4C")
        importance_chart.update_layout(
            xaxis_title=axis_title,
            yaxis_title="",
        )
        st.plotly_chart(
            _plot_style(importance_chart, height=420),
            width="stretch",
        )


def main() -> None:
    st.set_page_config(
        page_title="AeroMaintain",
        page_icon=":material/flight:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)
    public_bundle = os.environ.get("AEROMAINTAIN_PUBLIC_BUNDLE", "")
    run_id = os.environ.get("AEROMAINTAIN_RUN_ID", "")
    project_root = Path(os.environ.get("AEROMAINTAIN_PROJECT_ROOT", ".")).resolve()
    if not run_id and not public_bundle:
        st.error("AEROMAINTAIN_RUN_ID is required. Launch with aeromaintain app.")
        st.stop()
    try:
        artifacts = (
            load_public_bundle(Path(public_bundle))
            if public_bundle
            else load_verified_run(project_root, run_id)
        )
    except ArtifactValidationError as exc:
        st.error(str(exc))
        st.stop()

    st.sidebar.title("AeroMaintain")
    pages = ("Overview", "Engine risk")
    page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("Run")
    st.sidebar.write(artifacts.run_id)
    st.sidebar.caption(f"{artifacts.model_lock['dataset']} · simulated benchmark")
    st.sidebar.caption(artifacts.model_lock["protocol"]["label"])
    st.sidebar.caption(
        "Project complete. Final results across all protocols: "
        f"[README]({PROJECT_URL}#what-was-shown)"
    )

    if page == "Overview":
        _overview(artifacts)
    else:
        _engine_risk(artifacts)


if __name__ == "__main__":
    main()
