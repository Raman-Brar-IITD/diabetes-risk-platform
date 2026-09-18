"""Reusable presentational pieces shared across pages."""
import datetime as dt

import plotly.graph_objects as go
import streamlit as st

from src.agents.agents import label_for
from app import helpers as H
from app import services

TIER_COLORS = {"Low Risk": "#2ca02c", "Medium Risk": "#ff7f0e", "High Risk": "#d62728"}

NARRATIVE_NOTES = {
    "llm": "Written by an AI model using only this assessment's own numbers.",
    "template": "Standard wording — no AI writing service is configured.",
    "error_fallback": "Fallback wording — the step that normally writes this hit an error. "
                       "The risk level above is unaffected.",
}

LARGE_TEXT_CSS = "<style>html { font-size: 120%; }</style>"


def top_factors_from_report(report: dict) -> list[dict]:
    explanation = report.get("agent_outputs", {}).get("Explainability Agent", {}).get("output", {}) or {}
    return (explanation.get("output") or {}).get("top_factors") or []


def tier_banner(tier: str, probability: float):
    style = H.TIER_STYLE[tier]
    # Icon + word + colour: colour is never the only signal.
    st.markdown(f"## {style['icon']} :{style['color']}[{tier}]")
    st.write(style["summary"])
    st.caption(f"Model estimate: {probability:.0%}. This is a statistical estimate, not a diagnosis.")


def factor_lists(top_factors: list[dict]):
    raising = [f["feature"] for f in top_factors if f.get("direction") == "raises risk"]
    lowering = [f["feature"] for f in top_factors if f.get("direction") == "lowers risk"]
    if not raising and not lowering:
        return
    left, right = st.columns(2)
    with left:
        st.markdown("**Pushing your estimate up**")
        for name in raising or ["Nothing stood out"]:
            st.markdown(f"- :material/arrow_upward: {name}")
    with right:
        st.markdown("**Pulling your estimate down**")
        for name in lowering or ["Nothing stood out"]:
            st.markdown(f"- :material/arrow_downward: {name}")
    st.caption("These are patterns the model found in survey data. They show what moved this "
               "estimate, not what caused anyone's health.")


def shap_figure(pipeline: str, report: dict):
    labels = services.feature_labels_for(pipeline)
    prediction = report["agent_outputs"]["Prediction Agent"]["output"]
    items = sorted(prediction["shap_values"].items(), key=lambda x: -abs(x[1]))[:8]
    fig = go.Figure(go.Bar(
        x=[v for _, v in items],
        y=[label_for(f, labels) for f, _ in items],
        orientation="h",
        marker_color=["#d62728" if v > 0 else "#2ca02c" for _, v in items],
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Top contributing factors (SHAP, log-odds)",
        xaxis_title="Contribution to risk (log-odds; right = raises risk)",
        yaxis=dict(autorange="reversed"),
        height=350,
    )
    return fig


def agent_trace_rows(report: dict) -> list[dict]:
    rows = []
    for name, entry in report["agent_outputs"].items():
        output = entry["output"]
        note = "fallback used" if output.get("fallback_used") else ""
        rows.append({"Step": name, "Seconds": entry["elapsed_seconds"], "Note": note})
    return rows


def report_as_text(patient_name: str, pipeline: str, report: dict, saved: bool) -> str:
    lines = [
        "DIABETES RISK PLATFORM - ASSESSMENT REPORT",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"Check: {services.PIPELINE_NAMES[pipeline]}",
        f"Name: {patient_name}",
        "",
        f"Risk level: {report['risk_tier']}",
        f"Model estimate: {report['probability']:.0%}",
        f"Flagged for clinician review: {'yes' if report['alert']['alert_triggered'] and saved else 'no'}",
        "",
        "WHAT THIS MEANS", report["explanation"],
        "",
        "SUGGESTIONS", report["recommendation"],
        "",
        "This is a statistical decision-support estimate, not a medical diagnosis.",
        "Please discuss it with a clinician.",
    ]
    return "\n".join(lines)
