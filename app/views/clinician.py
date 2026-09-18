"""Clinician workspace: sign-in, worklist, case review, patient lookup."""
import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.persistence import db, repository
from app import components as C
from app import helpers as H
from app import services

logger = logging.getLogger(__name__)

TIERS = ["High Risk", "Medium Risk", "Low Risk"]
CHECK_LABEL = {"screening": "Quick check", "clinical": "Lab-based check"}
STATUS_OPTIONS = ["Awaiting review", "All", "Reviewed"]
PIPELINE_OPTIONS = {"All checks": None, "Quick check": "screening", "Lab-based check": "clinical"}


def _when(value) -> str:
    return value.strftime("%Y-%m-%d %H:%M UTC") if value else ""


def _rows_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "ID": r["assessment_id"], "Name": r["patient_name"],
        "Check": CHECK_LABEL[r["pipeline"]], "Risk": r["risk_tier"],
        "Estimate": f"{r['probability']:.0%}", "When": _when(r["created_at"]),
        "Reviewed": "Yes" if r["reviewed"] else "No",
    } for r in rows])


def _history_chart(history: list[dict]):
    meta = services.get_metadata("screening") or {}
    tiers = meta.get("risk_tiers", {"low": 0.3, "medium": 0.6})
    fig = go.Figure()
    for pipeline, label in CHECK_LABEL.items():
        points = [h for h in history if h["pipeline"] == pipeline]
        if points:
            fig.add_trace(go.Scatter(
                x=[p["created_at"] for p in points], y=[p["probability"] for p in points],
                mode="lines+markers", name=label,
                hovertemplate="%{x|%Y-%m-%d %H:%M}: %{y:.0%}<extra>" + label + "</extra>",
            ))
    fig.add_hline(y=tiers["low"], line_dash="dash", annotation_text="low/medium")
    fig.add_hline(y=tiers["medium"], line_dash="dash", annotation_text="medium/high")
    fig.update_layout(title="Estimated risk over time", height=320, yaxis_tickformat=".0%",
                      yaxis_range=[0, 1], yaxis_title="Model estimate", xaxis_title=None)
    return fig


def _history_block(session, patient_name: str):
    history = repository.get_patient_history(session, patient_name)
    if len(history) > 1:
        st.plotly_chart(_history_chart(history), width="stretch")
    else:
        st.caption("Only one assessment on record, so there is no trend yet.")
    frame = _rows_frame(list(reversed(history))).drop(columns=["Name"], errors="ignore")
    st.dataframe(frame, hide_index=True, width="stretch")


def _case_detail(session, assessment_id: int, reviewer_default: str):
    detail = repository.get_assessment(session, assessment_id)
    if detail is None:
        st.warning("That case no longer exists.")
        return
    pipeline = detail["pipeline"]

    st.subheader(detail["patient_name"])
    m1, m2 = st.columns(2)
    m1.metric("Risk level", detail["risk_tier"])
    m2.metric("Model estimate", f"{detail['probability']:.0%}")
    # Long text values truncate inside st.metric on narrow screens, so they
    # go in a caption instead.
    st.caption(f"{CHECK_LABEL[pipeline]} · recorded {_when(detail['created_at'])} · "
               f"model: {detail.get('model_name') or 'unknown'}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Answers given**")
        st.dataframe(pd.DataFrame(H.inputs_table(pipeline, detail["raw_patient_json"])),
                     hide_index=True, width="stretch")
    with right:
        st.markdown("**Why the model said this**")
        C.factor_lists(detail["top_factors"])
        if detail["explanation_text"]:
            st.write(detail["explanation_text"])
        else:
            st.caption("No explanation was stored for this older assessment.")
        if detail["recommendation_text"]:
            st.markdown("**Suggestions shown to the patient**")
            st.write(detail["recommendation_text"])

    st.markdown("**Review**")
    if detail["reviewed"]:
        st.success(f"Reviewed by {detail['reviewed_by']} on {_when(detail['reviewed_at'])}."
                   + (f" Note: {detail['review_note']}" if detail["review_note"] else ""),
                   icon=":material/check_circle:")
    else:
        reviewer = st.text_input("Reviewed by", value=reviewer_default, key=f"reviewer_{assessment_id}")
        note = st.text_area("Note (optional)", key=f"note_{assessment_id}")
        if st.button("Mark as reviewed", type="primary", key=f"btn_review_{assessment_id}"):
            if not reviewer.strip():
                st.warning("Enter who is reviewing this case.")
            else:
                repository.mark_reviewed(session, assessment_id, reviewer.strip(), note)
                st.session_state["flash"] = f"Marked {detail['patient_name']}'s case as reviewed."
                st.rerun()

    st.markdown("**This patient over time**")
    _history_block(session, detail["patient_name"])


def _worklist_tab(session, reviewer_default: str):
    c1, c2, c3, c4 = st.columns(4)
    pipeline = PIPELINE_OPTIONS[c1.selectbox("Check", list(PIPELINE_OPTIONS))]
    status = c2.selectbox("Status", STATUS_OPTIONS)
    tiers = c3.multiselect("Risk level", TIERS, default=["High Risk"])
    search = c4.text_input("Search by name")

    rows = repository.get_worklist(
        session, pipeline=pipeline, only_flagged=False, limit=200, tiers=tiers or None,
        unreviewed_only=(status == "Awaiting review"), search=search,
    )
    if status == "Reviewed":
        rows = [r for r in rows if r["reviewed"]]
    if not rows:
        st.info("No assessments match these filters.")
        return

    frame = _rows_frame(rows)
    st.caption(f"{len(rows)} assessment(s), newest first.")
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button("Download this list (.csv)", frame.to_csv(index=False),
                       file_name="worklist.csv", mime="text/csv", on_click="ignore",
                       icon=":material/download:")

    labels = {r["assessment_id"]: f"{r['patient_name']} · {r['risk_tier']} · {r['probability']:.0%} · "
                                  f"{_when(r['created_at'])}" for r in rows}
    chosen = st.selectbox("Open a case", list(labels), format_func=labels.get, index=None,
                          placeholder="Choose a case to review")
    if chosen is not None:
        st.divider()
        _case_detail(session, chosen, reviewer_default)


def _lookup_tab(session):
    query = st.text_input("Patient name", key="lookup_query")
    matches = repository.search_patients(session, query)
    if not matches:
        st.info("No patients found.")
        return
    st.dataframe(pd.DataFrame([{
        "Name": m["patient_name"], "Assessments": m["assessments"],
        "Latest risk": m["latest_tier"], "Last seen": _when(m["last_seen"]),
    } for m in matches]), hide_index=True, width="stretch")
    names = [m["patient_name"] for m in matches]
    chosen = st.selectbox("Open a patient", names, index=None, placeholder="Choose a patient")
    if chosen:
        st.subheader(chosen)
        _history_block(session, chosen)


def render_workspace(reviewer_default: str = ""):
    """The signed-in part of the page. Split out so it can be tested without
    the login widget."""
    flash = st.session_state.pop("flash", None)
    if flash:
        st.toast(flash, icon=":material/check_circle:")

    session = db.get_session()
    if session is None:
        st.info("No database is configured, so there is nothing to show yet.")
        return
    try:
        stats = repository.get_worklist_stats(session)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Assessments", stats["assessments"])
        k2.metric("Patients", stats["patients"])
        k3.metric("High risk", stats["high_risk"])
        k4.metric("Awaiting review", stats["awaiting_review"])

        worklist, lookup = st.tabs(["Worklist", "Patient lookup"])
        with worklist:
            _worklist_tab(session, reviewer_default)
        with lookup:
            _lookup_tab(session)
    finally:
        session.close()


def clinician_page():
    st.title("Clinician workspace")
    try:
        from app.auth import get_authenticator
        authenticator = get_authenticator()
    except Exception:
        logger.exception("Sign-in is not configured")
        st.error("Sign-in isn't configured on this site.")
        return

    authenticator.login()
    status = st.session_state.get("authentication_status")
    if status is False:
        st.error("Username or password is incorrect.")
        return
    if status is not True:
        st.info("Sign in to see the worklist.")
        return

    authenticator.logout(location="sidebar")
    render_workspace(st.session_state.get("name") or "")
