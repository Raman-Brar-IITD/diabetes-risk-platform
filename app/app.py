"""Diabetes Risk Platform — Streamlit dashboard.

Wraps two independent models (screening / clinical) in the same agent
pipeline and UI. Model inference and agent logic live in src/; this file
is presentation only.
"""
import os
import sys
import json

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.agents.agents import build_pipeline  # noqa: E402
from src.persistence import db, repository  # noqa: E402
from app import labels as L  # noqa: E402

st.set_page_config(page_title="Diabetes Risk Platform", page_icon="🩺", layout="wide")

PIPELINES = {
    "Community Screening (self-reported)": "screening",
    "Clinical Assessment (lab-based)": "clinical",
}


@st.cache_resource
def get_orchestrator(pipeline: str):
    if pipeline == "screening":
        from src.models.score_screening import score_patient
        return build_pipeline(score_patient, L.SCREENING_FEATURE_LABELS)
    from src.models.score_clinical import score_patient
    return build_pipeline(score_patient, L.CLINICAL_FEATURE_LABELS)


@st.cache_data
def get_metadata(pipeline: str):
    path = os.path.join(_ROOT, "outputs", pipeline, "models", "metadata.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def get_shap_export(pipeline: str):
    path = os.path.join(_ROOT, "outputs", pipeline, "shap_values_for_agent.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def screening_form():
    st.caption("Answer as you would on a health survey — no lab results needed.")
    c1, c2, c3 = st.columns(3)
    p = dict(L.SCREENING_DEFAULT_PATIENT)
    with c1:
        p["Age"] = st.selectbox("Age band", list(L.AGE_BANDS), format_func=lambda k: L.AGE_BANDS[k], index=6)
        p["Sex"] = st.selectbox("Sex", list(L.SEX), format_func=lambda k: L.SEX[k])
        p["BMI"] = st.number_input("BMI", 10.0, 80.0, float(p["BMI"]))
        p["GenHlth"] = st.selectbox("General health", list(L.GENHLTH), format_func=lambda k: L.GENHLTH[k], index=1)
    with c2:
        for field in ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack", "Smoker", "DiffWalk"]:
            p[field] = st.radio(L.SCREENING_FIELD_HELP[field], [0, 1], format_func=lambda v: L.YES_NO[v], horizontal=True, key=field)
    with c3:
        for field in ["PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "CholCheck"]:
            p[field] = st.radio(L.SCREENING_FIELD_HELP[field], [0, 1], format_func=lambda v: L.YES_NO[v], horizontal=True, key=field)
    p["MentHlth"] = st.slider("Days of poor mental health (past 30)", 0, 30, p["MentHlth"])
    p["PhysHlth"] = st.slider("Days of poor physical health (past 30)", 0, 30, p["PhysHlth"])
    p["Education"] = st.selectbox("Education", list(L.EDUCATION), format_func=lambda k: L.EDUCATION[k], index=4)
    p["Income"] = st.selectbox("Income", list(L.INCOME), format_func=lambda k: L.INCOME[k], index=5)
    return p


def clinical_form():
    st.caption("Enter actual lab / chart values.")
    c1, c2 = st.columns(2)
    p = dict(L.CLINICAL_DEFAULT_PATIENT)
    with c1:
        p["age"] = st.number_input("Age", 0, 120, int(p["age"]))
        p["gender"] = st.selectbox("Gender", list(L.GENDER))
        p["bmi"] = st.number_input("BMI", 10.0, 80.0, float(p["bmi"]))
        p["smoking_history"] = st.selectbox("Smoking history", list(L.SMOKING_HISTORY), format_func=lambda k: L.SMOKING_HISTORY[k])
    with c2:
        p["hypertension"] = st.radio("Hypertension diagnosed?", [0, 1], format_func=lambda v: L.YES_NO[v], horizontal=True)
        p["heart_disease"] = st.radio("Heart disease diagnosed?", [0, 1], format_func=lambda v: L.YES_NO[v], horizontal=True)
        p["HbA1c_level"] = st.number_input("HbA1c level (%)", 3.0, 15.0, float(p["HbA1c_level"]))
        p["blood_glucose_level"] = st.number_input("Blood glucose (mg/dL)", 50, 400, int(p["blood_glucose_level"]))
    return p


def render_assessment_tab(pipeline: str):
    st.info("Enter the patient's name below. This app stores assessment results "
            "(including the name and inputs you provide) for clinician review. Only "
            "enter information for a patient you have permission to assess — do not "
            "enter another person's details without their consent.")
    patient_name = st.text_input("Patient name")

    patient = screening_form() if pipeline == "screening" else clinical_form()

    if st.button("Run assessment", type="primary"):
        if not patient_name.strip():
            st.warning("Enter the patient's name before running an assessment.")
            return

        try:
            orchestrator = get_orchestrator(pipeline)
        except FileNotFoundError:
            st.error("No trained model found for this pipeline yet. Run the training "
                     "notebook / script and place its outputs in outputs/{}/models/ first.".format(pipeline))
            return

        report = orchestrator.run(patient, patient_id=patient_name.strip())

        try:
            session = db.get_session()
            if session is not None:
                repository.record_assessment(
                    session, patient_name.strip(), pipeline, patient, report,
                    meta=get_metadata(pipeline),
                )
                session.close()
        except Exception:
            pass  # persistence is best-effort; never break the assessment UX

        tier = report["risk_tier"]
        color = {"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red"}[tier]

        st.markdown(f"### Risk tier: :{color}[{tier}]  (probability {report['probability']:.0%})")

        pred = report["agent_outputs"]["Prediction Agent"]["output"]
        shap_items = sorted(pred["shap_values"].items(), key=lambda x: -abs(x[1]))[:8]
        fig = go.Figure(go.Bar(
            x=[v for _, v in shap_items], y=[f for f, _ in shap_items], orientation="h",
            marker_color=["#d62728" if v > 0 else "#2ca02c" for _, v in shap_items],
        ))
        fig.update_layout(title="Top contributing factors (SHAP, log-odds)", height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Explanation**")
        st.write(report["explanation"])
        st.markdown("**Suggestions**")
        st.write(report["recommendation"])
        if report["alert"]["alert_triggered"]:
            st.warning("Escalated to clinician alert queue.")


def render_cohort_tab(pipeline: str):
    df = get_shap_export(pipeline)
    if df is None:
        st.info("No evaluation-set export found yet for this pipeline.")
        return
    st.caption(f"Labeled test-set export, {len(df):,} rows. For evaluation only — not live patients.")
    st.plotly_chart(go.Figure(go.Histogram(x=df["predicted_prob"], nbinsx=30)).update_layout(
        title="Predicted probability distribution", height=300), use_container_width=True)


def render_about_tab(pipeline: str):
    meta = get_metadata(pipeline)
    st.markdown("This tool provides decision support only. It does not diagnose and does "
                "not replace clinical judgment.")
    if meta:
        st.json(meta)
    else:
        st.info("No metadata.json found yet for this pipeline.")


def render_worklist_tab(pipeline: str):
    from app.auth import get_authenticator

    authenticator = get_authenticator()
    authenticator.login()

    if st.session_state.get("authentication_status") is False:
        st.error("Username/password is incorrect.")
        return
    if st.session_state.get("authentication_status") is not True:
        st.info("Log in to view the clinician worklist.")
        return

    authenticator.logout()
    session = db.get_session()
    if session is None:
        st.info("No database configured — nothing to show yet.")
        return

    rows = repository.get_worklist(session, pipeline=pipeline, only_flagged=True)
    session.close()
    if not rows:
        st.info("No high-risk assessments recorded yet for this pipeline.")
        return
    st.caption(f"{len(rows)} high-risk assessment(s), most recent first.")
    st.dataframe(rows, use_container_width=True)


def _worklist_enabled():
    # st.secrets raises StreamlitSecretNotFoundError on any access, including
    # .get(), when no secrets.toml exists anywhere — not just an empty dict.
    try:
        return bool(st.secrets.get("features", {}).get("clinician_worklist", False))
    except Exception:
        return False


def main():
    st.title("🩺 Diabetes Risk Platform")
    pipeline_label = st.sidebar.radio("Pipeline", list(PIPELINES))
    pipeline = PIPELINES[pipeline_label]

    worklist_enabled = _worklist_enabled()

    tab_names = ["Assessment", "Cohort Insights", "About"]
    if worklist_enabled:
        tab_names.append("Clinician Worklist")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        render_assessment_tab(pipeline)
    with tabs[1]:
        render_cohort_tab(pipeline)
    with tabs[2]:
        render_about_tab(pipeline)
    if worklist_enabled:
        with tabs[3]:
            render_worklist_tab(pipeline)


if __name__ == "__main__":
    main()
