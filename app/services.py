"""Shared, cached access to models, metadata and persistence for every page."""
import json
import logging
import os

import pandas as pd
import streamlit as st

from src.agents.agents import build_pipeline
from src.persistence import db, repository
from app import labels as L

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PIPELINE_NAMES = {
    "screening": "Quick check (no lab results needed)",
    "clinical": "Lab-based check (I have HbA1c / glucose results)",
}


def feature_labels_for(pipeline: str) -> dict:
    return L.SCREENING_FEATURE_LABELS if pipeline == "screening" else L.CLINICAL_FEATURE_LABELS


@st.cache_resource(show_spinner=False)
def get_score_fn(pipeline: str):
    if pipeline == "screening":
        from src.models.score_screening import score_patient
    else:
        from src.models.score_clinical import score_patient
    return score_patient


def build_orchestrator(pipeline: str):
    """A fresh Orchestrator per assessment. It keeps per-run state (run_log,
    alert_log), so sharing one instance across every visitor's session would
    let concurrent runs overwrite each other's logs. Building one is cheap;
    the expensive part (the loaded model) is cached in get_score_fn."""
    return build_pipeline(get_score_fn(pipeline), feature_labels_for(pipeline))


@st.cache_data
def get_metadata(pipeline: str):
    path = os.path.join(ROOT, "outputs", pipeline, "models", "metadata.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def get_shap_export(pipeline: str):
    path = os.path.join(ROOT, "outputs", pipeline, "shap_values_for_agent.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def save_assessment(patient_name: str, pipeline: str, patient: dict, report: dict):
    """Returns (saved, reason). reason is None when saved, 'not_configured'
    when no database is set up, or 'error' if the write failed. Never raises:
    a save problem must not hide a completed assessment from the person."""
    try:
        session = db.get_session()
        if session is None:
            return False, "not_configured"
        try:
            repository.record_assessment(session, patient_name, pipeline, patient, report,
                                          meta=get_metadata(pipeline))
        finally:
            session.close()
        return True, None
    except Exception:
        logger.exception("Failed to persist assessment (pipeline=%s)", pipeline)
        return False, "error"


def storage_configured() -> bool:
    return db.is_configured()


def clinician_authenticated() -> bool:
    return st.session_state.get("authentication_status") is True


def worklist_enabled() -> bool:
    # st.secrets raises StreamlitSecretNotFoundError on any access, including
    # .get(), when no secrets.toml exists anywhere — not just an empty dict.
    try:
        return bool(st.secrets.get("features", {}).get("clinician_worklist", False))
    except Exception:
        return False
