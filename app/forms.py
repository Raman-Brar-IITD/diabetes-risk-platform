"""Patient-facing input forms. Each returns (patient_dict_or_None, missing_labels).

Health questions start unanswered (no silent "No" default) so a skipped
question can't quietly change the estimate; the page asks for anything
left blank instead. Every widget has a key so answers survive reruns,
page switches and the quick-check -> lab-check hand-off.
"""
import streamlit as st

from app import helpers as H
from app import labels as L

# Prefixes of widget keys whose state must survive navigation / pipeline
# switches. Buttons are excluded (their state can't be set via session_state).
PERSISTED_PREFIXES = ("scr_", "clin_", "patient_name", "save_result", "consent",
                      "pipeline_choice", "large_text", "referral_from")

SCREENING_YES_NO = [
    "HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack", "DiffWalk", "CholCheck",
    "Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost",
]
_HEALTH_HISTORY = ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack", "DiffWalk", "CholCheck"]
_LIFESTYLE = ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"]
_ACCESS = ["AnyHealthcare", "NoDocbcCost"]


def keep_form_state_alive():
    """Streamlit drops the state of widgets that aren't rendered in a run (for
    example the other pipeline's form). Re-assigning each value keeps it."""
    for key in list(st.session_state.keys()):
        if key.startswith(PERSISTED_PREFIXES):
            st.session_state[key] = st.session_state[key]


def _yes_no(label: str, key: str, missing: list):
    value = st.radio(label, [0, 1], format_func=lambda v: L.YES_NO[v],
                     index=None, horizontal=True, key=key)
    if value is None:
        missing.append(label)
    return value


def _select(label: str, options, key: str, missing: list, format_func=str):
    value = st.selectbox(label, options, format_func=format_func, index=None,
                         placeholder="Choose one", key=key)
    if value is None:
        missing.append(label)
    return value


def _number(label: str, key: str, missing: list, **kwargs):
    value = st.number_input(label, value=None, key=key, **kwargs)
    if value is None:
        missing.append(label)
    return value


# ------------------------------------------------------------------ BMI helper
def _apply_bmi(prefix: str, bmi_key: str):
    ss = st.session_state
    if ss[f"{prefix}_units"].startswith("Metric"):
        bmi = H.bmi_from_metric(ss[f"{prefix}_h_cm"], ss[f"{prefix}_w_kg"])
    else:
        bmi = H.bmi_from_imperial(ss[f"{prefix}_ft"], ss[f"{prefix}_in"], ss[f"{prefix}_lb"])
    ss[bmi_key] = bmi


def _bmi_helper(prefix: str, bmi_key: str):
    with st.expander("Don't know your BMI? Work it out from height and weight"):
        units = st.radio("Units", ["Metric (cm, kg)", "Imperial (feet/inches, pounds)"],
                         horizontal=True, key=f"{prefix}_units")
        if units.startswith("Metric"):
            height = st.number_input("Height (cm)", 100.0, 250.0, 170.0, 0.5, key=f"{prefix}_h_cm")
            weight = st.number_input("Weight (kg)", 25.0, 300.0, 70.0, 0.5, key=f"{prefix}_w_kg")
            preview = H.bmi_from_metric(height, weight)
        else:
            feet = st.number_input("Height (feet)", 3, 7, 5, 1, key=f"{prefix}_ft")
            inches = st.number_input("Height (extra inches)", 0, 11, 7, 1, key=f"{prefix}_in")
            pounds = st.number_input("Weight (pounds)", 55.0, 660.0, 154.0, 1.0, key=f"{prefix}_lb")
            preview = H.bmi_from_imperial(feet, inches, pounds)
        st.caption(f"Calculated BMI: **{preview}**")
        st.button("Use this BMI", key=f"btn_{prefix}_use_bmi", on_click=_apply_bmi,
                  args=(prefix, bmi_key))


# -------------------------------------------------------------- screening form
def screening_form():
    ss = st.session_state
    ss.setdefault("scr_MentHlth", 0)
    ss.setdefault("scr_PhysHlth", 0)
    ss.setdefault("scr_Education", 5)
    ss.setdefault("scr_Income", 6)
    missing: list[str] = []
    p: dict = {}

    with st.container(border=True):
        st.subheader("Step 1 of 4: About you")
        left, right = st.columns(2)
        with left:
            age = _number("Age (years)", "scr_Age", missing, min_value=18, max_value=120, step=1)
            p["Age"] = H.age_to_band(age) if age is not None else None
            p["Sex"] = _select("Sex", list(L.SEX), "scr_Sex", missing, lambda k: L.SEX[k])
        with right:
            p["BMI"] = _number("BMI", "scr_BMI", missing, min_value=H.BMI_MIN,
                               max_value=H.BMI_MAX, step=0.1, format="%.1f")
            p["GenHlth"] = _select("In general, how is your health?", list(L.GENHLTH),
                                   "scr_GenHlth", missing, lambda k: L.GENHLTH[k])
        _bmi_helper("scr_bmi", "scr_BMI")

    with st.container(border=True):
        st.subheader("Step 2 of 4: Health history")
        left, right = st.columns(2)
        for index, field in enumerate(_HEALTH_HISTORY):
            with (left if index % 2 == 0 else right):
                p[field] = _yes_no(L.SCREENING_FIELD_HELP[field], f"scr_{field}", missing)
        p["MentHlth"] = st.slider("Days of poor mental health in the past 30", 0, 30, key="scr_MentHlth")
        p["PhysHlth"] = st.slider("Days of poor physical health in the past 30", 0, 30, key="scr_PhysHlth")

    with st.container(border=True):
        st.subheader("Step 3 of 4: Lifestyle")
        left, right = st.columns(2)
        for index, field in enumerate(_LIFESTYLE):
            with (left if index % 2 == 0 else right):
                p[field] = _yes_no(L.SCREENING_FIELD_HELP[field], f"scr_{field}", missing)

    with st.container(border=True):
        st.subheader("Step 4 of 4: Access to care and background")
        left, right = st.columns(2)
        for index, field in enumerate(_ACCESS):
            with (left if index % 2 == 0 else right):
                p[field] = _yes_no(L.SCREENING_FIELD_HELP[field], f"scr_{field}", missing)
        p["Education"] = st.selectbox("Highest level of education", list(L.EDUCATION),
                                      format_func=lambda k: L.EDUCATION[k], key="scr_Education")
        p["Income"] = st.selectbox("Household income", list(L.INCOME),
                                   format_func=lambda k: L.INCOME[k], key="scr_Income")
        st.caption("The model uses these because they relate to access to care. If you'd rather "
                   "not say, leave the pre-selected answer — the estimate then assumes a typical one.")

    return (None if missing else p), missing


# --------------------------------------------------------------- clinical form
def clinical_form():
    missing: list[str] = []
    p: dict = {}

    if st.session_state.get("referral_from"):
        st.info("Some answers were carried over from the quick check. Please check them, then "
                "enter your lab results below. The two checks are separate models — the quick "
                "check's result is not used in this one.")

    with st.container(border=True):
        st.subheader("Step 1 of 3: About you")
        left, right = st.columns(2)
        with left:
            p["age"] = _number("Age (years)", "clin_age", missing, min_value=1, max_value=120, step=1)
            p["gender"] = _select("Gender", list(L.GENDER), "clin_gender", missing)
        with right:
            p["bmi"] = _number("BMI", "clin_bmi", missing, min_value=H.BMI_MIN,
                               max_value=H.BMI_MAX, step=0.1, format="%.1f")
            p["smoking_history"] = _select("Smoking history", list(L.SMOKING_HISTORY), "clin_smoking",
                                           missing, lambda k: L.SMOKING_HISTORY[k])
        _bmi_helper("clin_bmi_calc", "clin_bmi")

    with st.container(border=True):
        st.subheader("Step 2 of 3: Diagnosed conditions")
        left, right = st.columns(2)
        with left:
            p["hypertension"] = _yes_no("Diagnosed with high blood pressure?", "clin_hyp", missing)
        with right:
            p["heart_disease"] = _yes_no("Diagnosed with heart disease?", "clin_heart", missing)

    with st.container(border=True):
        st.subheader("Step 3 of 3: Lab results")
        st.caption("Copy these from your latest blood test report.")
        left, right = st.columns(2)
        with left:
            p["HbA1c_level"] = _number("HbA1c (%)", "clin_hba1c", missing, min_value=3.0,
                                       max_value=15.0, step=0.1, format="%.1f")
            st.caption(L.CLINICAL_FIELD_HELP["HbA1c_level"])
        with right:
            p["blood_glucose_level"] = _number("Blood glucose (mg/dL)", "clin_glucose", missing,
                                               min_value=50, max_value=400, step=1)

    return (None if missing else p), missing


def start_lab_confirmation():
    """Button callback: carry the shared answers from the quick check into the
    lab-based form and switch to it. Runs before the next rerun, so it may set
    widget keys that the sidebar / forms are about to create."""
    result = st.session_state.get("last_result")
    if not result or result["pipeline"] != "screening":
        return
    prefill = H.screening_to_clinical_prefill(result["patient"])
    st.session_state.update({
        "pipeline_choice": "clinical",
        "clin_age": prefill["age"],
        "clin_gender": prefill["gender"],
        "clin_bmi": prefill["bmi"],
        "clin_smoking": prefill["smoking_history"],
        "clin_hyp": prefill["hypertension"],
        "clin_heart": prefill["heart_disease"],
        "referral_from": result["patient_name"],
    })
