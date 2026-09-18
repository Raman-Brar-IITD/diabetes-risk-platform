"""Patient-facing page: answer the questions, get a plain-language result."""
import datetime as dt
import logging

import pandas as pd
import streamlit as st

from src.agents.agents import llm_configured
from app import components as C
from app import forms
from app import services

logger = logging.getLogger(__name__)

WHAT_IF_NOTE = ("Illustration only. The model reflects patterns in survey data, not cause and "
                "effect, so changing a habit does not guarantee this change.")


def _identity_block():
    storage = services.storage_configured()
    with st.container(border=True):
        st.subheader("Before you start")
        if storage:
            st.session_state.setdefault("save_result", True)
            save = st.checkbox("Save this result so a clinician can review it", key="save_result")
        else:
            save = False
        name = st.text_input("Name", key="patient_name")
        if save:
            consent = st.checkbox(
                "I have permission to enter this person's details and to have them stored "
                "for clinician review.", key="consent")
            st.caption("What is stored: the name, the answers and the result. Only signed-in "
                       "clinicians can see them. Details are on the Privacy page.")
        else:
            consent = True
            if storage:
                st.caption("Nothing will be saved, and the name is optional (it only appears on a "
                           "report you download). Because nothing is saved, no clinician will be "
                           "alerted about this result.")
            else:
                st.caption("This site doesn't store results, and the name is optional (it only "
                           "appears on a report you download).")
    return name, save, consent


def _run_assessment(name: str, save: bool, pipeline: str, patient: dict):
    display_name = name.strip() or "Anonymous"
    try:
        orchestrator = services.build_orchestrator(pipeline)
    except Exception:
        logger.exception("Failed to load the %s model", pipeline)
        st.error("The scoring model for this check isn't available right now. Please try again "
                 "later or contact the clinic.")
        return
    try:
        report = orchestrator.run(patient, patient_id=display_name)
    except Exception:
        logger.exception("Assessment run failed for pipeline=%s", pipeline)
        st.error("Something went wrong while working out this result. Please check the answers "
                 "and try again.")
        return

    saved, reason = (False, "declined" if services.storage_configured() else "not_configured")
    if save:
        saved, reason = services.save_assessment(display_name, pipeline, patient, report)

    now = dt.datetime.now()
    st.session_state["last_result"] = {
        "id": now.isoformat(), "time": now.strftime("%H:%M:%S"), "patient_name": display_name,
        "pipeline": pipeline, "patient": patient, "report": report,
        "saved": saved, "save_reason": reason,
    }
    st.session_state.setdefault("session_history", []).insert(0, {
        "Time": now.strftime("%H:%M:%S"), "Name": display_name,
        "Check": "Lab-based" if pipeline == "clinical" else "Quick",
        "Risk": report["risk_tier"], "Estimate": f"{report['probability']:.0%}",
    })


def _save_status(result: dict):
    alert = result["report"]["alert"]["alert_triggered"]
    if result["saved"]:
        if alert:
            st.warning("This result has been flagged for clinician review.", icon=":material/flag:")
        else:
            st.caption("Saved for clinician review.")
        return
    reason = result["save_reason"]
    if reason == "error":
        st.warning("We couldn't save this result, so no clinician has seen it. The result above is "
                   "still valid.", icon=":material/error:")
    elif alert:
        why = "you chose not to save it" if reason == "declined" else "saving isn't set up on this site"
        st.warning(f"This result looks higher-risk, but {why}, so no clinician has been notified. "
                   "Please contact your clinic.", icon=":material/error:")
    elif reason == "not_configured":
        st.caption("Saving isn't set up on this site, so this result was not stored.")


def _next_steps(result: dict):
    tier = result["report"]["risk_tier"]
    if result["pipeline"] == "screening":
        if tier in ("Medium Risk", "High Risk"):
            st.info("Diabetes is confirmed with a blood test (HbA1c or fasting glucose). If you "
                    "already have recent lab results you can run the lab-based check next. "
                    "Otherwise, ask a clinician about getting tested.", icon=":material/science:")
            st.button("I have lab results: continue to the lab-based check",
                      key="btn_referral", on_click=forms.start_lab_confirmation)
        else:
            st.info("A lower estimate is reassuring but not a guarantee. Keep up regular check-ups.",
                    icon=":material/favorite:")
    elif tier == "High Risk":
        st.info("Please see a clinician soon to confirm this result and plan follow-up.",
                icon=":material/stethoscope:")


def _what_if(result: dict):
    pipeline, base = result["pipeline"], result["patient"]
    variant = dict(base)
    key = f"wi_{result['id']}"
    with st.expander("What if something changed?"):
        st.caption(WHAT_IF_NOTE)
        if pipeline == "screening":
            variant["BMI"] = st.slider("BMI", 10.0, 60.0, min(max(float(base["BMI"]), 10.0), 60.0),
                                       0.5, key=f"{key}_bmi")
            for field, label in [("PhysActivity", "I do physical activity outside of work"),
                                 ("Fruits", "I eat fruit every day"),
                                 ("Veggies", "I eat vegetables every day"),
                                 ("HvyAlcoholConsump", "I drink heavily")]:
                variant[field] = int(st.checkbox(label, value=bool(base[field]), key=f"{key}_{field}"))
        else:
            variant["bmi"] = st.slider("BMI", 10.0, 60.0, min(max(float(base["bmi"]), 10.0), 60.0),
                                       0.5, key=f"{key}_bmi")
            st.caption("Lab values are results, not habits, so they aren't adjustable here.")
        if variant == base:
            return
        try:
            new = services.get_score_fn(pipeline)(variant)
        except Exception:
            logger.exception("What-if scoring failed")
            st.caption("Couldn't work this out just now.")
            return
        before = result["report"]["probability"]
        st.metric(f"Estimate with these changes ({new['risk_tier']})", f"{new['probability']:.0%}",
                  delta=f"{(new['probability'] - before) * 100:+.0f} points", delta_color="inverse")


def _technical_details(result: dict):
    report, pipeline = result["report"], result["pipeline"]
    meta = services.get_metadata(pipeline) or {}
    st.plotly_chart(C.shap_figure(pipeline, report), width="stretch")
    st.caption(f"Model: {meta.get('model_name', 'unknown')} · decision threshold "
               f"{meta.get('chosen_threshold', '?')} · raw probability "
               f"{report['probability']:.4f}")
    st.markdown("**Steps that produced this result**")
    st.dataframe(pd.DataFrame(C.agent_trace_rows(report)), hide_index=True, width="stretch")
    st.caption("Explanation: " + C.NARRATIVE_NOTES.get(report["explanation_source"], "")
               + "  Suggestions: " + C.NARRATIVE_NOTES.get(report["recommendation_source"], ""))


def _render_result(result: dict):
    report, pipeline = result["report"], result["pipeline"]
    st.divider()
    st.header("Your result")
    st.caption(f"{services.PIPELINE_NAMES[pipeline]} · {result['patient_name']} · {result['time']}")
    C.tier_banner(report["risk_tier"], report["probability"])
    _save_status(result)
    C.factor_lists(C.top_factors_from_report(report))

    st.subheader("What this means")
    st.write(report["explanation"])
    st.subheader("Suggestions")
    st.write(report["recommendation"])
    _next_steps(result)

    st.download_button(
        "Download this report (.txt)", on_click="ignore", icon=":material/download:",
        data=C.report_as_text(result["patient_name"], pipeline, report, result["saved"]),
        file_name=f"{result['patient_name'].replace(' ', '_')}_{pipeline}_result.txt",
        mime="text/plain",
    )
    _what_if(result)

    if "show_technical" not in st.session_state:
        st.session_state["show_technical"] = services.clinician_authenticated()
    st.toggle("Show technical details (for clinicians)", key="show_technical")
    if st.session_state["show_technical"]:
        _technical_details(result)


def assessment_page():
    pipeline = st.session_state.get("pipeline_choice", "screening")
    st.title("Check your diabetes risk")
    st.write("Answer the questions below to get an estimate. It takes about three minutes.")
    st.caption("This is decision support, not a diagnosis, and it doesn't replace a clinician.")
    if not llm_configured():
        st.caption("Explanations and suggestions use standard wording.")

    name, save, consent = _identity_block()
    patient, missing = forms.screening_form() if pipeline == "screening" else forms.clinical_form()

    if missing:
        st.caption(f"{len(missing)} question(s) still to answer.")
    if st.button("Get my result", type="primary", key="btn_run"):
        problems = []
        if save and not name.strip():
            problems.append("Enter a name, or untick 'Save this result'.")
        if save and not consent:
            problems.append("Please confirm you have permission to enter these details.")
        if missing:
            problems.append("Please answer: " + "; ".join(missing) + ".")
        if problems:
            st.warning("\n".join(f"- {p}" for p in problems))
        else:
            _run_assessment(name, save, pipeline, patient)

    result = st.session_state.get("last_result")
    if result:
        _render_result(result)

    history = st.session_state.get("session_history", [])
    if history:
        with st.expander(f"Results from this browser session ({len(history)})"):
            st.caption("Kept only in this browser tab. It is not saved anywhere else and disappears "
                       "on reload.")
            st.dataframe(pd.DataFrame(history), hide_index=True, width="stretch")
            if st.button("Clear this list", key="btn_clear_history"):
                st.session_state["session_history"] = []
                st.rerun()
