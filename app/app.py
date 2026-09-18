"""Diabetes Risk Platform — Streamlit entry point.

Pages: Check my risk (patients), Clinician workspace (signed-in clinicians,
enabled by a secrets flag), Model insights, Privacy. Model inference and
agent logic live in src/; page code lives in app/views/.
"""
import os
import sys

import streamlit as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app import forms, services  # noqa: E402
from app.components import LARGE_TEXT_CSS  # noqa: E402
from app.views.assessment import assessment_page  # noqa: E402
from app.views.clinician import clinician_page  # noqa: E402
from app.views.insights import insights_page, privacy_page  # noqa: E402

st.set_page_config(page_title="Diabetes Risk Platform", page_icon=":material/stethoscope:",
                   layout="wide")

CHECK_PAGE_TITLE = "Check my risk"
INSIGHTS_PAGE_TITLE = "Model insights"
# Pages where the check selector applies. Matched by title: the default page's
# url_path is "" (not the value passed in), so url_path can't identify it.
PIPELINE_PAGES = {CHECK_PAGE_TITLE, INSIGHTS_PAGE_TITLE}


def main():
    forms.keep_form_state_alive()
    st.session_state.setdefault("pipeline_choice", "screening")

    pages = [st.Page(assessment_page, title=CHECK_PAGE_TITLE, icon=":material/monitor_heart:",
                     url_path="check", default=True)]
    if services.worklist_enabled():
        pages.append(st.Page(clinician_page, title="Clinician workspace",
                             icon=":material/clinical_notes:", url_path="clinic"))
    pages += [
        st.Page(insights_page, title=INSIGHTS_PAGE_TITLE, icon=":material/insights:", url_path="insights"),
        st.Page(privacy_page, title="Privacy and access", icon=":material/privacy_tip:", url_path="privacy"),
    ]
    current = st.navigation(pages)

    with st.sidebar:
        if current.title in PIPELINE_PAGES:
            st.radio("Which check?", list(services.PIPELINE_NAMES),
                     format_func=services.PIPELINE_NAMES.get, key="pipeline_choice")
        st.toggle("Larger text", key="large_text")
        st.caption("Decision support only. Not a diagnosis.")
    if st.session_state.get("large_text"):
        st.markdown(LARGE_TEXT_CSS, unsafe_allow_html=True)

    current.run()


main()
