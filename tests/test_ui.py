"""Headless UI tests (Streamlit AppTest) for the patient and clinician pages.
Run with: pytest tests/"""
import pytest
from streamlit.testing.v1 import AppTest

from app import forms
from src.persistence import db, repository

ASSESSMENT = "from app.views.assessment import assessment_page\nassessment_page()"
WORKSPACE = "from app.views.clinician import render_workspace\nrender_workspace('Dr Test')"
INSIGHTS = "from app.views.insights import insights_page\ninsights_page()"


def _app(script: str) -> AppTest:
    return AppTest.from_string(script, default_timeout=120).run()


def _fill_screening(at, **overrides):
    values = {"HighBP": 1, "HighChol": 1, "Stroke": 0, "HeartDiseaseorAttack": 0, "DiffWalk": 1,
              "CholCheck": 1, "Smoker": 0, "PhysActivity": 0, "Fruits": 0, "Veggies": 0,
              "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0}
    values.update(overrides)
    at.number_input(key="scr_Age").set_value(58)
    at.selectbox(key="scr_Sex").set_value(1)
    at.number_input(key="scr_BMI").set_value(42.0)
    at.selectbox(key="scr_GenHlth").set_value(5)
    for field in forms.SCREENING_YES_NO:
        at.radio(key=f"scr_{field}").set_value(values[field])
    return at


def _submit(at, name="Test Patient"):
    at.text_input(key="patient_name").set_value(name)
    consent = [c for c in at.checkbox if c.key == "consent"]  # only shown when saving
    if consent:
        consent[0].check()
    return at.button(key="btn_run").click().run()


@pytest.fixture(autouse=True)
def _no_database_by_default(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_SessionLocal", None)


@pytest.fixture
def sqlite_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'ui.db').as_posix()}")
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_SessionLocal", None)
    yield
    if db._engine is not None:
        db._engine.dispose()


def _no_exceptions(at):
    assert not at.exception, [e.value for e in at.exception]


def test_health_questions_start_unanswered_and_block_submission(sqlite_db):
    at = _app(ASSESSMENT)
    _no_exceptions(at)
    assert at.radio(key="scr_HighBP").value is None  # no silent "No" default
    at.text_input(key="patient_name").set_value("Test")
    at.checkbox(key="consent").check()
    at.button(key="btn_run").click().run()
    assert "last_result" not in at.session_state
    assert any("Please answer" in w.value for w in at.warning)


def test_saving_requires_a_name_and_consent(sqlite_db):
    at = _fill_screening(_app(ASSESSMENT))
    at.button(key="btn_run").click().run()
    text = " ".join(w.value for w in at.warning)
    assert "Enter a name" in text and "permission" in text
    assert "last_result" not in at.session_state


def test_full_screening_run_shows_plain_language_result(sqlite_db):
    at = _submit(_fill_screening(_app(ASSESSMENT)))
    _no_exceptions(at)
    result = at.session_state["last_result"]
    assert result["report"]["risk_tier"] in ("Medium Risk", "High Risk")
    assert result["saved"] is True
    assert any(h.value == "Your result" for h in at.header)
    assert any("Pushing your estimate up" in m.value for m in at.markdown)
    session = db.get_session()
    try:
        assert repository.get_worklist_stats(session)["assessments"] == 1
        stored = repository.get_assessment(session, 1)
        assert stored["raw_patient_json"]["BMI"] == 42.0
        assert stored["explanation_text"]  # narrative is stored for the clinician
    finally:
        session.close()


def test_declining_to_save_stores_nothing_and_says_no_clinician_was_told(sqlite_db):
    at = _fill_screening(_app(ASSESSMENT))
    at.checkbox(key="save_result").uncheck()
    at = _submit(at, name="")
    _no_exceptions(at)
    result = at.session_state["last_result"]
    assert result["saved"] is False and result["save_reason"] == "declined"
    assert result["patient_name"] == "Anonymous"
    assert any("no clinician has been notified" in w.value for w in at.warning)
    session = db.get_session()
    try:
        assert repository.get_worklist_stats(session)["assessments"] == 0
    finally:
        session.close()


def test_result_survives_toggling_technical_details():
    at = _submit(_fill_screening(_app(ASSESSMENT)))
    at.toggle(key="show_technical").set_value(True).run()
    _no_exceptions(at)
    assert "last_result" in at.session_state
    assert any(h.value == "Your result" for h in at.header)
    assert at.dataframe  # the agent-steps table is now visible


def test_what_if_slider_rescoring_works():
    at = _submit(_fill_screening(_app(ASSESSMENT)))
    slider = [s for s in at.slider if s.key.startswith("wi_")][0]
    slider.set_value(25.0).run()
    _no_exceptions(at)
    assert any("Estimate with these changes" in m.label for m in at.metric)


def test_referral_moves_to_lab_check_with_shared_answers_prefilled():
    at = _submit(_fill_screening(_app(ASSESSMENT)))
    at.button(key="btn_referral").click().run()
    _no_exceptions(at)
    assert at.session_state["pipeline_choice"] == "clinical"
    assert at.number_input(key="clin_age").value == 57      # age 58 -> band 55-59 -> midpoint 57
    assert at.number_input(key="clin_bmi").value == 42.0
    assert at.radio(key="clin_hyp").value == 1
    assert at.number_input(key="clin_hba1c").value is None  # lab values are never prefilled
    assert any("carried over" in i.value for i in at.info)


def test_clinical_run_needs_labs_and_then_works():
    at = _app(ASSESSMENT)
    at.session_state["pipeline_choice"] = "clinical"
    at = at.run()
    at.number_input(key="clin_age").set_value(60)
    at.selectbox(key="clin_gender").set_value("Male")
    at.number_input(key="clin_bmi").set_value(35.0)
    at.selectbox(key="clin_smoking").set_value("current")
    at.radio(key="clin_hyp").set_value(1)
    at.radio(key="clin_heart").set_value(1)
    at.text_input(key="patient_name").set_value("Lab Patient")
    at.button(key="btn_run").click().run()
    assert "last_result" not in at.session_state  # labs still blank
    at.number_input(key="clin_hba1c").set_value(9.0)
    at.number_input(key="clin_glucose").set_value(250)
    at.button(key="btn_run").click().run()
    _no_exceptions(at)
    assert at.session_state["last_result"]["report"]["risk_tier"] == "High Risk"


def _seed(rows):
    session = db.get_session()
    try:
        for name, pipeline, tier, prob, alert in rows:
            report = {
                "agent_outputs": {
                    "Prediction Agent": {"output": {"probability": prob, "risk_tier": tier,
                                                     "predicted_class": int(alert), "threshold_used": 0.5}},
                    "Explainability Agent": {"output": {"output": {"top_factors": [
                        {"feature": "high blood pressure", "shap_value": 0.4, "direction": "raises risk"}]}}},
                },
                "alert": {"alert_triggered": alert},
                "explanation": f"Explanation for {name}.", "recommendation": "Suggestions.",
            }
            repository.record_assessment(session, name, pipeline, {"HighBP": 1, "BMI": 33.0, "Age": 7},
                                          report, meta={"model_name": "LightGBM"})
    finally:
        session.close()


def test_clinician_workspace_review_flow(sqlite_db):
    db.get_session().close()  # create tables
    _seed([("Asha Rao", "screening", "High Risk", 0.9, True),
           ("Asha Rao", "screening", "Medium Risk", 0.5, False),
           ("Bela Shah", "clinical", "High Risk", 0.97, True)])
    at = _app(WORKSPACE)
    _no_exceptions(at)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Awaiting review"] == "2" and metrics["High risk"] == "2"

    case_picker = [s for s in at.selectbox if s.label == "Open a case"][0]
    case_picker.set_value(1).run()
    _no_exceptions(at)
    assert any("Explanation for Asha Rao" in m.value for m in at.markdown)
    assert any(t.value == "Asha Rao" for t in at.subheader)

    at.button(key="btn_review_1").click().run()
    _no_exceptions(at)
    session = db.get_session()
    try:
        detail = repository.get_assessment(session, 1)
        assert detail["reviewed"] is True and detail["reviewed_by"] == "Dr Test"
    finally:
        session.close()
    assert {m.label: m.value for m in at.metric}["Awaiting review"] == "1"


def test_clinician_patient_lookup_shows_history(sqlite_db):
    db.get_session().close()
    _seed([("Asha Rao", "screening", "Medium Risk", 0.5, False),
           ("Asha Rao", "screening", "High Risk", 0.9, True)])
    at = _app(WORKSPACE)
    at.text_input(key="lookup_query").set_value("asha").run()
    picker = [s for s in at.selectbox if s.label == "Open a patient"][0]
    picker.set_value("Asha Rao").run()
    _no_exceptions(at)
    assert any("Asha Rao" == t.value for t in at.subheader)


def test_without_storage_the_page_says_nothing_is_stored():
    at = _fill_screening(_app(ASSESSMENT))
    assert not [c for c in at.checkbox if c.key in ("save_result", "consent")]
    assert any("doesn't store results" in c.value for c in at.caption)
    at = _submit(at)
    _no_exceptions(at)
    result = at.session_state["last_result"]
    assert result["saved"] is False and result["save_reason"] == "not_configured"
    assert any("saving isn't set up on this site" in w.value for w in at.warning)


def test_workspace_without_a_database_explains_itself():
    at = _app(WORKSPACE)
    _no_exceptions(at)
    assert any("No database is configured" in i.value for i in at.info)


@pytest.mark.parametrize("pipeline", ["screening", "clinical"])
def test_insights_page_renders_for_both_checks(pipeline):
    at = AppTest.from_string(INSIGHTS, default_timeout=120)
    at.session_state["pipeline_choice"] = pipeline
    at = at.run()
    _no_exceptions(at)
    assert any(m.label == "Deployed model" for m in at.metric)
