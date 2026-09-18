"""Smoke tests for the persistence layer, against in-memory SQLite —
never Postgres, never Streamlit. Run with: pytest tests/"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.persistence.models import Base
from src.persistence.db import ensure_columns
from src.persistence.repository import (
    get_or_create_patient, record_assessment, get_worklist, get_assessment, mark_reviewed,
    get_patient_history, search_patients, get_worklist_stats,
)


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _fake_report(risk_tier="High Risk", probability=0.9, alert_triggered=True):
    return {
        "agent_outputs": {
            "Prediction Agent": {
                "output": {
                    "probability": probability,
                    "risk_tier": risk_tier,
                    "predicted_class": int(alert_triggered),
                    "threshold_used": 0.5,
                },
            },
        },
        "alert": {"alert_triggered": alert_triggered},
    }


def test_get_or_create_patient_reuses_existing_ref():
    session = _session()
    first = get_or_create_patient(session, "Jane Doe")
    second = get_or_create_patient(session, "Jane Doe")
    assert first.id == second.id


def test_record_assessment_inserts_expected_fields():
    session = _session()
    report = _fake_report()
    meta = {"model_name": "LightGBM"}
    assessment = record_assessment(
        session, "Jane Doe", "screening", {"BMI": 32}, report, meta=meta,
    )
    assert assessment.id is not None
    assert assessment.probability == 0.9
    assert assessment.risk_tier == "High Risk"
    assert assessment.model_name == "LightGBM"
    assert assessment.alert_triggered is True
    assert assessment.raw_patient_json == {"BMI": 32}


def test_get_worklist_filters_to_high_risk_and_orders_by_recency():
    session = _session()
    record_assessment(session, "Low Risk Patient", "screening", {}, _fake_report(
        risk_tier="Low Risk", probability=0.1, alert_triggered=False))
    record_assessment(session, "High Risk Patient", "screening", {}, _fake_report(
        risk_tier="High Risk", probability=0.95, alert_triggered=True))

    rows = get_worklist(session, only_flagged=True)
    assert len(rows) == 1
    assert rows[0]["patient_name"] == "High Risk Patient"


def test_get_worklist_respects_pipeline_filter_and_limit():
    session = _session()
    record_assessment(session, "Screening Patient", "screening", {}, _fake_report())
    record_assessment(session, "Clinical Patient", "clinical", {}, _fake_report())

    rows = get_worklist(session, pipeline="clinical", only_flagged=True)
    assert len(rows) == 1
    assert rows[0]["pipeline"] == "clinical"

    limited = get_worklist(session, only_flagged=True, limit=1)
    assert len(limited) == 1


def _report_with_factors(**kwargs):
    report = _fake_report(**kwargs)
    report["explanation"] = "Because of X."
    report["recommendation"] = "Do Y."
    report["agent_outputs"]["Explainability Agent"] = {"output": {"output": {"top_factors": [
        {"feature": "high blood pressure", "shap_value": 0.5, "direction": "raises risk"}]}}}
    return report


def test_get_or_create_patient_treats_wildcard_characters_literally():
    session = _session()
    john = get_or_create_patient(session, "John")
    j_hn = get_or_create_patient(session, "J_hn")
    assert john.id != j_hn.id


def test_record_assessment_keeps_narratives_and_top_factors():
    session = _session()
    saved = record_assessment(session, "Jane Doe", "screening", {"BMI": 32}, _report_with_factors())
    detail = get_assessment(session, saved.id)
    assert detail["explanation_text"] == "Because of X."
    assert detail["recommendation_text"] == "Do Y."
    assert detail["top_factors"][0]["feature"] == "high blood pressure"
    assert detail["raw_patient_json"] == {"BMI": 32}
    assert detail["reviewed"] is False


def test_get_assessment_returns_none_for_unknown_id():
    assert get_assessment(_session(), 999) is None


def test_mark_reviewed_records_who_and_when():
    session = _session()
    saved = record_assessment(session, "Jane Doe", "screening", {}, _fake_report())
    assert mark_reviewed(session, saved.id, "Dr. Rao", "Called patient") is True
    detail = get_assessment(session, saved.id)
    assert detail["reviewed"] is True
    assert detail["reviewed_by"] == "Dr. Rao"
    assert detail["review_note"] == "Called patient"
    assert detail["reviewed_at"] is not None
    assert mark_reviewed(session, 999, "Dr. Rao") is False


def test_worklist_unreviewed_only_and_tier_and_search_filters():
    session = _session()
    a = record_assessment(session, "Asha Rao", "screening", {}, _fake_report())
    record_assessment(session, "Bela Shah", "screening", {}, _fake_report(
        risk_tier="Medium Risk", probability=0.5, alert_triggered=False))
    mark_reviewed(session, a.id, "Dr. Rao")

    assert [r["patient_name"] for r in get_worklist(session, only_flagged=True, unreviewed_only=True)] == []
    medium = get_worklist(session, only_flagged=False, tiers=["Medium Risk"])
    assert [r["patient_name"] for r in medium] == ["Bela Shah"]
    found = get_worklist(session, only_flagged=False, search="asha")
    assert [r["patient_name"] for r in found] == ["Asha Rao"]
    assert found[0]["reviewed"] is True


def test_patient_history_is_oldest_first_and_case_insensitive():
    session = _session()
    record_assessment(session, "Asha Rao", "screening", {}, _fake_report(probability=0.4, risk_tier="Medium Risk", alert_triggered=False))
    record_assessment(session, "asha rao", "screening", {}, _fake_report(probability=0.9))
    history = get_patient_history(session, "ASHA RAO")
    assert [round(h["probability"], 1) for h in history] == [0.4, 0.9]
    assert get_patient_history(session, "Someone Else") == []


def test_search_patients_reports_count_and_latest_tier():
    session = _session()
    record_assessment(session, "Asha Rao", "screening", {}, _fake_report(risk_tier="Low Risk", probability=0.1, alert_triggered=False))
    record_assessment(session, "Asha Rao", "clinical", {}, _fake_report())
    found = search_patients(session, "asha")
    assert found[0]["assessments"] == 2
    assert found[0]["latest_tier"] == "High Risk"
    assert search_patients(session, "zzz") == []


def test_worklist_stats_counts_alerts_awaiting_review():
    session = _session()
    a = record_assessment(session, "Asha Rao", "screening", {}, _fake_report())
    record_assessment(session, "Bela Shah", "clinical", {}, _fake_report())
    record_assessment(session, "Chitra", "screening", {}, _fake_report(
        risk_tier="Low Risk", probability=0.1, alert_triggered=False))
    mark_reviewed(session, a.id, "Dr. Rao")
    stats = get_worklist_stats(session)
    assert stats == {"assessments": 3, "patients": 3, "high_risk": 2, "awaiting_review": 1}
    assert get_worklist_stats(session, pipeline="clinical")["assessments"] == 1


def test_ensure_columns_upgrades_an_existing_older_table_in_place():
    from sqlalchemy import inspect, text
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE assessments (id INTEGER PRIMARY KEY, patient_id INTEGER, pipeline VARCHAR)"))
    ensure_columns(engine)
    columns = {c["name"] for c in inspect(engine).get_columns("assessments")}
    assert {"explanation_text", "recommendation_text", "top_factors_json",
            "reviewed", "reviewed_by", "reviewed_at", "review_note"} <= columns
    ensure_columns(engine)  # second run must be a harmless no-op
