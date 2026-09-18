"""Smoke tests for the persistence layer, against in-memory SQLite —
never Postgres, never Streamlit. Run with: pytest tests/"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.persistence.models import Base
from src.persistence.repository import get_or_create_patient, record_assessment, get_worklist


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
