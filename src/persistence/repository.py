"""Plain functions over an injected session — no module-level global session,
so tests can pass an in-memory SQLite session and never touch Postgres or
Streamlit. This is the durable replacement for AlertAgent.alert_log: one
Assessment row per Orchestrator.run() call, not just high-risk ones.
"""
from src.persistence.models import Assessment, Patient


def get_or_create_patient(session, name: str) -> Patient:
    name = name.strip()
    patient = (
        session.query(Patient)
        .filter(Patient.name.ilike(name))
        .first()
    )
    if patient is None:
        patient = Patient(name=name)
        session.add(patient)
        session.flush()
    return patient


def record_assessment(session, patient_name: str, pipeline: str, patient_input: dict,
                       report: dict, meta: dict | None = None) -> Assessment:
    patient = get_or_create_patient(session, patient_name)
    prediction = report["agent_outputs"]["Prediction Agent"]["output"]
    meta = meta or {}

    assessment = Assessment(
        patient_id=patient.id,
        pipeline=pipeline,
        probability=prediction["probability"],
        risk_tier=prediction["risk_tier"],
        predicted_class=prediction["predicted_class"],
        threshold_used=prediction.get("threshold_used"),
        model_name=meta.get("model_name"),
        model_version=meta.get("model_version"),
        alert_triggered=report["alert"]["alert_triggered"],
        raw_patient_json=patient_input,
    )
    session.add(assessment)
    session.commit()
    return assessment


def get_worklist(session, pipeline: str | None = None, only_flagged: bool = True,
                  limit: int = 50) -> list[dict]:
    query = session.query(Assessment, Patient).join(Patient, Assessment.patient_id == Patient.id)
    if pipeline:
        query = query.filter(Assessment.pipeline == pipeline)
    if only_flagged:
        query = query.filter(Assessment.alert_triggered.is_(True))
    query = query.order_by(Assessment.created_at.desc()).limit(limit)

    return [
        {
            "patient_name": patient.name,
            "pipeline": assessment.pipeline,
            "probability": assessment.probability,
            "risk_tier": assessment.risk_tier,
            "model_name": assessment.model_name,
            "created_at": assessment.created_at,
        }
        for assessment, patient in query.all()
    ]
