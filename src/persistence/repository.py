"""Plain functions over an injected session — no module-level global session,
so tests can pass an in-memory SQLite session and never touch Postgres or
Streamlit. This is the durable replacement for AlertAgent.alert_log: one
Assessment row per Orchestrator.run() call, not just high-risk ones.
"""
import datetime as dt

from sqlalchemy import func

from src.persistence.models import Assessment, Patient


def _name_key(name: str) -> str:
    return name.strip().lower()


def get_or_create_patient(session, name: str) -> Patient:
    name = name.strip()
    # Exact case-insensitive match. (ilike would treat "_" and "%" in a name
    # as wildcards and could merge two different patients' records.)
    patient = (
        session.query(Patient)
        .filter(func.lower(Patient.name) == _name_key(name))
        .first()
    )
    if patient is None:
        patient = Patient(name=name)
        session.add(patient)
        session.flush()
    return patient


def _top_factors(report: dict) -> list:
    explanation = report.get("agent_outputs", {}).get("Explainability Agent", {}).get("output", {}) or {}
    return (explanation.get("output") or {}).get("top_factors") or []


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
        explanation_text=report.get("explanation"),
        recommendation_text=report.get("recommendation"),
        top_factors_json=_top_factors(report),
        reviewed=False,
    )
    session.add(assessment)
    session.commit()
    return assessment


def _row(assessment: Assessment, patient: Patient) -> dict:
    return {
        "assessment_id": assessment.id,
        "patient_name": patient.name,
        "pipeline": assessment.pipeline,
        "probability": assessment.probability,
        "risk_tier": assessment.risk_tier,
        "model_name": assessment.model_name,
        "created_at": assessment.created_at,
        "alert_triggered": bool(assessment.alert_triggered),
        "reviewed": bool(assessment.reviewed),
    }


def get_worklist(session, pipeline: str | None = None, only_flagged: bool = True,
                  limit: int = 50, *, tiers: list[str] | None = None,
                  unreviewed_only: bool = False, search: str | None = None) -> list[dict]:
    query = session.query(Assessment, Patient).join(Patient, Assessment.patient_id == Patient.id)
    if pipeline:
        query = query.filter(Assessment.pipeline == pipeline)
    if only_flagged:
        query = query.filter(Assessment.alert_triggered.is_(True))
    if tiers:
        query = query.filter(Assessment.risk_tier.in_(tiers))
    if unreviewed_only:
        query = query.filter(Assessment.reviewed.isnot(True))
    if search and search.strip():
        query = query.filter(Patient.name.ilike(f"%{search.strip()}%"))
    query = query.order_by(Assessment.created_at.desc(), Assessment.id.desc()).limit(limit)
    return [_row(a, p) for a, p in query.all()]


def get_assessment(session, assessment_id: int) -> dict | None:
    """Full record for one assessment, including stored inputs and narratives."""
    found = (
        session.query(Assessment, Patient)
        .join(Patient, Assessment.patient_id == Patient.id)
        .filter(Assessment.id == assessment_id)
        .first()
    )
    if found is None:
        return None
    assessment, patient = found
    detail = _row(assessment, patient)
    detail.update({
        "predicted_class": assessment.predicted_class,
        "threshold_used": assessment.threshold_used,
        "raw_patient_json": assessment.raw_patient_json or {},
        "explanation_text": assessment.explanation_text,
        "recommendation_text": assessment.recommendation_text,
        "top_factors": assessment.top_factors_json or [],
        "reviewed_by": assessment.reviewed_by,
        "reviewed_at": assessment.reviewed_at,
        "review_note": assessment.review_note,
    })
    return detail


def mark_reviewed(session, assessment_id: int, reviewer: str, note: str = "") -> bool:
    assessment = session.get(Assessment, assessment_id)
    if assessment is None:
        return False
    assessment.reviewed = True
    assessment.reviewed_by = reviewer
    assessment.reviewed_at = dt.datetime.utcnow()
    assessment.review_note = note.strip() or None
    session.commit()
    return True


def get_patient_history(session, patient_name: str, pipeline: str | None = None) -> list[dict]:
    """Oldest first, so it plots directly as a trend."""
    query = (
        session.query(Assessment, Patient)
        .join(Patient, Assessment.patient_id == Patient.id)
        .filter(func.lower(Patient.name) == _name_key(patient_name))
    )
    if pipeline:
        query = query.filter(Assessment.pipeline == pipeline)
    query = query.order_by(Assessment.created_at.asc(), Assessment.id.asc())
    return [_row(a, p) for a, p in query.all()]


def search_patients(session, query_text: str, limit: int = 20) -> list[dict]:
    text_ = (query_text or "").strip()
    query = session.query(Patient)
    if text_:
        query = query.filter(Patient.name.ilike(f"%{text_}%"))
    results = []
    for patient in query.order_by(Patient.name.asc()).limit(limit).all():
        latest = (
            session.query(Assessment)
            .filter(Assessment.patient_id == patient.id)
            .order_by(Assessment.created_at.desc(), Assessment.id.desc())
            .first()
        )
        count = session.query(func.count(Assessment.id)).filter(Assessment.patient_id == patient.id).scalar()
        results.append({
            "patient_name": patient.name,
            "assessments": count,
            "latest_tier": latest.risk_tier if latest else None,
            "last_seen": latest.created_at if latest else None,
        })
    return results


def get_worklist_stats(session, pipeline: str | None = None) -> dict:
    base = session.query(Assessment)
    if pipeline:
        base = base.filter(Assessment.pipeline == pipeline)
    return {
        "assessments": base.count(),
        "patients": session.query(func.count(func.distinct(Assessment.patient_id)))
                           .filter(*( [Assessment.pipeline == pipeline] if pipeline else [] )).scalar(),
        "high_risk": base.filter(Assessment.risk_tier == "High Risk").count(),
        "awaiting_review": base.filter(Assessment.alert_triggered.is_(True),
                                        Assessment.reviewed.isnot(True)).count(),
    }
