"""ORM models for patient identity and assessment history.

One Patient row per distinct name; one Assessment row per Orchestrator.run()
call (not just high-risk ones), backing the clinician worklist.

Columns added after the first release (explanation/recommendation text,
top factors, review status) are also listed in db.ADDED_COLUMNS so an
already-created table is upgraded in place — create_all() alone never adds
columns to an existing table.
"""
import datetime as dt

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True)
    name = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    assessments = relationship("Assessment", back_populates="patient")


class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    pipeline = Column(String, nullable=False)
    probability = Column(Float, nullable=False)
    risk_tier = Column(String, nullable=False)
    predicted_class = Column(Integer, nullable=False)
    threshold_used = Column(Float, nullable=True)
    model_name = Column(String, nullable=True)
    model_version = Column(String, nullable=True)
    alert_triggered = Column(Boolean, nullable=False, default=False)
    raw_patient_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)

    # Added after v1 — see db.ADDED_COLUMNS.
    explanation_text = Column(Text, nullable=True)
    recommendation_text = Column(Text, nullable=True)
    top_factors_json = Column(JSON, nullable=True)
    reviewed = Column(Boolean, default=False)
    reviewed_by = Column(String, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    review_note = Column(Text, nullable=True)

    patient = relationship("Patient", back_populates="assessments")
