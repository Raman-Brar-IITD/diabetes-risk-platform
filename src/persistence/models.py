"""ORM models for patient identity and assessment history.

One Patient row per distinct name; one Assessment row per Orchestrator.run()
call (not just high-risk ones), backing the clinician worklist.
"""
import datetime as dt

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, JSON,
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

    patient = relationship("Patient", back_populates="assessments")
