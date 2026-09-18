"""Smoke tests against the committed trained artifacts. Run with: pytest tests/"""
from app.labels import SCREENING_DEFAULT_PATIENT, CLINICAL_DEFAULT_PATIENT
from src.models.score_screening import score_patient as score_screening
from src.models.score_clinical import score_patient as score_clinical

KNOWN_TIERS = {"Low Risk", "Medium Risk", "High Risk"}


def test_score_screening_default_patient_valid_shape():
    result = score_screening(SCREENING_DEFAULT_PATIENT)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_tier"] in KNOWN_TIERS
    assert result["shap_values"]


def test_score_clinical_default_patient_valid_shape():
    result = score_clinical(CLINICAL_DEFAULT_PATIENT)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_tier"] in KNOWN_TIERS
    assert result["shap_values"]


def test_score_screening_worse_inputs_increase_probability():
    worse = dict(SCREENING_DEFAULT_PATIENT)
    worse.update({"HighBP": 1, "HighChol": 1, "BMI": 42, "GenHlth": 5})
    baseline = score_screening(SCREENING_DEFAULT_PATIENT)["probability"]
    worsened = score_screening(worse)["probability"]
    assert worsened > baseline


def test_score_clinical_worse_inputs_increase_probability():
    worse = dict(CLINICAL_DEFAULT_PATIENT)
    worse.update({"HbA1c_level": 9.5, "blood_glucose_level": 260, "hypertension": 1})
    baseline = score_clinical(CLINICAL_DEFAULT_PATIENT)["probability"]
    worsened = score_clinical(worse)["probability"]
    assert worsened > baseline
