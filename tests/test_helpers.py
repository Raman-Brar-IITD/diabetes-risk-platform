"""Tests for the pure UI helpers. Run with: pytest tests/"""
from app import helpers as H
from app import labels as L


def test_age_to_band_edges():
    assert H.age_to_band(18) == 1
    assert H.age_to_band(24) == 1
    assert H.age_to_band(25) == 2
    assert H.age_to_band(52) == 7
    assert H.age_to_band(79) == 12
    assert H.age_to_band(80) == 13
    assert H.age_to_band(104) == 13
    assert H.age_to_band(12) == 1  # under 18 clamps to the first band


def test_band_midpoint_round_trips_to_the_same_band():
    for band, midpoint in H.BAND_MIDPOINTS.items():
        assert H.age_to_band(midpoint) == band


def test_bmi_from_metric_and_imperial_agree():
    metric = H.bmi_from_metric(170, 70)          # 24.2
    imperial = H.bmi_from_imperial(5, 7, 154.3)  # ~ same person
    assert abs(metric - 24.2) < 0.1
    assert abs(metric - imperial) < 0.3


def test_bmi_is_clamped_to_the_range_the_form_accepts():
    assert H.bmi_from_metric(200, 20) == H.BMI_MIN
    assert H.bmi_from_metric(120, 300) == H.BMI_MAX


def test_prefill_carries_shared_answers_but_never_lab_values():
    patient = dict(L.SCREENING_DEFAULT_PATIENT, Age=9, Sex=0, BMI=31.5, HighBP=1,
                   HeartDiseaseorAttack=0, Smoker=1)
    prefill = H.screening_to_clinical_prefill(patient)
    assert prefill == {"age": 62, "gender": "Female", "bmi": 31.5, "hypertension": 1,
                       "heart_disease": 0, "smoking_history": "ever"}
    assert "HbA1c_level" not in prefill and "blood_glucose_level" not in prefill


def test_prefill_values_are_valid_clinical_form_options():
    prefill = H.screening_to_clinical_prefill(L.SCREENING_DEFAULT_PATIENT)
    assert prefill["gender"] in L.GENDER
    assert prefill["smoking_history"] in L.SMOKING_HISTORY


def test_format_input_value_is_human_readable():
    assert H.format_input_value("HighBP", 1) == "Yes"
    assert H.format_input_value("Age", 7) == "50-54"
    assert H.format_input_value("GenHlth", 4) == "Fair"
    assert H.format_input_value("smoking_history", "never") == "Never smoked"
    assert H.format_input_value("BMI", 26.5) == "26.5"


def test_input_label_keeps_acronyms_intact():
    assert H.input_label("screening", "BMI") == "BMI"
    assert H.input_label("clinical", "HbA1c_level").startswith("HbA1c")


def test_every_tier_has_a_non_colour_cue_and_a_sentence():
    for tier in ("Low Risk", "Medium Risk", "High Risk"):
        style = H.TIER_STYLE[tier]
        assert style["icon"].startswith(":material/")
        assert style["summary"].endswith(".")
