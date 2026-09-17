"""Smoke tests — no external data needed. Run with: pytest tests/"""
import pandas as pd

from src.features.screening_features import engineer_features as screening_fe, ALL_COLS as SCREENING_COLS
from src.features.clinical_features import engineer_features as clinical_fe, ALL_COLS as CLINICAL_COLS
from src.models.common import build_preprocessor


def test_screening_features_single_row():
    row = pd.DataFrame([{
        "HighBP": 1, "HighChol": 0, "CholCheck": 1, "BMI": 32, "Smoker": 0,
        "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1,
        "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
        "GenHlth": 2, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 1,
        "Age": 9, "Education": 5, "Income": 7,
    }])
    out = screening_fe(row)
    assert list(out.columns) == SCREENING_COLS
    assert out.loc[0, "Health_Risk_Score"] == 2  # HighBP + BMI>=30
    assert out.loc[0, "BMI_Category"] == "Obese"


def test_screening_preprocessor_consistent_single_vs_batch():
    from src.features.screening_features import CONTINUOUS_COLS, CATEGORICAL_COLS
    rows = pd.DataFrame([
        {"HighBP": 1, "HighChol": 0, "CholCheck": 1, "BMI": 32, "Smoker": 0, "Stroke": 0,
         "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
         "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 2,
         "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 1, "Age": 9, "Education": 5, "Income": 7},
        {"HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 21, "Smoker": 0, "Stroke": 0,
         "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
         "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 1,
         "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0, "Age": 3, "Education": 6, "Income": 8},
    ])
    fe = screening_fe(rows)
    pre = build_preprocessor(CONTINUOUS_COLS, CATEGORICAL_COLS)
    batch_out = pre.fit_transform(fe)
    feature_names = list(pre.get_feature_names_out())
    assert any("BMI_Category_Obese" in c for c in feature_names)

    # Transform just the "Obese" row alone — must match the batch result exactly,
    # proving the single-row case isn't silently losing category info.
    single_out = pre.transform(fe.iloc[[0]])
    obese_idx = feature_names.index([c for c in feature_names if "BMI_Category_Obese" in c][0])
    assert single_out[0, obese_idx] == batch_out[0, obese_idx] == 1


def test_clinical_features_shape():
    rows = pd.DataFrame([
        {"gender": "Female", "age": 50, "hypertension": 1, "heart_disease": 0,
         "smoking_history": "never", "bmi": 27.0, "HbA1c_level": 6.8, "blood_glucose_level": 140},
        {"gender": "Male", "age": 30, "hypertension": 0, "heart_disease": 0,
         "smoking_history": "current", "bmi": 22.0, "HbA1c_level": 5.0, "blood_glucose_level": 90},
    ])
    out = clinical_fe(rows)
    assert list(out.columns) == CLINICAL_COLS
    assert out.loc[0, "Comorbidity_Score"] == 1
    assert out.loc[0, "HbA1c_Category"] == "Diabetes_range"
    assert out.loc[1, "HbA1c_Category"] == "Normal"
