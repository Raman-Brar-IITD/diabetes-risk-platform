"""Derived features for the clinical (lab-based) pipeline.

Category bands use standard ADA cutoffs. One-hot encoding happens in the
ColumnTransformer (src/models/common.py build_preprocessor), not here —
see the note in screening_features.py for why.
"""
import pandas as pd

CONTINUOUS_COLS = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
CATEGORICAL_COLS = ["gender", "smoking_history", "HbA1c_Category", "Glucose_Category"]
PASSTHROUGH_COLS = ["hypertension", "heart_disease", "Comorbidity_Score"]
ALL_COLS = CONTINUOUS_COLS + CATEGORICAL_COLS + PASSTHROUGH_COLS


def _hba1c_category(v):
    if v < 5.7:
        return "Normal"
    if v < 6.5:
        return "Prediabetes"
    return "Diabetes_range"


def _glucose_category(v):
    if v < 100:
        return "Normal"
    if v < 126:
        return "Prediabetes"
    return "Diabetes_range"


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["Comorbidity_Score"] = X["hypertension"] + X["heart_disease"]
    X["HbA1c_Category"] = X["HbA1c_level"].apply(_hba1c_category)
    X["Glucose_Category"] = X["blood_glucose_level"].apply(_glucose_category)
    return X[ALL_COLS]
