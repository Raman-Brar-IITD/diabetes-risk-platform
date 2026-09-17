"""Derived features for the screening pipeline. Single source of truth —
imported by training, scoring, and the notebook alike.

One-hot encoding of BMI_Category is NOT done here — it happens in the
ColumnTransformer (src/models/common.py build_preprocessor), which is fit
once on the full training set and applied consistently after. Doing it
here with pd.get_dummies would re-derive categories from whatever rows are
passed in, which silently breaks on a single-row scoring call (only one
category is ever present, so drop_first drops it entirely).
"""
import pandas as pd

CONTINUOUS_COLS = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education",
                    "Income", "Health_Risk_Score", "Lifestyle_Score"]
CATEGORICAL_COLS = ["BMI_Category"]
PASSTHROUGH_COLS = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
                     "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
                     "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"]
ALL_COLS = CONTINUOUS_COLS + CATEGORICAL_COLS + PASSTHROUGH_COLS


def _bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["Health_Risk_Score"] = (
        X["HighBP"] + X["HighChol"] + X["HeartDiseaseorAttack"]
        + X["Stroke"] + (X["BMI"] >= 30).astype(int)
    )
    X["Lifestyle_Score"] = X["PhysActivity"] + X["Fruits"] + X["Veggies"] - X["Smoker"]
    X["BMI_Category"] = X["BMI"].apply(_bmi_category)
    return X[ALL_COLS]
