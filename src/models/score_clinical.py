"""Self-contained scoring for one patient, clinical pipeline."""
import os
import json
import numpy as np
import pandas as pd
import joblib
import shap

from src.features.clinical_features import engineer_features
from src.models.common import to_named_frame

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                          "outputs", "clinical", "models")

_preprocessor = joblib.load(os.path.join(_MODEL_DIR, "preprocessor.pkl"))
_model = joblib.load(os.path.join(_MODEL_DIR, "model.pkl"))
with open(os.path.join(_MODEL_DIR, "feature_names.json")) as f:
    FEATURE_NAMES = json.load(f)
with open(os.path.join(_MODEL_DIR, "metadata.json")) as f:
    META = json.load(f)

THRESHOLD = META["chosen_threshold"]
TIERS = META["risk_tiers"]
_explainer = shap.TreeExplainer(_model)


def score_patient(patient: dict) -> dict:
    X = engineer_features(pd.DataFrame([patient]))
    X_scaled = _preprocessor.transform(X)  # column selection by name — safe for 1 row
    # Some native model backends (LightGBM's Windows build in particular)
    # can crash with a low-level memory access violation on an array that
    # isn't C-contiguous float64. Cheap insurance regardless of which model
    # ends up deployed.
    X_scaled = np.ascontiguousarray(X_scaled, dtype=np.float64)
    X_scaled = to_named_frame(X_scaled, FEATURE_NAMES)

    prob = float(_model.predict_proba(X_scaled)[0, 1])
    predicted_class = int(prob >= THRESHOLD)
    tier = "Low Risk" if prob < TIERS["low"] else "Medium Risk" if prob < TIERS["medium"] else "High Risk"

    shap_vals = _explainer.shap_values(X_scaled)[0]
    shap_dict = {feat: float(v) for feat, v in zip(FEATURE_NAMES, shap_vals)}

    return {
        "probability": prob,
        "risk_tier": tier,
        "predicted_class": predicted_class,
        "threshold_used": THRESHOLD,
        "shap_base_value": float(_explainer.expected_value),
        "shap_values": shap_dict,
    }
