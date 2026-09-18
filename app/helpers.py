"""Pure helpers for the UI — no Streamlit import, so they're unit-testable."""
from app import labels as L

AGE_BAND_STARTS = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
BAND_MIDPOINTS = {1: 21, 2: 27, 3: 32, 4: 37, 5: 42, 6: 47, 7: 52, 8: 57, 9: 62, 10: 67, 11: 72, 12: 77, 13: 82}

BMI_MIN, BMI_MAX = 10.0, 80.0


def age_to_band(age: int) -> int:
    """BRFSS 13-level age band for a real age (under 18 is clamped to band 1)."""
    band = 1
    for index, start in enumerate(AGE_BAND_STARTS, start=1):
        if int(age) >= start:
            band = index
    return band


def _clamp_bmi(value: float) -> float:
    return round(min(max(value, BMI_MIN), BMI_MAX), 1)


def bmi_from_metric(height_cm: float, weight_kg: float) -> float:
    return _clamp_bmi(weight_kg / ((height_cm / 100) ** 2))


def bmi_from_imperial(feet: float, inches: float, pounds: float) -> float:
    total_inches = feet * 12 + inches
    return _clamp_bmi(703 * pounds / (total_inches ** 2))


def screening_to_clinical_prefill(patient: dict) -> dict:
    """Fields that carry over when someone continues from the quick check to
    the lab-based check. Lab values (HbA1c, glucose) are deliberately NOT
    prefilled — they must come from an actual result. The two models stay
    independent; this only saves retyping shared answers."""
    return {
        "age": BAND_MIDPOINTS[int(patient.get("Age", 7))],
        "gender": "Male" if int(patient.get("Sex", 0)) == 1 else "Female",
        "bmi": float(patient["BMI"]),
        "hypertension": int(patient["HighBP"]),
        "heart_disease": int(patient["HeartDiseaseorAttack"]),
        "smoking_history": "ever" if int(patient.get("Smoker", 0)) == 1 else "never",
    }


TIER_STYLE = {
    "Low Risk": {
        "icon": ":material/check_circle:", "color": "green",
        "summary": "Your answers point to a lower chance of diabetes.",
    },
    "Medium Risk": {
        "icon": ":material/warning:", "color": "orange",
        "summary": "Your answers point to a moderate chance of diabetes.",
    },
    "High Risk": {
        "icon": ":material/error:", "color": "red",
        "summary": "Your answers point to a higher chance of diabetes.",
    },
}

_BINARY_KEYS = {
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "DiffWalk", "hypertension", "heart_disease",
}
_CODED = {"Age": L.AGE_BANDS, "GenHlth": L.GENHLTH, "Sex": L.SEX,
          "Education": L.EDUCATION, "Income": L.INCOME,
          "smoking_history": L.SMOKING_HISTORY}


def input_label(pipeline: str, key: str) -> str:
    help_text = (L.SCREENING_FIELD_HELP if pipeline == "screening" else L.CLINICAL_FIELD_HELP).get(key)
    labels = L.SCREENING_FEATURE_LABELS if pipeline == "screening" else L.CLINICAL_FEATURE_LABELS
    label = labels.get(key, help_text or key)
    return label[:1].upper() + label[1:]  # not .capitalize(): it would turn "BMI" into "Bmi"


def format_input_value(key: str, value) -> str:
    """Human-readable value for a stored raw input (e.g. 1 -> Yes, 7 -> 50-54)."""
    if key in _BINARY_KEYS:
        return L.YES_NO.get(int(value), str(value))
    if key in _CODED:
        try:
            return str(_CODED[key].get(value if key == "smoking_history" else int(value), value))
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def inputs_table(pipeline: str, raw: dict) -> list[dict]:
    return [{"Question": input_label(pipeline, k), "Answer": format_input_value(k, v)}
            for k, v in raw.items()]
