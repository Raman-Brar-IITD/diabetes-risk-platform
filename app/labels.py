"""Human-readable labels and default form values, one block per pipeline."""

# ---------------------------------------------------------------- screening
AGE_BANDS = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80 or older",
}
GENHLTH = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}
EDUCATION = {
    1: "Never attended school / kindergarten only", 2: "Elementary (grades 1-8)",
    3: "Some high school (grades 9-11)", 4: "High school graduate / GED",
    5: "Some college or technical school", 6: "College graduate",
}
INCOME = {
    1: "Less than $10,000", 2: "$10,000-$14,999", 3: "$15,000-$19,999",
    4: "$20,000-$24,999", 5: "$25,000-$34,999", 6: "$35,000-$49,999",
    7: "$50,000-$74,999", 8: "$75,000 or more",
}
SEX = {0: "Female", 1: "Male"}
YES_NO = {0: "No", 1: "Yes"}

SCREENING_FIELD_HELP = {
    "HighBP": "Ever told by a health professional you have high blood pressure.",
    "HighChol": "Ever told by a health professional you have high cholesterol.",
    "CholCheck": "Cholesterol check within the past 5 years.",
    "BMI": "Body Mass Index (kg/m^2).",
    "Smoker": "Smoked at least 100 cigarettes in your entire life.",
    "Stroke": "Ever told you had a stroke.",
    "HeartDiseaseorAttack": "Coronary heart disease or heart attack, ever.",
    "PhysActivity": "Physical activity in the past 30 days, outside of work.",
    "Fruits": "Consume fruit one or more times per day.",
    "Veggies": "Consume vegetables one or more times per day.",
    "HvyAlcoholConsump": "Heavy drinker (14+/week men, 7+/week women).",
    "AnyHealthcare": "Has any kind of healthcare coverage.",
    "NoDocbcCost": "Needed to see a doctor in the past year but couldn't afford it.",
    "GenHlth": "Self-rated general health.",
    "MentHlth": "Days of poor mental health in the past 30 days.",
    "PhysHlth": "Days of poor physical health in the past 30 days.",
    "DiffWalk": "Serious difficulty walking or climbing stairs.",
    "Sex": "Sex as recorded in the survey.",
    "Age": "Age band.",
    "Education": "Highest level of education completed.",
    "Income": "Annual household income band.",
}

SCREENING_RAW_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

SCREENING_DEFAULT_PATIENT = {
    "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 26,
    "Smoker": 0, "Stroke": 0, "HeartDiseaseorAttack": 0,
    "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 2, "MentHlth": 2,
    "PhysHlth": 2, "DiffWalk": 0, "Sex": 1, "Age": 7, "Education": 5, "Income": 6,
}

SCREENING_FEATURE_LABELS = {
    "HighBP": "high blood pressure", "HighChol": "high cholesterol",
    "CholCheck": "cholesterol check history", "BMI": "BMI",
    "Smoker": "smoking history", "Stroke": "stroke history",
    "HeartDiseaseorAttack": "heart disease / attack history",
    "PhysActivity": "physical activity level", "Fruits": "fruit intake",
    "Veggies": "vegetable intake", "HvyAlcoholConsump": "heavy alcohol consumption",
    "AnyHealthcare": "healthcare access", "NoDocbcCost": "cost-related care avoidance",
    "GenHlth": "self-rated general health", "MentHlth": "recent mental health",
    "PhysHlth": "recent physical health", "DiffWalk": "mobility",
    "Sex": "sex", "Age": "age band", "Education": "education level", "Income": "income band",
    "Health_Risk_Score": "combined clinical risk score", "Lifestyle_Score": "lifestyle score",
    "BMI_Category": "BMI category",
}

# ------------------------------------------------------------------ clinical
GENDER = {"Female": "Female", "Male": "Male", "Other": "Other"}
SMOKING_HISTORY = {
    "never": "Never smoked", "former": "Former smoker", "current": "Current smoker",
    "not current": "Smoked before, not currently", "ever": "Has smoked at some point",
    "No Info": "Not recorded",
}

CLINICAL_FIELD_HELP = {
    "gender": "Sex/gender as recorded.",
    "age": "Age in years.",
    "hypertension": "Diagnosed with high blood pressure.",
    "heart_disease": "Diagnosed with heart disease.",
    "smoking_history": "Smoking status.",
    "bmi": "Body Mass Index (kg/m^2).",
    "HbA1c_level": "HbA1c lab result (%). ADA bands: <5.7 normal, 5.7-6.4 prediabetes, >=6.5 diabetes range.",
    "blood_glucose_level": "Blood glucose lab result (mg/dL).",
}

CLINICAL_RAW_FEATURES = [
    "gender", "age", "hypertension", "heart_disease", "smoking_history",
    "bmi", "HbA1c_level", "blood_glucose_level",
]

CLINICAL_DEFAULT_PATIENT = {
    "gender": "Female", "age": 45, "hypertension": 0, "heart_disease": 0,
    "smoking_history": "never", "bmi": 26.0, "HbA1c_level": 5.5, "blood_glucose_level": 100,
}

CLINICAL_FEATURE_LABELS = {
    "gender": "gender", "age": "age", "hypertension": "hypertension",
    "heart_disease": "heart disease", "smoking_history": "smoking history",
    "bmi": "BMI", "HbA1c_level": "HbA1c level", "blood_glucose_level": "blood glucose level",
    "Comorbidity_Score": "combined comorbidity score",
    # One-hot columns actually present in the trained model's feature set
    # (see outputs/clinical/models/feature_names.json) — previously
    # missing here, so any of these landing in a patient's top SHAP
    # factors would have shown up as a raw column name like
    # "HbA1c_Category_Prediabetes" instead of a readable label.
    "HbA1c_Category": "HbA1c band", "Glucose_Category": "glucose band",
}
