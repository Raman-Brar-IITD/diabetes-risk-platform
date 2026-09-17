"""Load and clean the BRFSS screening dataset."""
import pandas as pd

BINARY_COLS = [
    "Diabetes_binary", "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]
ORDINAL_COLS = ["GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {before - len(df):,} duplicate rows -> {len(df):,} remain")

    df[BINARY_COLS] = df[BINARY_COLS].astype(int)
    df[ORDINAL_COLS] = df[ORDINAL_COLS].astype(int)

    # BMI values above ~80 are implausible for a living respondent; flag, don't drop.
    n_outliers = (df["BMI"] > 80).sum()
    if n_outliers:
        print(f"Warning: {n_outliers} rows have BMI > 80 (likely data entry errors)")

    return df
