"""Load and clean the lab-based clinical dataset."""
import pandas as pd

REQUIRED_COLS = [
    "gender", "age", "hypertension", "heart_disease", "smoking_history",
    "bmi", "HbA1c_level", "blood_glucose_level", "diabetes",
]


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {before - len(df):,} duplicate rows -> {len(df):,} remain")

    df["hypertension"] = df["hypertension"].astype(int)
    df["heart_disease"] = df["heart_disease"].astype(int)
    df["diabetes"] = df["diabetes"].astype(int)

    # Sanity ranges — clinically implausible values are flagged, not silently dropped.
    checks = {
        "bmi": (10, 80),
        "HbA1c_level": (3, 15),
        "blood_glucose_level": (50, 400),
        "age": (0, 120),
    }
    for col, (lo, hi) in checks.items():
        n_bad = (~df[col].between(lo, hi)).sum()
        if n_bad:
            print(f"Warning: {n_bad} rows have {col} outside [{lo}, {hi}]")

    return df
