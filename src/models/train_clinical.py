"""Retrains the single production model for the clinical pipeline.

Same scope note as train_screening.py — this regenerates artifacts for the
already-chosen model; the 8-model comparison lives on Kaggle.

Only runs when executed directly:
    python -m src.models.train_clinical
"""
import os
import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.etl.clinical_etl import load_and_clean
from src.features.clinical_features import engineer_features, CONTINUOUS_COLS, CATEGORICAL_COLS
from src.models.common import load_config, build_preprocessor, clean_feature_names, to_named_frame, evaluate, save_json

MODEL_NAME = "XGBoost"


def main():
    cfg = load_config()
    cc = cfg["clinical"]
    rs = cfg["random_state"]
    out_dir = cc["output_dir"]
    model_dir = os.path.join(out_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    df = load_and_clean(cc["raw_data"])
    X = df.drop(columns=[cc["target"]])
    y = df[cc["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=rs, stratify=y
    )
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    preprocessor = build_preprocessor(CONTINUOUS_COLS, CATEGORICAL_COLS)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    feature_names = clean_feature_names(preprocessor.get_feature_names_out())
    X_train_scaled = to_named_frame(X_train_scaled, feature_names)
    X_test_scaled = to_named_frame(X_test_scaled, feature_names)

    model = XGBClassifier(
        random_state=rs, eval_metric="logloss",
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    )
    model.fit(X_train_scaled, y_train)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    metrics = evaluate(y_test, y_prob, cc["threshold"])
    print("Test metrics:", metrics)

    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.pkl"))
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    save_json(feature_names, os.path.join(model_dir, "feature_names.json"))
    save_json({
        "model_name": MODEL_NAME,
        "chosen_threshold": cc["threshold"],
        "test_metrics": metrics,
        "random_state": rs,
        "risk_tiers": cc["risk_tiers"],
    }, os.path.join(model_dir, "metadata.json"))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    agent_df = pd.DataFrame(shap_values, columns=feature_names)
    agent_df.insert(0, "predicted_prob", y_prob)
    agent_df.insert(1, "predicted_class", (y_prob >= cc["threshold"]).astype(int))
    agent_df.insert(2, "true_class", y_test.reset_index(drop=True))
    agent_df["shap_base_value"] = explainer.expected_value
    agent_df.to_csv(os.path.join(out_dir, "shap_values_for_agent.csv"), index=False)

    print(f"Artifacts written to {model_dir}")


if __name__ == "__main__":
    main()
