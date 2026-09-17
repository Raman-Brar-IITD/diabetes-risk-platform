"""Shared helpers used by both pipelines' train/score scripts."""
import json
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)


def load_config(path: str = "config/model_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_preprocessor(continuous_cols: list, categorical_cols: list) -> ColumnTransformer:
    """Scales continuous columns, one-hot encodes categorical columns, and
    passes the rest through untouched. Fit once on the full training set —
    categories and scaling stats are then fixed, so a single-row transform
    at inference time behaves identically to a full-batch one. Selects
    columns by name, so column order in the input DataFrame doesn't matter.
    """
    return ColumnTransformer(
        [
            ("scale", StandardScaler(), continuous_cols),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore",
                                      sparse_output=False), categorical_cols),
        ],
        remainder="passthrough",
    )


def clean_feature_names(names) -> list:
    """ColumnTransformer.get_feature_names_out() prefixes every name with its
    step name (e.g. 'scale__BMI', 'remainder__HighBP'). Strip that prefix so
    downstream label lookups (app/labels.py FEATURE_LABELS, which match on
    the original column name) still work."""
    return [n.split("__", 1)[1] if "__" in n else n for n in names]


def to_named_frame(X, feature_names: list) -> pd.DataFrame:
    """Wraps a preprocessor's array output as a DataFrame with real column
    names. Matters at both train and inference time — some estimators
    (Explainable Boosting Machine in particular) silently fall back to
    generic 'feature_0001' names when fit on a bare ndarray."""
    return pd.DataFrame(X, columns=feature_names)


def evaluate(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def best_threshold(y_true, y_prob, metric: str = "f1") -> float:
    """Sweep thresholds 0.1-0.9 and return the one maximizing `metric`."""
    best_t, best_score = 0.5, -1
    for t in np.arange(0.10, 0.90, 0.01):
        score = evaluate(y_true, y_prob, t)[metric]
        if score > best_score:
            best_t, best_score = t, score
    return round(float(best_t), 2)


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
