"""Model transparency: how the deployed model behaves on held-out data."""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agents.agents import label_for
from app import components as C
from app import services

_EXCLUDED = {"predicted_prob", "predicted_class", "true_class", "shap_base_value"}


def _behaviour_tab(pipeline: str):
    df = services.get_shap_export(pipeline)
    if df is None:
        st.info("No evaluation-set export found yet for this check.")
        return
    st.caption(f"Labeled test-set export, {len(df):,} rows. For evaluation only, not live patients.")

    meta = services.get_metadata(pipeline)
    tiers = meta["risk_tiers"] if meta else {"low": 0.3, "medium": 0.6}

    col1, col2 = st.columns([3, 2])
    with col1:
        hist = go.Figure(go.Histogram(x=df["predicted_prob"], nbinsx=30, marker_color="#4c78a8"))
        hist.add_vline(x=tiers["low"], line_dash="dash", line_color=C.TIER_COLORS["Medium Risk"],
                       annotation_text="low/medium")
        hist.add_vline(x=tiers["medium"], line_dash="dash", line_color=C.TIER_COLORS["High Risk"],
                       annotation_text="medium/high")
        hist.update_layout(title="Predicted probability distribution", height=320,
                           xaxis_title="Predicted probability", yaxis_title="Patients")
        st.plotly_chart(hist, width="stretch")
    with col2:
        labels = ["Low Risk", "Medium Risk", "High Risk"]
        tier_labels = pd.cut(df["predicted_prob"], bins=[-0.01, tiers["low"], tiers["medium"], 1.01],
                             labels=labels)
        counts = tier_labels.value_counts().reindex(labels).fillna(0)
        tier_fig = go.Figure(go.Bar(x=counts.index, y=counts.values,
                                    marker_color=[C.TIER_COLORS[t] for t in counts.index]))
        tier_fig.update_layout(title="Test-set patients per risk level", height=320, yaxis_title="Patients")
        st.plotly_chart(tier_fig, width="stretch")

    st.markdown("**Which factors matter most overall** (mean absolute SHAP value across this test set)")
    feature_labels = services.feature_labels_for(pipeline)
    feature_cols = [c for c in df.columns if c not in _EXCLUDED]
    if feature_cols:
        importance = df[feature_cols].abs().mean().sort_values(ascending=False).head(12)
        fig = go.Figure(go.Bar(
            x=importance.values[::-1],
            y=[label_for(f, feature_labels) for f in importance.index[::-1]],
            orientation="h", marker_color="#4c78a8",
        ))
        fig.update_layout(height=350, xaxis_title="Mean |SHAP value| (log-odds)")
        st.plotly_chart(fig, width="stretch")
        st.caption("The model's overall behaviour across the whole test set. It complements the "
                   "per-person factors shown with each result.")

    if {"predicted_class", "true_class"} <= set(df.columns):
        st.markdown("**Test-set performance** (computed live from this export)")
        tp = int(((df["predicted_class"] == 1) & (df["true_class"] == 1)).sum())
        fp = int(((df["predicted_class"] == 1) & (df["true_class"] == 0)).sum())
        fn = int(((df["predicted_class"] == 0) & (df["true_class"] == 1)).sum())
        tn = int(((df["predicted_class"] == 0) & (df["true_class"] == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = (tp + tn) / len(df) if len(df) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy:.1%}")
        m2.metric("Precision", f"{precision:.1%}")
        m3.metric("Recall", f"{recall:.1%}")
        m4.metric("F1", f"{f1:.1%}")
        st.caption(f"At the deployed threshold: TP {tp:,}, FP {fp:,}, FN {fn:,}, TN {tn:,}. This should "
                   f"match `test_metrics` in `outputs/{pipeline}/models/metadata.json`.")


def _details_tab(pipeline: str):
    meta = services.get_metadata(pipeline)
    if not meta:
        st.info("No metadata.json found yet for this check.")
        return
    m1, m2, m3 = st.columns(3)
    m1.metric("Deployed model", meta.get("model_name", "-"))
    m2.metric("Decision threshold", meta.get("chosen_threshold", "-"))
    tiers = meta.get("risk_tiers", {})
    m3.metric("Risk level cut-offs", f"low<{tiers.get('low', '?')}, med<{tiers.get('medium', '?')}")

    test_metrics = meta.get("test_metrics")
    if test_metrics:
        st.markdown("**Held-out test-set metrics** (from training time)")
        cols = st.columns(len(test_metrics))
        for col, (name, value) in zip(cols, test_metrics.items()):
            col.metric(name.replace("_", " ").upper(), f"{value:.1%}" if value <= 1 else f"{value:.3f}")

    best, deployed = meta.get("overall_best_model"), meta.get("model_name")
    if best and deployed and best != deployed:
        st.caption(f"Note: **{best}** scored best overall in the model comparison, but **{deployed}** is "
                   "what's deployed here, because per-person explanations need a tree-based model. See "
                   "`docs/model_card.md` and `notebooks/kaggle_training.ipynb`.")

    comparison = meta.get("comparison_table")
    if comparison:
        with st.expander("Full model comparison table"):
            frame = pd.DataFrame(comparison).set_index("model")
            st.dataframe(frame.style.format("{:.3f}"), width="stretch")
    with st.expander("Raw metadata.json"):
        st.json(meta)


def insights_page():
    st.title("Model insights")
    pipeline = st.session_state.get("pipeline_choice", "screening")
    st.caption(f"Showing: {services.PIPELINE_NAMES[pipeline]}. Change it with the selector in the sidebar.")
    st.write("This tool provides decision support only. It does not diagnose and does not replace "
             "clinical judgment.")
    behaviour, details = st.tabs(["How the model behaves", "Model details"])
    with behaviour:
        _behaviour_tab(pipeline)
    with details:
        _details_tab(pipeline)


_PRIVACY_DOC = os.path.join(services.ROOT, "docs", "privacy_and_access.md")


def privacy_page():
    st.title("Privacy and access")
    st.write("You can use the quick check without saving anything: untick "
             "**Save this result** at the top of the check.")
    if os.path.exists(_PRIVACY_DOC):
        with open(_PRIVACY_DOC, encoding="utf-8") as f:
            st.markdown(f.read())
