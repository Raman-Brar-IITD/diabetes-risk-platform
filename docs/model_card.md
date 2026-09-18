# Model Card: Diabetes Risk Platform

## Intended use
Decision-support for early identification of diabetes risk, in two tiers:
1. **Screening** — self-reported risk factors, for broad/low-cost triage.
2. **Clinical** — lab-confirmed values, for use once results exist.
Neither model diagnoses. Outputs are a probability, a risk tier, and an
explanation — always paired with a "consult a clinician" disclaimer
(enforced in code by `apply_guardrails` in `src/agents/agents.py`).

## Why the two models are kept separate
The screening model's output is deliberately **not** used as an input
feature to the clinical model. The two datasets cover different, only
partially overlapping populations and feature sets — chaining them would
either require fabricating missing survey fields for clinical-dataset
patients, or reintroduce the same signal twice (e.g. BMI directly, and a
"risk score" derived partly from BMI), which inflates apparent
multicollinearity and muddies SHAP attribution. The two are connected at
the decision layer instead: a high screening risk tier is a trigger to
recommend lab-based confirmation, not a number fed into the next model.

## Training data
See `docs/dataset_card_screening.md` and `docs/dataset_card_clinical.md`.

## Models compared
Logistic Regression, Random Forest, XGBoost, a small neural network,
LightGBM, CatBoost, an Explainable Boosting Machine (glass-box), and a
stacked ensemble (out-of-fold LR+RF+XGBoost -> meta-learner). Comparison
methodology: `notebooks/kaggle_training.ipynb`. Metrics and the chosen
model/threshold per pipeline are written to `outputs/*/models/metadata.json`
at training time — that file is the source of truth, not this document.

## Explainability
SHAP (TreeExplainer) for the deployed tree-based model. The EBM's native
per-feature shape functions are compared against SHAP output in the
notebook as a cross-check of whether the two independently-derived
explanations agree.

## Limitations
- Neither dataset is real hospital EHR data (see dataset cards)
- Probabilities are not calibrated for any specific deployment population
- No monitoring/drift detection is implemented — see README "Future work"
- See `docs/privacy_and_access.md` for what identity/persistence data this
  app stores and what a real clinical deployment would still need
