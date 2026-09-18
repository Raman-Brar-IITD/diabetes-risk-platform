# Diabetes Risk Platform

Two-tier diabetes risk assessment: a low-friction **screening** model for
self-reported risk factors, and a lab-based **clinical** model for use once
a patient has actual test results. Both feed the same explainability
dashboard and agent pipeline.

| Tier | Data | Question it answers |
|---|---|---|
| Screening | BRFSS 2015 survey (self-report) | Should this person be flagged for testing? |
| Clinical | HbA1c / glucose lab dataset | Given real labs, what's their risk? |

The two models are independent — screening output is not fed into the
clinical model as a feature (see `docs/model_card.md` for why).

## Project layout

```
config/         thresholds, paths, random_state
data/           raw CSVs (not committed — see data/*/raw/README.md)
src/etl/        load + clean each dataset
src/features/   feature engineering, one module per pipeline
src/models/     scoring (lightweight) and single-model retraining scripts
src/agents/     prediction -> explanation -> recommendation -> alert
app/            Streamlit dashboard
notebooks/      full 8-model comparison notebook, built to run on Kaggle
outputs/        trained artifacts (populated after training)
docs/           dataset cards, model card
```

## Why training lives on Kaggle

The full model comparison (8 models x 2 pipelines, threshold tuning, SHAP)
is too heavy for a laptop. `notebooks/kaggle_training.ipynb` is a
self-contained notebook meant to be uploaded to Kaggle, run there, and its
`outputs/` download dropped into this repo's `outputs/` folder.

The `src/models/train_*.py` scripts in this repo are a **lighter, separate
thing**: they just retrain the one already-chosen production model, so you
can regenerate artifacts locally without the full comparison. They don't
run on import — only via `python -m src.models.train_screening`.

## Quickstart

```bash
pip install -r requirements/app.txt

# 1. Place data (see data/screening/raw/README.md and data/clinical/raw/README.md)
# 2. Run the Kaggle notebook, download its outputs/ into this repo's outputs/
#    (or run the lightweight local retrain scripts below for a quick single-model version)
python -m src.models.train_screening
python -m src.models.train_clinical

# 3. Launch the dashboard
streamlit run app/app.py
```

## Deliverables mapping

| Deliverable | Where |
|---|---|
| ETL Pipeline | `src/etl/` |
| Feature Engineering | `src/features/` |
| ML Models Comparison | `notebooks/kaggle_training.ipynb` |
| Explainability Dashboard | `app/`, `src/agents/` |

## Known limitations

See `docs/dataset_card_screening.md` and `docs/dataset_card_clinical.md`.
In short: screening data is self-reported survey data, not lab data; the
clinical dataset is a public research dataset, not real hospital records.
This is a decision-support prototype, not a diagnostic tool.

## Future work

Optional, off-by-default additions on top of the base app:

- **Persistence** (`src/persistence/`): each assessment is recorded to a
  hosted Postgres database (e.g. Supabase) when `DATABASE_URL` is set in
  secrets — replacing the in-memory `AlertAgent.alert_log`, which never
  survived a process restart and was never read back anywhere.
- **Clinician Worklist tab**: a login-gated view of recent high-risk
  assessments, enabled by setting `features.clinician_worklist = true` in
  secrets. See `app/.streamlit/secrets.toml.example` for the required keys
  and `docs/privacy_and_access.md` for what this access model does and
  doesn't cover.
- Not implemented: monitoring/drift detection, model/dataset versioning
  beyond `metadata.json`, RBAC, audit logging of worklist views.
