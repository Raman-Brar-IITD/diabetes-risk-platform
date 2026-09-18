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
app/            Streamlit app: app.py (navigation), views/ (pages), forms.py,
                components.py, services.py, helpers.py (pure, unit-tested)
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

## Pages

- **Check my risk** (patients): a stepped form with plain-language results, the
  reasons behind the estimate, next steps, a downloadable report and
  what-if sliders. A high quick-check result offers to continue to the
  lab-based check with shared answers prefilled (the two models stay
  independent). Patients can untick *Save this result* to store nothing.
- **Clinician workspace** (signed-in clinicians, off by default): KPIs, a
  filterable worklist, a case view with the stored answers and explanation,
  "mark reviewed" with a note, and a patient lookup with a risk trend.
- **Model insights**: how the deployed model behaves on held-out data.
- **Privacy and access**: renders `docs/privacy_and_access.md`.

## Optional, off-by-default features

- **Persistence** (`src/persistence/`): assessments are recorded to a hosted
  Postgres database (e.g. Supabase) when `DATABASE_URL` is set in secrets.
  Missing columns are added to an existing table automatically.
- **Clinician workspace**: enabled by `features.clinician_worklist = true` in
  secrets. See `app/.streamlit/secrets.toml.example` for the required keys and
  `docs/privacy_and_access.md` for what this access model does and doesn't cover.
- **AI-written narratives**: set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) to
  have the explanation and suggestions written by an LLM; otherwise standard
  wording is used. Settings live in the `agents:` block of
  `config/model_config.yaml`.

## Future work

- Monitoring/drift detection, model/dataset versioning beyond `metadata.json`.
- Role-based access control and audit logging of who viewed which record.
- Translated (e.g. Hindi) patient text — needs clinical review of every string.
- A PDF report (the current download is plain text).
