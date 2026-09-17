# Clinical pipeline outputs

Populated by either:
- `notebooks/kaggle_training.ipynb` (full comparison, run on Kaggle, then download here), or
- `python -m src.models.train_clinical` (quick single-model regeneration, local)

Expected files after training:
```
models/preprocessor.pkl
models/model.pkl
models/feature_names.json
models/metadata.json
shap_values_for_agent.csv
```
