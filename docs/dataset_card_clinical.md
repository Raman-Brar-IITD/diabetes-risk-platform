# Dataset Card: Clinical (lab-based)

**Source:** Kaggle, `iammustafatz/diabetes-prediction-dataset`.

**Size:** 100,000 rows, 9 columns (8 features + binary target).

**Target:** `diabetes` (binary). Label definition (whether prediabetes is
folded into the positive class, as it is in the screening dataset) has not
been independently verified against the screening dataset's definition —
check before comparing the two models' outputs directly.

## Strengths
- Includes real lab values: `HbA1c_level`, `blood_glucose_level` — the
  actual diagnostic tests for diabetes, not self-report proxies
- Large enough (100k rows) for the same model-comparison methodology used
  on the screening dataset
- No access barriers (unlike MIMIC-IV, which needs PhysioNet credentialing)

## Limitations
- Community-curated Kaggle dataset, not a peer-reviewed clinical release —
  original collection methodology is less transparently documented than
  BRFSS or MIMIC
- Cross-sectional, not longitudinal — one snapshot per record
- No hospital context (admissions, procedures, medications) — narrower
  than the "hospital records" framing in the original business brief
- Several near-identical re-uploads exist on Kaggle with slightly
  different extra columns — confirm which exact version/schema is in use

## Verdict
Good for: demonstrating a lab-based, higher-fidelity risk model as the
second tier of a two-stage screening -> confirmation pipeline.
Not suitable as: a stand-in for real hospital EHR data (that would require
MIMIC-IV or an actual hospital data-use agreement).
