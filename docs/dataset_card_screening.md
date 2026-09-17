# Dataset Card: Screening (BRFSS 2015)

**Source:** CDC Behavioral Risk Factor Surveillance System, 2015. Public,
Kaggle mirror `alexteboul/diabetes-health-indicators-dataset`.

**Size:** 253,680 rows, 22 columns, no missing values, ~9.5% exact
duplicate rows (removed in ETL).

**Target:** `Diabetes_binary` — collapses prediabetes and diagnosed
diabetes into one positive class (13.9% positive).

## Strengths
- Large, clean, well-documented, free, no access barriers
- Broad population coverage across US states
- Cheap to collect in practice (phone survey, no lab draw)

## Limitations
- Self-reported, not measured — no lab values at all
- Annual survey, single snapshot per respondent — no longitudinal history
- General public, not a hospital patient population — base rates and
  feature distributions likely differ from any specific clinical setting
- Coarse ordinal bands for age/education/income; BMI self-reported
- Collected in 2015

## Verdict
Good for: a low-cost, broad-population **pre-screening** tool — closer in
spirit to the CDC/ADA Prediabetes Risk Test than to a diagnostic aid.
Not suitable as: a source of clinical/diagnostic-grade predictions.
Recalibration and validation on a real target population is required
before using outputs operationally.
