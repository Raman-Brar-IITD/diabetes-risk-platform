# Privacy & Access Model

This document states plainly what identity and health data this app stores,
how access is controlled, and what a real clinical deployment would still
need beyond what's here. It follows the same "be explicit about scope gaps"
convention as `docs/dataset_card_screening.md` and `docs/dataset_card_clinical.md`.

## What is stored, and how it's collected

When persistence is configured (`DATABASE_URL` set), every assessment run —
not just high-risk ones — is recorded: the patient's **name** (as typed into
the public form), the **complete raw input** (every survey/lab field
submitted), and the model's output (probability, risk tier, model name).

The name and inputs are collected through an **unauthenticated public form**,
with an on-screen notice asking the person entering data to have permission
to assess the patient named. That notice is the only safeguard at entry —
it is not a signed consent form, and there is no access control on who can
submit an assessment for whom.

## Access control

- The self-assessment flow (Assessment / Cohort Insights / About tabs) is
  and remains fully public and unauthenticated.
- The **Clinician Worklist** tab, when enabled, is gated by a username/password
  login (`streamlit-authenticator`) with a signed cookie. This controls who
  can *view* the worklist; it does not control who can *submit* an assessment.
- There is no role separation beyond "has a worklist login or doesn't" — no
  per-clinician record ownership, no audit log of who viewed which patient's
  data, no session-level access logging.

## What this does not provide

This is a portfolio/demo access model, not a compliant clinical one. It
does not include:

- Signed, informed patient consent (only an on-screen notice)
- Role-based access control or per-user audit logging
- Encryption-at-rest guarantees beyond whatever the database provider
  (e.g. Supabase) applies by default
- A Business Associate Agreement (BAA) with the database provider
- IRB or institutional compliance review
- HIPAA compliance of any kind

A real hospital deployment would need all of the above, at minimum, before
handling real patient data — consistent with `docs/dataset_card_clinical.md`
and `docs/dataset_card_screening.md` already stating that neither dataset
here is real hospital EHR data. Treat this app's persistence and worklist
features as a demonstration of the *shape* of a clinical workflow, not a
compliant implementation of one.
