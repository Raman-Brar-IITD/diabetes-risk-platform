# Privacy & Access Model

This document states plainly what identity and health data this app stores,
how access is controlled, and what a real clinical deployment would still
need beyond what's here. It follows the same "be explicit about scope gaps"
convention as `docs/dataset_card_screening.md` and `docs/dataset_card_clinical.md`.

## What is stored, and how it's collected

When persistence is configured (`DATABASE_URL` set) and the person leaves
**Save this result** ticked, every assessment run — not just high-risk ones —
is recorded: the patient's **name** (as typed into the public form), the
**complete raw input** (every survey/lab field submitted), the model's output
(probability, risk tier, model name), the **explanation and suggestions** that
were shown, and, later, a clinician's **review status and note**.

The name and inputs are collected through an **unauthenticated public form**.
The person must tick a box confirming they have permission to enter these
details. That box is the only safeguard at entry — it is not a signed consent
form, and there is no access control on who can submit an assessment for whom.

**Opting out.** Unticking *Save this result* stores nothing (the name is then
optional and only appears on a report the person downloads). The result page
says plainly that, because nothing was saved, no clinician has been notified.

**Who can see stored data.** Only signed-in clinicians, in the Clinician
workspace: the worklist, each case's answers and explanation, and a
patient-by-name history with a risk trend. The public pages never show anyone
else's records; the "results from this browser session" list is held only in
that visitor's own session and is not saved.

**Optional AI-written narratives.** If an `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY` is configured, the explanation and suggestion text is written
by that provider. The prompt contains only the risk level, the probability and
the names of the top contributing factors — never the patient's name or raw
answers — but it does leave this system. Without a key, standard template
wording is used.

## Access control

- The self-assessment flow (Assessment / Cohort Insights / About tabs) is
  and remains fully public and unauthenticated.
- The **Clinician workspace** page, when enabled, is gated by a username/password
  login (`streamlit-authenticator`) with a signed cookie. This controls who
  can *view* stored records; it does not control who can *submit* an assessment.
- Reviewing a case records the reviewer's name and an optional note, but that
  name is typed by the reviewer (defaulting to the login name) — it is a
  convenience, not a tamper-proof audit trail.
- There is no role separation beyond "has a login or doesn't" — no
  per-clinician record ownership, no audit log of who *viewed* which patient's
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
