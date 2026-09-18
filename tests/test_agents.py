"""Smoke tests for the agent pipeline. Run with: pytest tests/"""
import pytest

from src.agents import agents as A
from src.agents.agents import (
    AlertAgent, apply_guardrails, build_pipeline, label_for,
    BANNED_PATTERNS, DISCLAIMER,
)


@pytest.fixture(autouse=True)
def _clear_llm_cache():
    # _call_llm only caches successful completions, but different tests
    # reuse similar prompt text — clear between tests so one test's
    # monkeypatched provider can't leak a cached result into another.
    A._llm_cache.clear()
    yield
    A._llm_cache.clear()


def test_alert_agent_triggers_on_high_risk():
    agent = AlertAgent()
    result = agent.run("MRN123", {"risk_tier": "High Risk", "probability": 0.9})
    assert result["alert_triggered"] is True
    assert len(agent.alert_log) == 1
    assert agent.alert_log[0]["patient_id"] == "MRN123"


def test_alert_agent_no_alert_on_low_risk():
    agent = AlertAgent()
    result = agent.run("MRN123", {"risk_tier": "Low Risk", "probability": 0.1})
    assert result["alert_triggered"] is False
    assert agent.alert_log == []


def test_alert_agent_respects_custom_tiers():
    agent = AlertAgent(alert_tiers=["Medium Risk", "High Risk"])
    result = agent.run("MRN123", {"risk_tier": "Medium Risk", "probability": 0.5})
    assert result["alert_triggered"] is True


def test_apply_guardrails_appends_disclaimer():
    text = "General guidance about diet and exercise."
    out = apply_guardrails(text)
    assert DISCLAIMER in out


def test_apply_guardrails_does_not_duplicate_disclaimer():
    # Regression check: the old implementation skipped the disclaimer
    # whenever the substring "diagnos" appeared anywhere in the text, even
    # if the actual disclaimer sentence wasn't there — e.g. a narrative
    # that says "further diagnosis may be needed" without ever adding the
    # required "consult a clinician" wording would have gone out with no
    # disclaimer at all.
    text = "Further diagnosis may be needed after these results."
    out = apply_guardrails(text)
    assert DISCLAIMER in out
    # But a narrative that already includes the disclaimer verbatim
    # shouldn't get a second copy appended.
    already_safe = f"Some narrative. {DISCLAIMER}"
    out2 = apply_guardrails(already_safe)
    assert out2.count(DISCLAIMER) == 1


def test_apply_guardrails_skips_disclaimer_when_narrative_already_says_it():
    text = "Your tier is High. This is a statistical estimate, not a diagnosis — please see a clinician."
    out = apply_guardrails(text)
    assert out == text  # no second, near-duplicate disclaimer appended


def test_apply_guardrails_redacts_banned_phrase_in_place():
    text = f"{BANNED_PATTERNS[0]}, based on this score."
    out = apply_guardrails(text)
    assert "[GUARDRAIL]" in out
    assert DISCLAIMER in out
    # The diagnostic-sounding phrase itself must not still be visible in
    # the patient-facing portion of the text — flagging it after the fact
    # isn't enough, since the unsafe sentence would still be sitting right
    # there. (The audit note that follows names the phrase on purpose, for
    # transparency, so only check the text that comes before it.)
    patient_facing = out.split("\n\n[GUARDRAIL]")[0]
    assert BANNED_PATTERNS[0] not in patient_facing.lower()


def test_label_for_exact_match():
    assert label_for("HighBP", {"HighBP": "high blood pressure"}) == "high blood pressure"


def test_label_for_one_hot_suffix_uses_longest_matching_key():
    labels = {"BMI": "BMI", "BMI_Category": "BMI category"}
    # Must prefer the more specific "BMI_Category" key over the shorter
    # "BMI" key, which is also a valid (but wrong) prefix match.
    assert label_for("BMI_Category_Obese", labels) == "BMI category (Obese)"


def test_label_for_falls_back_to_raw_name():
    assert label_for("Some_Unmapped_Feature", {}) == "Some Unmapped Feature"


def test_explainer_agent_falls_back_without_llm(monkeypatch):
    monkeypatch.setattr(A, "_call_llm", lambda prompt: None)
    explainer = A.ExplainerAgent({"HighBP": "high blood pressure"})
    prediction = {
        "risk_tier": "High Risk",
        "probability": 0.8,
        "shap_values": {"HighBP": 0.5, "Age": -0.1},
    }
    result = explainer.run({}, prediction)
    assert "High Risk" in result["narrative"]
    assert "high blood pressure" in result["narrative"]
    assert result["narrative_source"] == "template"


def test_explainer_agent_reports_llm_source_when_available(monkeypatch):
    monkeypatch.setattr(A, "_call_llm", lambda prompt: "An LLM-written explanation.")
    explainer = A.ExplainerAgent({"HighBP": "high blood pressure"})
    prediction = {"risk_tier": "High Risk", "probability": 0.8, "shap_values": {"HighBP": 0.5}}
    result = explainer.run({}, prediction)
    assert result["narrative"] == "An LLM-written explanation."
    assert result["narrative_source"] == "llm"


def test_recommender_agent_general_guidance_when_nothing_retrieved(monkeypatch):
    monkeypatch.setattr(A, "_call_llm", lambda prompt: None)
    retriever = A.SimpleRetriever(A.GUIDELINE_SNIPPETS)
    monkeypatch.setattr(retriever, "retrieve", lambda query, k=2: [])
    recommender = A.RecommenderAgent(retriever, {})
    prediction = {"risk_tier": "Low Risk"}
    explanation = {"top_factors": [("Age", -0.1)]}
    result = recommender.run(prediction, explanation)
    assert "General guidance" in result["narrative"]
    assert result["narrative_source"] == "template"


def test_call_llm_falls_back_to_openai_when_anthropic_fails(monkeypatch):
    # Both providers configured; Anthropic errors out, OpenAI should still
    # be tried rather than giving up immediately — this is new behavior,
    # the old version returned None as soon as the first configured
    # provider's call raised, even with a second key available.
    monkeypatch.setattr(A, "ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.setattr(A, "OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setattr(A, "LLM_MAX_RETRIES", 1)

    def _broken_anthropic(prompt, max_tokens, timeout):
        raise RuntimeError("simulated Anthropic outage")

    def _working_openai(prompt, max_tokens, timeout):
        return "response from openai"

    monkeypatch.setattr(A, "_try_anthropic", _broken_anthropic)
    monkeypatch.setattr(A, "_try_openai", _working_openai)

    assert A._call_llm("unique prompt for fallback test") == "response from openai"


def test_call_llm_returns_none_when_all_providers_fail(monkeypatch):
    monkeypatch.setattr(A, "ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.setattr(A, "OPENAI_API_KEY", "")
    monkeypatch.setattr(A, "LLM_MAX_RETRIES", 1)
    monkeypatch.setattr(A, "_try_anthropic", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))

    assert A._call_llm("unique prompt for all-fail test") is None


def _stub_score_fn(patient):
    return {
        "probability": 0.42,
        "predicted_class": 0,
        "risk_tier": "Medium Risk",
        "threshold_used": 0.5,
        "shap_base_value": 0.0,
        "shap_values": {"BMI": 0.1},
    }


def test_orchestrator_threads_patient_id_through_report(monkeypatch):
    monkeypatch.setattr(A, "_call_llm", lambda prompt: None)
    orchestrator = build_pipeline(_stub_score_fn, {"BMI": "BMI"})
    report = orchestrator.run({}, patient_id="MRN123")
    assert report["patient_id"] == "MRN123"
    assert report["alert"]["patient_id"] == "MRN123"


def test_orchestrator_defaults_patient_id_when_omitted(monkeypatch):
    monkeypatch.setattr(A, "_call_llm", lambda prompt: None)
    orchestrator = build_pipeline(_stub_score_fn, {"BMI": "BMI"})
    report = orchestrator.run({})
    assert report["patient_id"] == "demo_patient"


def test_orchestrator_records_narrative_sources(monkeypatch):
    monkeypatch.setattr(A, "_call_llm", lambda prompt: None)
    orchestrator = build_pipeline(_stub_score_fn, {"BMI": "BMI"})
    report = orchestrator.run({})
    assert report["explanation_source"] == "template"
    assert report["recommendation_source"] == "template"


def test_orchestrator_survives_explainer_failure(monkeypatch):
    # A broken/flaky Explainability Agent (e.g. an LLM provider outage that
    # somehow escapes _call_llm's own handling, or a bad SHAP payload)
    # should degrade to a safe fallback narrative instead of taking the
    # whole assessment down — the prediction itself is still valid and
    # worth showing.
    orchestrator = build_pipeline(_stub_score_fn, {"BMI": "BMI"})

    def _broken_run(patient, prediction, top_n=4):
        raise RuntimeError("simulated explainer crash")

    monkeypatch.setattr(orchestrator.explainer_agent, "run", _broken_run)
    report = orchestrator.run({}, patient_id="MRN123")

    assert report["patient_id"] == "MRN123"
    assert report["risk_tier"] == "Medium Risk"
    assert report["explanation_source"] == "error_fallback"
    assert DISCLAIMER in report["explanation"]
    exp_output = report["agent_outputs"]["Explainability Agent"]["output"]
    assert exp_output["fallback_used"] is True
