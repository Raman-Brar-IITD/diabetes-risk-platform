"""Smoke tests for the agent pipeline. Run with: pytest tests/"""
from src.agents import agents as A
from src.agents.agents import (
    AlertAgent, apply_guardrails, build_pipeline, BANNED_PATTERNS, DISCLAIMER,
)


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


def test_apply_guardrails_appends_disclaimer():
    text = "General guidance about diet and exercise."
    out = apply_guardrails(text)
    assert DISCLAIMER in out


def test_apply_guardrails_flags_banned_phrase():
    text = f"{BANNED_PATTERNS[0]}, based on this score."
    out = apply_guardrails(text)
    assert "[GUARDRAIL]" in out


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


def test_recommender_agent_general_guidance_when_nothing_retrieved(monkeypatch):
    monkeypatch.setattr(A, "_call_llm", lambda prompt: None)
    retriever = A.SimpleRetriever(A.GUIDELINE_SNIPPETS)
    monkeypatch.setattr(retriever, "retrieve", lambda query, k=2: [])
    recommender = A.RecommenderAgent(retriever, {})
    prediction = {"risk_tier": "Low Risk"}
    explanation = {"top_factors": [("Age", -0.1)]}
    result = recommender.run(prediction, explanation)
    assert "General guidance" in result["narrative"]


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
