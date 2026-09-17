"""Prediction -> Explanation -> Recommendation -> Alert agent pipeline.

Works with either pipeline: pass in a scoring function and a feature-label
dict via build_pipeline(). Guardrails screen every patient-facing string
for diagnostic-sounding language and append the required disclaimer.
"""
import os
import time
import datetime as dt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def _call_llm(prompt: str):
    """Returns an LLM string, or None if no key is set / the call fails."""
    try:
        if ANTHROPIC_API_KEY:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            msg = client.messages.create(
                model="claude-sonnet-4-6", max_tokens=250,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        if OPENAI_API_KEY:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini", max_tokens=250,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
    except Exception as e:
        print(f"LLM call failed, falling back to template: {e}")
    return None


class PredictionAgent:
    name = "Prediction Agent"

    def __init__(self, score_fn):
        self.score_fn = score_fn

    def run(self, patient: dict) -> dict:
        result = self.score_fn(patient)
        return {
            "agent": self.name,
            "output": {
                "probability": result["probability"],
                "predicted_class": result["predicted_class"],
                "risk_tier": result["risk_tier"],
                "threshold_used": result["threshold_used"],
            },
            "probability": result["probability"],
            "risk_tier": result["risk_tier"],
            "predicted_class": result["predicted_class"],
            "shap_values": result["shap_values"],
            "shap_base_value": result["shap_base_value"],
        }


class ExplainerAgent:
    name = "Explainability Agent"

    def __init__(self, feature_labels: dict):
        self.feature_labels = feature_labels

    def _label(self, feat):
        for key, label in self.feature_labels.items():
            if feat.startswith(key):
                return label
        return feat

    def run(self, patient: dict, prediction: dict, top_n: int = 4) -> dict:
        shap_items = sorted(prediction["shap_values"].items(), key=lambda x: -abs(x[1]))
        top_factors = shap_items[:top_n]
        raising = [(f, v) for f, v in top_factors if v > 0]
        lowering = [(f, v) for f, v in top_factors if v < 0]

        facts = (
            f"Predicted risk tier: {prediction['risk_tier']} "
            f"(probability {prediction['probability']:.2f}).\n"
            f"Top factors RAISING risk: {', '.join(self._label(f) for f, _ in raising) or 'none'}.\n"
            f"Top factors LOWERING risk: {', '.join(self._label(f) for f, _ in lowering) or 'none'}.\n"
        )
        prompt = (
            "You are a health-explainer assistant. Using ONLY the facts below, write a "
            "2-3 sentence plain-language explanation of why this risk score came out this "
            "way. Do not add facts not given. Do not diagnose. End by noting this is not "
            "a diagnosis.\n\n" + facts
        )

        narrative = _call_llm(prompt)
        if narrative is None:
            parts = [f"Your estimated risk tier is **{prediction['risk_tier']}** "
                     f"(model probability {prediction['probability']:.0%})."]
            if raising:
                parts.append("This is driven mainly by: "
                              + ", ".join(self._label(f) for f, _ in raising) + ".")
            if lowering:
                parts.append("Working in your favor: "
                              + ", ".join(self._label(f) for f, _ in lowering) + ".")
            parts.append("This is a statistical estimate, not a diagnosis — "
                          "please discuss it with a clinician.")
            narrative = " ".join(parts)

        return {
            "agent": self.name,
            "output": {
                "narrative": narrative,
                "top_factors": [
                    {"feature": self._label(f), "shap_value": float(v),
                     "direction": "raises risk" if v > 0 else "lowers risk"}
                    for f, v in top_factors
                ],
            },
            "narrative": narrative,
            "top_factors": top_factors,
        }


GUIDELINE_SNIPPETS = [
    {"topic": "BMI / weight", "text": "Losing 5-7% of body weight through modest calorie "
     "reduction and regular activity is associated with lower diabetes risk in people "
     "with elevated BMI."},
    {"topic": "physical activity", "text": "At least 150 minutes of moderate activity per "
     "week is a commonly recommended target for insulin sensitivity and cardiovascular risk."},
    {"topic": "blood pressure", "text": "Persistently elevated blood pressure compounds "
     "diabetes risk; routine monitoring and clinician follow-up is recommended."},
    {"topic": "cholesterol", "text": "High cholesterol alongside high BMI or blood pressure "
     "raises cardiometabolic risk; a lipid panel is recommended if not done recently."},
    {"topic": "diet", "text": "Increasing fruit and vegetable intake and reducing processed "
     "or sugary foods is a commonly recommended first lifestyle change."},
    {"topic": "smoking", "text": "Smoking cessation improves insulin sensitivity over time, "
     "in addition to cardiovascular benefits."},
    {"topic": "alcohol", "text": "Heavy alcohol consumption is associated with worse "
     "glycemic control; reducing intake is commonly recommended."},
    {"topic": "mobility / general health", "text": "Difficulty with mobility or low "
     "self-rated health is linked to reduced activity, itself a modifiable risk factor — "
     "low-impact activity plans are often recommended here."},
    {"topic": "lab follow-up", "text": "Elevated HbA1c or blood glucose warrants a repeat "
     "confirmatory lab test and clinician follow-up per standard care guidelines."},
    {"topic": "access to care", "text": "Cost-related avoidance of care delays diagnosis; "
     "community health centers and sliding-scale clinics are a starting point."},
]


class SimpleRetriever:
    """TF-IDF retrieval over the guideline snippets above."""

    def __init__(self, snippets):
        self.snippets = snippets
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([s["text"] for s in snippets])

    def retrieve(self, query: str, k: int = 2):
        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        return [self.snippets[i] for i in top_idx if sims[i] > 0]


class RecommenderAgent:
    name = "Recommendation Agent"

    def __init__(self, retriever: SimpleRetriever, feature_labels: dict):
        self.retriever = retriever
        self.feature_labels = feature_labels

    def _label(self, feat):
        for key, label in self.feature_labels.items():
            if feat.startswith(key):
                return label
        return feat

    def run(self, prediction: dict, explanation: dict) -> dict:
        raising_factors = [f for f, v in explanation["top_factors"] if v > 0]
        query = " ".join(self._label(f) for f in raising_factors) or "general diabetes prevention"
        retrieved = self.retriever.retrieve(query, k=3)

        prompt = (
            "You are a lifestyle-recommendation assistant for a diabetes early-warning "
            "tool. Using ONLY the guideline snippets below, write up to 3 short, "
            "non-diagnostic suggestions. Never state a diagnosis.\n\n"
            f"Risk tier: {prediction['risk_tier']}\nRelevant snippets:\n"
            + "\n".join(f"- [{s['topic']}] {s['text']}" for s in retrieved)
        )

        narrative = _call_llm(prompt)
        if narrative is None:
            if not retrieved:
                narrative = ("No specific risk-driving factors were flagged. General "
                              "guidance: maintain regular activity, a balanced diet, "
                              "and routine checkups.")
            else:
                bullets = [f"- Re: {s['topic']} — {s['text']}" for s in retrieved]
                narrative = ("Based on your top risk factors, consider:\n" + "\n".join(bullets)
                              + "\n\nThese are general suggestions, not medical advice — "
                              "please discuss any changes with a clinician.")

        return {
            "agent": self.name,
            "output": {
                "query_used": query,
                "recommendations": [s["text"] for s in retrieved],
                "sources": [s["topic"] for s in retrieved],
                "narrative": narrative,
            },
            "narrative": narrative,
            "sources": retrieved,
        }


class AlertAgent:
    name = "Alert & Escalation Agent"

    def __init__(self, high_risk_tier="High Risk"):
        self.high_risk_tier = high_risk_tier
        self.alert_log = []

    def run(self, patient_id, prediction: dict) -> dict:
        should_alert = prediction["risk_tier"] == self.high_risk_tier
        action = "Escalated to clinician alert queue" if should_alert else "No escalation required"
        record = {
            "patient_id": patient_id,
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "risk_tier": prediction["risk_tier"],
            "probability": round(prediction["probability"], 4),
            "alert_triggered": should_alert,
        }
        if should_alert:
            self.alert_log.append(record)

        return {
            "agent": self.name,
            "output": {"decision": action, "alert_triggered": should_alert, "alert_record": record},
            "patient_id": patient_id,
            "timestamp": record["timestamp"],
            "risk_tier": record["risk_tier"],
            "probability": record["probability"],
            "alert_triggered": should_alert,
        }


BANNED_PATTERNS = [
    "you have diabetes", "you are diabetic", "i diagnose", "this confirms you",
    "you definitely", "you will develop",
]
DISCLAIMER = "This is a statistical estimate, not a medical diagnosis. Please consult a clinician."


def apply_guardrails(text: str) -> str:
    lowered = text.lower()
    for phrase in BANNED_PATTERNS:
        if phrase in lowered:
            text += f"\n\n[GUARDRAIL] Flagged diagnostic-sounding language ('{phrase}')."
    if "diagnos" not in lowered:
        text += f"\n\n{DISCLAIMER}"
    return text


class Orchestrator:
    def __init__(self, prediction_agent, explainer_agent, recommender_agent, alert_agent):
        self.prediction_agent = prediction_agent
        self.explainer_agent = explainer_agent
        self.recommender_agent = recommender_agent
        self.alert_agent = alert_agent
        self.run_log = []

    def _log(self, step, detail="", elapsed_seconds=None):
        entry = {"step": step, "time": dt.datetime.now().isoformat(timespec="seconds"), "detail": detail}
        if elapsed_seconds is not None:
            entry["elapsed_seconds"] = round(elapsed_seconds, 4)
        self.run_log.append(entry)

    def _run_agent(self, name, fn):
        start = time.perf_counter()
        self._log(f"{name}: start")
        result = fn()
        elapsed = time.perf_counter() - start
        self._log(f"{name}: done", elapsed_seconds=elapsed)
        return result, elapsed

    def run(self, patient: dict, patient_id="demo_patient") -> dict:
        self.run_log = []

        prediction, t1 = self._run_agent("Prediction Agent", lambda: self.prediction_agent.run(patient))
        explanation, t2 = self._run_agent("Explainability Agent", lambda: self.explainer_agent.run(patient, prediction))
        recommendation, t3 = self._run_agent("Recommendation Agent", lambda: self.recommender_agent.run(prediction, explanation))
        alert, t4 = self._run_agent("Alert Agent", lambda: self.alert_agent.run(patient_id, prediction))

        agent_outputs = {
            "Prediction Agent": {"output": prediction, "elapsed_seconds": round(t1, 4)},
            "Explainability Agent": {"output": explanation, "elapsed_seconds": round(t2, 4)},
            "Recommendation Agent": {"output": recommendation, "elapsed_seconds": round(t3, 4)},
            "Alert Agent": {"output": alert, "elapsed_seconds": round(t4, 4)},
        }

        return {
            "patient_id": patient_id,
            "risk_tier": prediction["risk_tier"],
            "probability": prediction["probability"],
            "agent_outputs": agent_outputs,
            "explanation": apply_guardrails(explanation["narrative"]),
            "recommendation": apply_guardrails(recommendation["narrative"]),
            "alert": alert,
            "run_log": list(self.run_log),
        }


def build_pipeline(score_fn, feature_labels: dict) -> Orchestrator:
    """Wires up all four agents around one pipeline's scoring function."""
    return Orchestrator(
        PredictionAgent(score_fn),
        ExplainerAgent(feature_labels),
        RecommenderAgent(SimpleRetriever(GUIDELINE_SNIPPETS), feature_labels),
        AlertAgent(),
    )
