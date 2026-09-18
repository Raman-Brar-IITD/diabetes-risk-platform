"""Prediction -> Explanation -> Recommendation -> Alert agent pipeline.

Works with either pipeline: pass in a scoring function and a feature-label
dict via build_pipeline(). Guardrails screen every patient-facing string
for diagnostic-sounding language, neutralize it, and append the required
disclaimer.

Settings (LLM model/timeout/retries, how many SHAP factors to explain, how
many guideline snippets to retrieve, which risk tiers alert) come from the
optional `agents:` block in config/model_config.yaml, with hardcoded
fallbacks if that file or block is missing — see _load_agent_settings().
Model names can be overridden per-deployment with the ANTHROPIC_MODEL /
OPENAI_MODEL env vars without touching the config file.
"""
import os
import re
import time
import logging
import datetime as dt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "model_config.yaml"
)


def _load_agent_settings() -> dict:
    """Reads the optional `agents:` block of config/model_config.yaml.
    Never raises — falls back to hardcoded defaults if the file, the yaml
    package, or the block itself is missing, so importing this module
    works the same in tests, scripts, and the app regardless of cwd."""
    defaults = {
        "top_n_factors": 4,
        "retrieval_k": 3,
        "llm_model_anthropic": "claude-sonnet-5",
        "llm_model_openai": "gpt-4o-mini",
        "llm_max_tokens": 250,
        "llm_timeout_seconds": 15,
        "llm_max_retries": 2,
        "alert_tiers": ["High Risk"],
    }
    try:
        import yaml
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        defaults.update(cfg.get("agents") or {})
    except Exception as e:  # missing file, bad yaml, no yaml package, etc.
        logger.debug("Falling back to default agent settings (%s)", e)
    return defaults


_SETTINGS = _load_agent_settings()

# Model names are the one thing worth overriding per-deployment (e.g. to
# pin a specific dated snapshot, or swap providers) without editing config.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", _SETTINGS["llm_model_anthropic"])
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", _SETTINGS["llm_model_openai"])
LLM_MAX_TOKENS = int(_SETTINGS["llm_max_tokens"])
LLM_TIMEOUT_SECONDS = float(_SETTINGS["llm_timeout_seconds"])
LLM_MAX_RETRIES = max(1, int(_SETTINGS["llm_max_retries"]))
DEFAULT_TOP_N_FACTORS = int(_SETTINGS["top_n_factors"])
DEFAULT_RETRIEVAL_K = int(_SETTINGS["retrieval_k"])
DEFAULT_ALERT_TIERS = tuple(_SETTINGS["alert_tiers"])


def llm_configured() -> bool:
    """Whether an LLM key is set at all, so the dashboard can explain why
    narratives are templated (no key) vs. show them as AI-generated."""
    return bool(ANTHROPIC_API_KEY or OPENAI_API_KEY)


def _try_anthropic(prompt: str, max_tokens: int, timeout: float) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=timeout)
    msg = client.messages.create(
        model=ANTHROPIC_MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _try_openai(prompt: str, max_tokens: int, timeout: float) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=timeout)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# Successful completions only — a transient failure should get a fresh
# retry next time the same prompt comes in, not a cached None forever.
_llm_cache: dict = {}


def _call_llm(prompt: str):
    """Returns an LLM string, or None if no key is set / every attempt with
    every configured provider fails. Tries Anthropic first, then OpenAI —
    unlike a single-provider-only attempt, a down/misconfigured primary
    provider no longer silently forces every narrative to the template
    fallback when a second key is also available. Retries each provider
    up to LLM_MAX_RETRIES times before moving on."""
    cache_key = prompt
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    providers = [
        ("Anthropic", ANTHROPIC_API_KEY, _try_anthropic),
        ("OpenAI", OPENAI_API_KEY, _try_openai),
    ]
    for provider_name, key, fn in providers:
        if not key:
            continue
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                result = fn(prompt, LLM_MAX_TOKENS, LLM_TIMEOUT_SECONDS)
                _llm_cache[cache_key] = result
                return result
            except ImportError as e:
                logger.warning(
                    "%s configured but its SDK isn't installed (%s); "
                    "trying next provider / falling back to template.", provider_name, e,
                )
                break  # retrying won't help a missing package
            except Exception as e:
                logger.warning(
                    "%s call failed (attempt %d/%d): %s", provider_name, attempt, LLM_MAX_RETRIES, e,
                )
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


def label_for(feat: str, feature_labels: dict) -> str:
    """Human-readable label for a raw (possibly one-hot-encoded) feature
    name, e.g. 'BMI_Category_Obese' -> 'BMI category (Obese)' given
    {'BMI_Category': 'BMI category'}. Falls back to the longest matching
    prefix (so a more specific key like 'BMI_Category' wins over the
    shorter 'BMI'), and finally to the raw name with underscores turned
    into spaces if nothing matches at all.
    """
    if feat in feature_labels:
        return feature_labels[feat]

    best_key = None
    for key in feature_labels:
        if feat == key or feat.startswith(key + "_"):
            if best_key is None or len(key) > len(best_key):
                best_key = key
    if best_key is None:
        return feat.replace("_", " ")

    label = feature_labels[best_key]
    suffix = feat[len(best_key):].lstrip("_").replace("_", " ")
    return f"{label} ({suffix})" if suffix else label


class ExplainerAgent:
    name = "Explainability Agent"

    def __init__(self, feature_labels: dict):
        self.feature_labels = feature_labels

    def _label(self, feat):
        return label_for(feat, self.feature_labels)

    def run(self, patient: dict, prediction: dict, top_n: int = DEFAULT_TOP_N_FACTORS) -> dict:
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
        source = "llm" if narrative is not None else "template"
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
                "narrative_source": source,
                "top_factors": [
                    {"feature": self._label(f), "shap_value": float(v),
                     "direction": "raises risk" if v > 0 else "lowers risk"}
                    for f, v in top_factors
                ],
            },
            "narrative": narrative,
            "narrative_source": source,
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

    def __init__(self, retriever: SimpleRetriever, feature_labels: dict, k: int = DEFAULT_RETRIEVAL_K):
        self.retriever = retriever
        self.feature_labels = feature_labels
        self.k = k

    def _label(self, feat):
        return label_for(feat, self.feature_labels)

    def run(self, prediction: dict, explanation: dict) -> dict:
        raising_factors = [f for f, v in explanation["top_factors"] if v > 0]
        query = " ".join(self._label(f) for f in raising_factors) or "general diabetes prevention"
        retrieved = self.retriever.retrieve(query, k=self.k)

        prompt = (
            "You are a lifestyle-recommendation assistant for a diabetes early-warning "
            "tool. Using ONLY the guideline snippets below, write up to 3 short, "
            "non-diagnostic suggestions. Never state a diagnosis.\n\n"
            f"Risk tier: {prediction['risk_tier']}\nRelevant snippets:\n"
            + "\n".join(f"- [{s['topic']}] {s['text']}" for s in retrieved)
        )

        narrative = _call_llm(prompt)
        source = "llm" if narrative is not None else "template"
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
                "narrative_source": source,
            },
            "narrative": narrative,
            "narrative_source": source,
            "sources": retrieved,
        }


class AlertAgent:
    name = "Alert & Escalation Agent"

    def __init__(self, alert_tiers=DEFAULT_ALERT_TIERS):
        if isinstance(alert_tiers, str):
            alert_tiers = (alert_tiers,)
        self.alert_tiers = set(alert_tiers)
        self.alert_log = []

    def run(self, patient_id, prediction: dict) -> dict:
        should_alert = prediction["risk_tier"] in self.alert_tiers
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
    "you definitely", "you will develop", "you have been diagnosed", "this is your diagnosis",
    "you do not have diabetes", "you are not diabetic",
]
DISCLAIMER = "This is a statistical estimate, not a medical diagnosis. Please consult a clinician."
_REDACTION = "[a possible risk factor pattern]"
# Matches wording that already says "not a (medical) diagnosis" — narrower than
# the old bare "diagnos" substring, which also matched e.g. "further diagnosis
# may be needed" and wrongly suppressed the disclaimer.
_ALREADY_DISCLAIMED = re.compile(r"\bnot (?:a |an )?(?:medical )?diagnos", re.IGNORECASE)


def apply_guardrails(text: str) -> str:
    """Neutralizes diagnostic-sounding phrasing in patient-facing text
    rather than merely flagging it (the flag alone previously still left
    the offending sentence visible), then appends the standard disclaimer
    unless it's already present. Checks for the literal disclaimer text
    rather than the substring 'diagnos', which used to also match — and
    therefore silently suppress the disclaimer for — any narrative that
    happened to mention "diagnosis" without actually including it.
    """
    cleaned = text
    flagged = []
    for phrase in BANNED_PATTERNS:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        if pattern.search(cleaned):
            flagged.append(phrase)
            cleaned = pattern.sub(_REDACTION, cleaned)

    if flagged:
        cleaned += ("\n\n[GUARDRAIL] Removed diagnostic-sounding language: "
                    + ", ".join(f"'{p}'" for p in flagged) + ".")
    if DISCLAIMER not in cleaned and not _ALREADY_DISCLAIMED.search(cleaned):
        cleaned += f"\n\n{DISCLAIMER}"
    return cleaned


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

    def _run_safe_agent(self, name, fn, fallback: dict):
        """Like _run_agent, but a failure here (e.g. a flaky LLM provider,
        or a retrieval bug) degrades to `fallback` instead of taking down
        the whole assessment — the prediction is the one thing worth
        actually showing the person, and it's already computed by the
        time this runs."""
        start = time.perf_counter()
        self._log(f"{name}: start")
        try:
            result = fn()
        except Exception as e:
            logger.exception("%s failed; using fallback output", name)
            result = dict(fallback)
            result["fallback_used"] = True
            result["fallback_reason"] = str(e)
            self._log(f"{name}: fallback", detail=str(e))
        elapsed = time.perf_counter() - start
        self._log(f"{name}: done", elapsed_seconds=elapsed)
        return result, elapsed

    def run(self, patient: dict, patient_id="demo_patient") -> dict:
        self.run_log = []

        prediction, t1 = self._run_agent("Prediction Agent", lambda: self.prediction_agent.run(patient))

        explanation, t2 = self._run_safe_agent(
            "Explainability Agent",
            lambda: self.explainer_agent.run(patient, prediction),
            fallback={
                "narrative": "An explanation couldn't be generated for this assessment. "
                             "The risk tier and probability above still reflect the model's output.",
                "narrative_source": "error_fallback",
                "top_factors": [],
            },
        )
        recommendation, t3 = self._run_safe_agent(
            "Recommendation Agent",
            lambda: self.recommender_agent.run(prediction, explanation),
            fallback={
                "narrative": "General guidance: maintain regular activity, a balanced diet, "
                             "and routine checkups.",
                "narrative_source": "error_fallback",
                "sources": [],
            },
        )
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
            "explanation_source": explanation.get("narrative_source", "unknown"),
            "recommendation": apply_guardrails(recommendation["narrative"]),
            "recommendation_source": recommendation.get("narrative_source", "unknown"),
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
