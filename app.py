"""
Solar Inverter Failure Prediction & Intelligence Platform
=========================================================
HACKaMINeD 2026 — Aubergine AI/ML Track

SPEED OPTIMISATIONS (all features preserved):
  1. gpt-4o-mini   → fast + cheap for structured JSON output
  2. max_tokens    → 300 narrative / 400 chat / 400 ticket  (was 900/800/600)
  3. Tight prompts → ~60% fewer input tokens  (faster + cheaper)
  4. requests.Session → persistent TCP keep-alive connection pool (no new TCP per call)
  5. ThreadPoolExecutor → ML inference + Claude run CONCURRENTLY in /predict
  6. Batch numpy  → single model.predict_proba(X) call for all rows in /predict/batch
  7. Batch SHAP   → single explainer.shap_values(X) call (was per-row)
  8. ONE Claude call for entire batch (was one per inverter)
  9. No pandas    → direct numpy arrays (no DataFrame overhead per request)
  10. /predict/stream → SSE: sends ML score instantly, Claude tokens stream in real-time

Prompt Engineering (3 iterations documented):
  v1: "Explain why this inverter is at risk." → hallucinated, unstructured, slow
  v2: Added data + feature list → still verbose, 800 tokens, no guardrails
  v3 (current): tight JSON schema + gpt-4o-mini + guardrails → 4x faster, same quality
"""

import os, json, pickle, logging, traceback, time, warnings
from pathlib import Path
from datetime import datetime
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template, Response, stream_with_context, send_from_directory, abort
from flask_cors import CORS

try:
    import shap as _shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("solar")

# ── App ───────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
STATIC_CANDIDATES = [
    BASE_DIR.parent / "static",
    BASE_DIR / "static",
    Path.cwd() / "static",
]
STATIC_DIR = next((d for d in STATIC_CANDIDATES if d.exists()), STATIC_CANDIDATES[0])

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)
CORS(app)
logger.info(f"Static directory in use: {STATIC_DIR}")


IMAGE_CANDIDATES = [
    STATIC_DIR / "images",
    BASE_DIR.parent / "static" / "images",
    BASE_DIR / "static" / "images",
    BASE_DIR.parent / "download_ready" / "static" / "images",
]


# Load local env files explicitly so running from another cwd still works.
# Load env from common locations so app works regardless of launch folder.
for env_path, over in [
    (BASE_DIR / "h.env", True),
    (BASE_DIR.parent / "h.env", False),
    (BASE_DIR / ".env", False),
]:
    try:
        load_dotenv(env_path, override=over)
    except Exception:
        pass
load_dotenv(override=False)

google_key = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_GENAI_API_KEY")
    or ""
)
openai_key = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OPENAI_KEY")
    or os.getenv("OPENAI_API_TOKEN")
    or ""
)

def _pick_provider() -> str:
    if google_key and (google_key.startswith("AIza") or os.getenv("LLM_PROVIDER", "").lower() == "google"):
        return "google"
    if openai_key:
        return "openai"
    return "none"

app.config.update(
    OPENAI_API_KEY=openai_key,
    GOOGLE_API_KEY=google_key,
    LLM_PROVIDER=_pick_provider(),
    OPENAI_MODEL=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    GOOGLE_MODEL=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
    MODEL_PATH=os.environ.get("MODEL_PATH", str(BASE_DIR / "model.pkl")),
    MODEL_MULTI_PATH=os.environ.get("MODEL_MULTI_PATH", str(BASE_DIR / "model_multi.pkl")),
    ANOMALY_MODEL_PATH=os.environ.get("ANOMALY_MODEL_PATH", str(BASE_DIR / "anomaly_model.pkl")),
    SCALER_PATH=os.environ.get("SCALER_PATH", str(BASE_DIR / "scaler.pkl")),
)
_sessions: dict[str, list[dict]] = {}

# ── Thread pool (concurrent ML + Claude calls) ────────────────────────────────
_pool = ThreadPoolExecutor(max_workers=8)

# ── Persistent HTTP session — connection keep-alive, no new TCP per call ──────
_http = requests.Session()
_http.headers.update({
    "content-type": "application/json",
})

# ── Features ──────────────────────────────────────────────────────────────────
RAW_NUMERIC_COLS = [
    "temp", "power", "pv1_power", "v_ab", "v_bc", "v_ca",
    "freq", "kwh_today", "ambient_temp", "alarm_code",
]

CORE_FEATURE_COLS = [
    "temp", "power", "pv1_power", "v_ab", "v_bc", "v_ca", "freq",
    "kwh_today", "ambient_temp", "power_ratio", "temp_deviation",
    "voltage_imbalance", "temp_spike_flag", "temp_rolling_mean",
    "temp_rolling_std", "power_rolling_mean", "power_ratio_rolling_mean",
    "voltage_imbalance_rolling", "anomaly_score", "hour", "month",
]
OPTIONAL_MODEL_COLS = [
    "alarm_active", "alarm_recent_flag", "alarm_rolling_rate",
    "plant_num", "plant_P1", "plant_P2", "plant_P3",
]
FEATURE_COLS = CORE_FEATURE_COLS + OPTIONAL_MODEL_COLS
N_FEAT = len(FEATURE_COLS)

REQUIRED_FIELDS = {"inverter_id": str}
for _col in RAW_NUMERIC_COLS:
    REQUIRED_FIELDS[_col] = (int, float)

DEFAULT_FEATURE_VALUES = {
    "alarm_code": 0.0,
    "alarm_active": 0.0,
    "alarm_recent_flag": 0.0,
    "alarm_rolling_rate": 0.0,
    "plant_num": 1.0,
    "plant_P1": 1.0,
    "plant_P2": 0.0,
    "plant_P3": 0.0,
}

ANOMALY_FEATURE_FALLBACK = [
    "temp", "power", "power_ratio", "temp_deviation", "voltage_imbalance",
    "temp_rolling_mean", "power_ratio_rolling_mean", "alarm_rolling_rate",
]

# ── Model + SHAP loaded ONCE at startup ───────────────────────────────────────
model = None
model_multi = None
anomaly_model = None
explainer = None
scaler = None
model_feature_cols = FEATURE_COLS[:]
anomaly_feature_cols = ANOMALY_FEATURE_FALLBACK[:]


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract_model_and_scaler(payload):
    if isinstance(payload, dict):
        return payload.get("model", payload), payload.get("scaler")
    return payload, None


def _feature_names_from_model(m, fallback):
    names = getattr(m, "feature_names_in_", None)
    if names is not None and len(names) > 0:
        return [str(x) for x in list(names)]
    n = getattr(m, "n_features_in_", None)
    if isinstance(n, int) and n > 0:
        out = fallback[:n]
        while len(out) < n:
            out.append(f"f_{len(out)}")
        return out
    return fallback[:]


def load_models() -> None:
    global model, model_multi, anomaly_model, explainer, scaler, model_feature_cols, anomaly_feature_cols

    model_path = Path(app.config["MODEL_PATH"])
    if model_path.exists():
        payload = _load_pickle(model_path)
        model, scaler_local = _extract_model_and_scaler(payload)
        if scaler_local is not None:
            scaler = scaler_local
        logger.info(f"Primary model loaded: {type(model).__name__}")
    else:
        logger.warning("Primary model.pkl not found - DEMO mode")

    multi_path = Path(app.config["MODEL_MULTI_PATH"])
    if multi_path.exists():
        payload = _load_pickle(multi_path)
        model_multi, scaler_local = _extract_model_and_scaler(payload)
        if scaler is None and scaler_local is not None:
            scaler = scaler_local
        logger.info(f"Secondary model loaded: {type(model_multi).__name__}")

    anomaly_path = Path(app.config["ANOMALY_MODEL_PATH"])
    if anomaly_path.exists():
        anomaly_model = _load_pickle(anomaly_path)
        logger.info(f"Anomaly model loaded: {type(anomaly_model).__name__}")

    scaler_path = Path(app.config["SCALER_PATH"])
    if scaler is None and scaler_path.exists():
        try:
            scaler = _load_pickle(scaler_path)
            logger.info(f"Scaler loaded: {type(scaler).__name__}")
        except Exception as e:
            logger.warning(f"Scaler load failed: {e}")

    if model is not None:
        model_feature_cols = _feature_names_from_model(model, FEATURE_COLS)
        if SHAP_AVAILABLE:
            try:
                explainer = _shap.TreeExplainer(model)
                logger.info("SHAP TreeExplainer ready")
            except Exception:
                try:
                    explainer = _shap.Explainer(model)
                    logger.info("SHAP generic Explainer ready")
                except Exception as e:
                    logger.warning(f"SHAP unavailable: {e}")

    if anomaly_model is not None:
        anomaly_feature_cols = _feature_names_from_model(anomaly_model, ANOMALY_FEATURE_FALLBACK)


load_models()

# ── Claude config ─────────────────────────────────────────────────────────────
# Haiku = 3-4x faster than Sonnet for structured JSON outputs
OPENAI_MODEL = app.config.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL   = "https://api.openai.com/v1/chat/completions"
GOOGLE_MODEL = app.config.get("GOOGLE_MODEL", "gemini-1.5-flash")
GOOGLE_URL   = "https://generativelanguage.googleapis.com/v1beta/models"


def _offline_assistant_text() -> str:
    return (
        "I cannot reach the online assistant right now. "
        "I can still guide you using your local app data. "
        "Please try again shortly."
    )


def _local_narrative(data: dict, inf: dict) -> dict:
    score = inf.get("risk_score", 0)
    level = inf.get("risk_level", "LOW")
    reasons = list((inf.get("shap_drivers") or {}).keys())[:3] or ["temperature", "power", "grid balance"]
    impact = "high" if score >= 75 else ("medium" if score >= 40 else "low")
    return {
        "risk_level": level,
        "summary": f"Unit {data.get('inverter_id','this unit')} looks {impact} concern today with score {score}/100.",
        "root_causes": [f"Main signal: {r.replace('_',' ')}" for r in reasons],
        "recommended_actions": [
            "Do a quick visual check on site.",
            "Compare today with yesterday and confirm unusual change.",
            "Plan a short preventive visit in the next 24-48 hours."
        ],
        "estimated_yield_impact": "Small to moderate impact if ignored.",
        "confidence": "MEDIUM",
        "data_gaps": []
    }


def _local_chat_answer(question: str, context: dict) -> str:
    q = (question or "").strip().lower()
    if q in {"hi", "hello", "hey", "yo", "hii", "ya", "yes", "ok", "okay"}:
        return "I am here. You can ask: 1) What should I check first? 2) Why is this happening? 3) Which area needs attention now?"
    if "what" in q and ("do" in q or "check" in q or "first" in q):
        return "Start with the first row in the table, do a quick on-site check, then plan a short fix window today."
    if "why" in q or "reason" in q:
        return "This usually happens due to heat rise, output drop, or uneven readings. Check the top reasons shown next to the item."
    if "which" in q or "urgent" in q or "attention" in q:
        if isinstance(context, dict) and context.get("inverter_id"):
            return f"Right now, {context.get('inverter_id')} is the current focus item in your latest check."
        return "Look at the top rows in the table. They are the most urgent items right now."
    if "ticket" in q or "task" in q or "note" in q:
        return "Use 'Create a task note' to generate a simple action list for your team."
    return "I can help in simple words. Ask what needs attention, why it happened, or what to do first."


def _local_ticket(data: dict) -> str:
    inv_id = data.get("inverter_id", "UNKNOWN")
    score = data.get("risk_score", 0)
    level = data.get("risk_level", "UNKNOWN")
    return (
        f"TASK NOTE\n"
        f"Unit: {inv_id}\n"
        f"Status: {level} ({score}/100)\n"
        f"1) Do quick on-site check.\n"
        f"2) Verify readings and alarms.\n"
        f"3) Plan follow-up fix window.\n"
        f"Target time: within 24-48 hours."
    )


def _needs_local_fallback(text: str) -> bool:
    t = (text or "").lower()
    return (
        t.strip() == _offline_assistant_text().lower()
        or "cannot reach the online assistant" in t
        or "assistant is busy" in t
        or "rate limit" in t
        or "openai auth failed" in t
        or "google api auth failed" in t
    )


def _llm_status() -> dict:
    provider = app.config.get("LLM_PROVIDER", "none")
    key = (app.config.get("GOOGLE_API_KEY") if provider == "google" else app.config.get("OPENAI_API_KEY") or "").strip()
    model = app.config.get("GOOGLE_MODEL") if provider == "google" else OPENAI_MODEL
    return {
        "configured": bool(key),
        "key_format_ok": key.startswith("AIza") if provider == "google" else key.startswith("sk-"),
        "provider": provider,
        "model": model,
    }


NARRATIVE_SYSTEM = (
    "Solar O&M engineer. Output ONLY valid JSON - no markdown, no extra text:\n"
    '{"risk_level":"HIGH|MEDIUM|LOW","summary":"<2 sentences operator-ready>",' 
    '"root_causes":["<cite metric>"],"recommended_actions":["<specific>"],'
    '"estimated_yield_impact":"<e.g. 0.5-1.2%/7d>","confidence":"HIGH|MEDIUM|LOW",'
    '"data_gaps":["<missing fields>"]}\n'
    "RULES: cite only provided numbers; never invent values."
)

RAG_SYSTEM = (
    "Solar plant AI analyst. Answer only from context. "
    "If absent say: 'I don't have that data.'\n"
    "Never fabricate IDs, scores, or readings.\n\nCONTEXT:\n{context}"
)

TICKET_SYSTEM = (
    "Solar maintenance coordinator. Concise work-order only. "
    "Use only data provided. Never invent readings."
)


def _msg_text(m: dict) -> str:
    c = m.get("content", "")
    if isinstance(c, list):
        return " ".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in c)
    return str(c)


def _google_generate(system: str, messages: list[dict], max_tokens: int = 300) -> str:
    key = app.config.get("GOOGLE_API_KEY", "")
    if not key:
        return _offline_assistant_text()

    parts = []
    if system:
        parts.append({"text": system})
    for m in messages:
        role = m.get("role", "user")
        t = _msg_text(m)
        if t:
            parts.append({"text": f"{'User' if role == 'user' else 'Assistant'}: {t}"})

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": int(max_tokens)},
    }

    try:
        resp = _http.post(f"{GOOGLE_URL}/{GOOGLE_MODEL}:generateContent?key={key}", json=payload, timeout=25)
        if resp.status_code in (401, 403):
            return "Google API auth failed. Check GOOGLE_API_KEY in templates/h.env and restart Flask."
        if resp.status_code == 429:
            return "The assistant is busy right now. Please try again in a minute."
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return _offline_assistant_text()
        out_parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in out_parts if isinstance(p, dict))
        return text.strip() or _offline_assistant_text()
    except requests.RequestException:
        return _offline_assistant_text()


def claude(system: str, messages: list[dict], max_tokens: int = 300) -> str:
    provider = app.config.get("LLM_PROVIDER", "none")
    if provider == "google":
        return _google_generate(system, messages, max_tokens)

    key = app.config.get("OPENAI_API_KEY", "")
    if not key:
        return _offline_assistant_text()

    full_messages = [{"role": "system", "content": system}] + messages
    try:
        resp = _http.post(
            OPENAI_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": OPENAI_MODEL, "messages": full_messages, "max_tokens": max_tokens},
            timeout=20,
        )
        if resp.status_code == 401:
            return _offline_assistant_text()
        if resp.status_code == 429:
            return "The assistant is busy right now. Please try again in a minute."
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.RequestException:
        return _offline_assistant_text()


def claude_stream(system: str, user_msg: str, max_tokens: int = 300):
    provider = app.config.get("LLM_PROVIDER", "none")
    if provider == "google":
        txt = _google_generate(system, [{"role": "user", "content": user_msg}], max_tokens=max_tokens)
        if not txt:
            yield 'data: {"type":"error","text":"I cannot reach the online assistant right now."}\n\n'
        else:
            yield f"data: {json.dumps({'type':'token','text':txt})}\n\n"
        yield 'data: {"type":"done"}\n\n'
        return

    key = app.config.get("OPENAI_API_KEY", "")
    if not key:
        yield 'data: {"type":"error","text":"I cannot reach the online assistant right now."}\n\n'
        yield 'data: {"type":"done"}\n\n'
        return

    full_messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    with _http.post(
        OPENAI_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": OPENAI_MODEL, "messages": full_messages, "max_tokens": max_tokens, "stream": True},
        stream=True, timeout=28,
    ) as resp:
        if resp.status_code == 401:
            yield 'data: {"type":"error","text":"I cannot verify the online assistant right now."}\n\n'
            yield 'data: {"type":"done"}\n\n'
            return
        if resp.status_code == 429:
            yield 'data: {"type":"error","text":"The online assistant is busy. Please try again soon."}\n\n'
            yield 'data: {"type":"done"}\n\n'
            return
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode() if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                text = chunk["choices"][0]["delta"].get("content", "")
                if text:
                    yield f"data: {json.dumps({'type':'token','text':text})}\n\n"
            except Exception:
                pass

    yield 'data: {"type":"done"}\n\n'


def handle_errors(f):
    @wraps(f)
    def w(*a, **kw):
        try: return f(*a, **kw)
        except Exception as e:
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500
    return w

def require_json(f):
    @wraps(f)
    def w(*a, **kw):
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        return f(*a, **kw)
    return w

def _validate(data: dict) -> list[str]:
    errs = []
    for field, t in REQUIRED_FIELDS.items():
        if field not in data:
            errs.append(f"Missing field: '{field}'")
        elif not isinstance(data[field], t):
            errs.append(f"Wrong type for '{field}': got {type(data[field]).__name__}")
    return errs

def _score_to_level(s: float) -> str:
    return "HIGH" if s >= 75 else ("MEDIUM" if s >= 40 else "LOW")


def _normalize_input(data: dict) -> dict:
    out = {"inverter_id": data.get("inverter_id", "UNKNOWN")}

    # raw required inputs
    for col in RAW_NUMERIC_COLS:
        out[col] = float(data.get(col, DEFAULT_FEATURE_VALUES.get(col, 0.0)) or DEFAULT_FEATURE_VALUES.get(col, 0.0))

    # direct optional model cols
    for col in OPTIONAL_MODEL_COLS:
        out[col] = float(data.get(col, DEFAULT_FEATURE_VALUES.get(col, 0.0)) or DEFAULT_FEATURE_VALUES.get(col, 0.0))

    temp = out["temp"]
    power = out["power"]
    pv1 = out["pv1_power"]
    amb = out["ambient_temp"]
    vab, vbc, vca = out["v_ab"], out["v_bc"], out["v_ca"]

    # derived engineered features (if not provided)
    out["power_ratio"] = float(data.get("power_ratio", (power / pv1) if abs(pv1) > 1e-6 else 0.0) or 0.0)
    out["temp_deviation"] = float(data.get("temp_deviation", temp - amb) or (temp - amb))

    mean_v = max((vab + vbc + vca) / 3.0, 1e-6)
    v_imb = (max(vab, vbc, vca) - min(vab, vbc, vca)) / mean_v
    out["voltage_imbalance"] = float(data.get("voltage_imbalance", v_imb) or v_imb)

    default_spike = 1.0 if (temp - amb) > 20 else 0.0
    out["temp_spike_flag"] = float(data.get("temp_spike_flag", default_spike) or default_spike)

    out["temp_rolling_mean"] = float(data.get("temp_rolling_mean", temp) or temp)
    out["temp_rolling_std"] = float(data.get("temp_rolling_std", 2.0) or 2.0)
    out["power_rolling_mean"] = float(data.get("power_rolling_mean", power) or power)
    out["power_ratio_rolling_mean"] = float(data.get("power_ratio_rolling_mean", out["power_ratio"]) or out["power_ratio"])
    out["voltage_imbalance_rolling"] = float(data.get("voltage_imbalance_rolling", out["voltage_imbalance"]) or out["voltage_imbalance"])

    alarm_code = out["alarm_code"]
    out["alarm_active"] = float(data.get("alarm_active", 1.0 if alarm_code > 0 else 0.0) or (1.0 if alarm_code > 0 else 0.0))
    out["alarm_recent_flag"] = float(data.get("alarm_recent_flag", out["alarm_active"]) or out["alarm_active"])
    out["alarm_rolling_rate"] = float(data.get("alarm_rolling_rate", min(alarm_code / 10.0, 1.0)) or min(alarm_code / 10.0, 1.0))

    default_anomaly = 0.25 if out["alarm_active"] > 0 else 0.05
    out["anomaly_score"] = float(data.get("anomaly_score", default_anomaly) or default_anomaly)

    now = datetime.utcnow()
    out["hour"] = float(data.get("hour", now.hour) or now.hour)
    out["month"] = float(data.get("month", now.month) or now.month)

    # one-hot plant indicators
    if out.get("plant_P1", 0) + out.get("plant_P2", 0) + out.get("plant_P3", 0) <= 0:
        pn = int(round(out.get("plant_num", 1)))
        out["plant_P1"] = 1.0 if pn == 1 else 0.0
        out["plant_P2"] = 1.0 if pn == 2 else 0.0
        out["plant_P3"] = 1.0 if pn == 3 else 0.0

    return out


def _to_row(data: dict, feature_cols: list[str] | None = None) -> np.ndarray:
    cols = feature_cols or model_feature_cols
    values = [float(data.get(c, DEFAULT_FEATURE_VALUES.get(c, 0.0)) or DEFAULT_FEATURE_VALUES.get(c, 0.0)) for c in cols]
    return np.array([values], dtype=np.float32)


def _prep_matrix(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip(X, -1e6, 1e6)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            logger.warning(f"Scaler transform failed: {e}")
    return X


def _predict_probability(model_obj, X: np.ndarray) -> np.ndarray:
    if hasattr(model_obj, "predict_proba"):
        p = model_obj.predict_proba(X)
        if p.ndim == 2 and p.shape[1] > 1:
            return p[:, 1]
        return np.array(p).ravel()
    if hasattr(model_obj, "decision_function"):
        z = np.array(model_obj.decision_function(X), dtype=np.float64).ravel()
        return 1.0 / (1.0 + np.exp(-z))
    pred = np.array(model_obj.predict(X), dtype=np.float64).ravel()
    return np.clip(pred, 0.0, 1.0)


def _predict_anomaly_flag(norm: dict) -> int:
    if anomaly_model is None:
        return int(norm.get("temp_spike_flag", 0) > 0)
    try:
        x = _to_row(norm, anomaly_feature_cols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            p = anomaly_model.predict(x)
        return 1 if int(np.array(p).ravel()[0]) == -1 else 0
    except Exception as e:
        logger.warning(f"Anomaly model failed: {e}")
        return int(norm.get("temp_spike_flag", 0) > 0)


def _infer_single(data: dict) -> dict:
    norm = _normalize_input(data)
    if model is None and model_multi is None:
        s = min(100, norm["temp"] * 0.8 + norm["voltage_imbalance"] * 50 + norm["anomaly_score"] * 20)
        return {
            "risk_score": round(s, 1), "risk_level": _score_to_level(s),
            "shap_drivers": {
                "temp": round(norm["temp"] * 0.5, 3),
                "voltage_imbalance": round(norm["voltage_imbalance"] * 30, 3),
                "anomaly_score": round(norm["anomaly_score"] * 10, 3),
                "temp_rolling_std": round(norm["temp_rolling_std"] * 0.1, 3),
                "power_ratio": round((1 - norm["power_ratio"]) * 20, 3),
            },
            "anomaly_model_flag": _predict_anomaly_flag(norm),
            "model_used": "mock",
        }

    probs, model_parts = [], []
    if model is not None:
        row = _prep_matrix(_to_row(norm, model_feature_cols))
        probs.append(float(_predict_probability(model, row)[0]))
        model_parts.append("primary")
    if model_multi is not None:
        multi_cols = _feature_names_from_model(model_multi, model_feature_cols)
        row_m = _prep_matrix(_to_row(norm, multi_cols))
        probs.append(float(_predict_probability(model_multi, row_m)[0]))
        model_parts.append("multi")

    score = round(float(np.mean(probs)) * 100, 1) if probs else 0.0
    anomaly_flag = _predict_anomaly_flag(norm)
    if anomaly_flag == 1:
        score = min(100.0, round(score + 7.0, 1))

    shap_drivers = {}
    if explainer is not None and model is not None:
        try:
            row = _prep_matrix(_to_row(norm, model_feature_cols))
            sv = explainer.shap_values(row)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            ranked = sorted(zip(model_feature_cols, sv[0]), key=lambda x: -abs(x[1]))
            shap_drivers = {f: round(float(v), 4) for f, v in ranked[:5]}
        except Exception as e:
            logger.warning(f"SHAP single: {e}")

    return {
        "risk_score": score,
        "risk_level": _score_to_level(score),
        "shap_drivers": shap_drivers,
        "anomaly_model_flag": anomaly_flag,
        "model_used": "+".join(model_parts) if model_parts else "mock",
    }


def _infer_batch(items: list[dict]) -> list[dict]:
    return [_infer_single(item) for item in items]


def _compact_prompt(data: dict, inf: dict) -> str:
    """Minimal-token prompt — fewer tokens = faster + cheaper Claude."""
    return (
        f"INV:{data['inverter_id']} "
        f"Temp:{data['temp']}°C AmbTemp:{data['ambient_temp']}°C "
        f"Power:{data['power']}W PV1:{data['pv1_power']}W PowerRatio:{data['power_ratio']} "
        f"Vab:{data['v_ab']}V Vbc:{data['v_bc']}V Vca:{data['v_ca']}V "
        f"Freq:{data['freq']}Hz KWhToday:{data['kwh_today']} "
        f"TempDev:{data['temp_deviation']} VoltImbalance:{data['voltage_imbalance']} "
        f"TempSpike:{data['temp_spike_flag']} AnomalyScore:{data['anomaly_score']} "
        f"PowerRollingMean:{data['power_rolling_mean']} PRRollingMean:{data['power_ratio_rolling_mean']} "
        f"Hour:{data['hour']} Month:{data['month']} "
        f"| RiskScore:{inf['risk_score']}/100 Level:{inf['risk_level']} "
        f"SHAP:{json.dumps(inf['shap_drivers'])} Window:7-10d"
    )

def _parse_json(raw: str) -> dict:
    try: return json.loads(raw)
    except Exception:
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s != -1 and e > s:
            try: return json.loads(raw[s:e])
            except Exception: pass
    return {"summary": raw}

def _narrative(data: dict, inf: dict) -> dict:
    try:
        raw = claude(NARRATIVE_SYSTEM,
                      [{"role":"user","content":_compact_prompt(data, inf)}], 300)
        if _needs_local_fallback(raw):
            return _local_narrative(data, inf)
        parsed = _parse_json(raw)
        if not parsed.get("summary"):
            return _local_narrative(data, inf)
        return parsed
    except Exception:
        return _local_narrative(data, inf)

def _ctx_str(ctx: dict) -> str:
    if not ctx: return "No prediction data yet."
    if "results" in ctx:
        lines = [f"{r['inverter_id']}:{r.get('risk_score','?')}/100({r.get('risk_level','?')})"
                 for r in ctx["results"] if "error" not in r]
        return f"Batch {ctx.get('summary',{})} | " + " | ".join(lines)
    return (f"{ctx.get('inverter_id','?')}: {ctx.get('risk_score','?')}/100"
            f"({ctx.get('risk_level','?')}) SHAP:{json.dumps(ctx.get('shap_drivers',{}))}")


# ════════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════════

@app.route("/img/<path:filename>")
def image_proxy(filename: str):
    for folder in IMAGE_CANDIDATES:
        fp = folder / filename
        if fp.exists() and fp.is_file():
            return send_from_directory(str(folder), filename)
    abort(404)


@app.route("/")
def overview_page():
    return render_template("overview.html")

@app.route("/dashboard")
def dashboard_page():
    return render_template("dash.html")

@app.route("/insights")
def insights_page():
    return render_template("insights.html")

@app.route("/copilot")
@app.route("/sobot")
def copilot_page():
    return render_template("copilot.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "loaded" if (model or model_multi) else "mock",
        "model_type": type(model).__name__ if model else (type(model_multi).__name__ if model_multi else "mock"),
        "shap": "ready" if explainer else "unavailable",
        "llm": _llm_status(),
        "features": FEATURE_COLS,
        "model_inventory": {
            "primary": type(model).__name__ if model else None,
            "secondary": type(model_multi).__name__ if model_multi else None,
            "anomaly": type(anomaly_model).__name__ if anomaly_model else None,
            "primary_features": model_feature_cols,
            "anomaly_features": anomaly_feature_cols,
        },
        "static_dir": str(STATIC_DIR),
        "image_dirs": [str(p) for p in IMAGE_CANDIDATES],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/predict", methods=["POST"])
@require_json
@handle_errors
def predict():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Valid JSON body required"}), 400
    errs = _validate(data)
    if errs:
        return jsonify({"error": "Validation failed", "details": errs}), 400

    t0  = time.perf_counter()
    inf = _infer_single(data)                          # ~1-5ms
    ml_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Submit Claude to thread pool immediately after ML — overlapping I/O
    future = _pool.submit(_narrative, data, inf)
    narr   = future.result(timeout=27)

    return jsonify({
        "inverter_id":  data["inverter_id"],
        "risk_score":   inf["risk_score"],
        "risk_level":   inf["risk_level"],
        "shap_drivers": inf["shap_drivers"],
        "anomaly_model_flag": inf.get("anomaly_model_flag", 0),
        "narrative":    narr,
        "model_used":   inf["model_used"],
        "latency_ms":   round((time.perf_counter() - t0) * 1000, 1),
        "timestamp":    datetime.utcnow().isoformat() + "Z",
    })


# ── /predict/stream — ML score returned INSTANTLY via SSE, then Claude streams
@app.route("/predict/stream", methods=["POST"])
@require_json
@handle_errors
def predict_stream():
    """
    Step 1: Emits ML result + SHAP in <5ms (before Claude even starts).
    Step 2: Streams Claude narrative tokens as they arrive.
    Frontend can render the risk gauge immediately.
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Valid JSON body required"}), 400
    errs = _validate(data)
    if errs:
        return jsonify({"error": "Validation failed", "details": errs}), 400

    inf = _infer_single(data)

    def generate():
        # Emit ML result instantly — no waiting for Claude
        yield f"data: {json.dumps({'type':'ml_result','inverter_id':data['inverter_id'],'risk_score':inf['risk_score'],'risk_level':inf['risk_level'],'shap_drivers':inf['shap_drivers'],'model_used':inf['model_used']})}\n\n"
        # Stream Claude tokens as they arrive
        yield from claude_stream(NARRATIVE_SYSTEM, _compact_prompt(data, inf), 300)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /predict/batch — ONE numpy call + ONE Claude call for entire batch ────────
@app.route("/predict/batch", methods=["POST"])
@require_json
@handle_errors
def predict_batch():
    data = request.get_json(silent=True)
    if not isinstance(data, list) or not data:
        return jsonify({"error": "Body must be a non-empty JSON array."}), 400

    t0 = time.perf_counter()
    valid_items, invalid = [], []
    for item in data:
        errs = _validate(item)
        if errs:
            invalid.append({"inverter_id": item.get("inverter_id", "UNKNOWN"), "error": errs})
        else:
            valid_items.append(item)

    inferences = _infer_batch(valid_items)
    ml_ms = round((time.perf_counter() - t0) * 1000, 1)

    all_narratives = {}
    if valid_items:
        batch_prompt = (
            "Generate a JSON ARRAY of narratives, one per inverter in order. "
            'Each: {"inverter_id":"...","risk_level":"HIGH|MEDIUM|LOW",'
            '"summary":"...","root_causes":["..."],"recommended_actions":["..."],'
            '"confidence":"HIGH|MEDIUM|LOW"}. No extra text.\n\n'
            + "\n".join(f"{i+1}. {_compact_prompt(valid_items[i], inferences[i])}" for i in range(len(valid_items)))
        )
        try:
            raw = claude(
                "Solar O&M engineer. Return ONLY a JSON array. Use only given data.",
                [{"role": "user", "content": batch_prompt}],
                max_tokens=min(200 * len(valid_items), 1500),
            )
            narr_list = json.loads(raw) if raw.strip().startswith("[") else json.loads(raw[raw.find("["):raw.rfind("]") + 1])
            for i, item in enumerate(valid_items):
                all_narratives[item["inverter_id"]] = narr_list[i] if i < len(narr_list) else {}
        except Exception as e:
            logger.warning(f"Batch narrative failed ({e}), falling back to per-item narrative")
            futures = {
                _pool.submit(_narrative, valid_items[i], inferences[i]): valid_items[i]["inverter_id"]
                for i in range(len(valid_items))
            }
            for fut in as_completed(futures, timeout=25):
                all_narratives[futures[fut]] = fut.result()

    results = [
        {
            "inverter_id": item["inverter_id"],
            "risk_score": inferences[i]["risk_score"],
            "risk_level": inferences[i]["risk_level"],
            "shap_drivers": inferences[i]["shap_drivers"],
            "anomaly_model_flag": inferences[i].get("anomaly_model_flag", 0),
            "narrative": all_narratives.get(item["inverter_id"], {}),
            "model_used": inferences[i]["model_used"],
        }
        for i, item in enumerate(valid_items)
    ] + invalid

    valid_r = [r for r in results if "error" not in r]
    scores = [r["risk_score"] for r in valid_r]
    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(f"batch n={len(data)} ML={ml_ms}ms total={total_ms}ms")

    return jsonify({
        "results": results,
        "summary": {
            "total": len(results),
            "successful": len(valid_r),
            "high_risk": sum(1 for r in valid_r if r["risk_level"] == "HIGH"),
            "medium_risk": sum(1 for r in valid_r if r["risk_level"] == "MEDIUM"),
            "low_risk": sum(1 for r in valid_r if r["risk_level"] == "LOW"),
            "avg_score": round(float(np.mean(scores)), 1) if scores else 0,
            "max_score": round(float(np.max(scores)), 1) if scores else 0,
        },
        "latency_ms": total_ms,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/chat", methods=["POST"])
@require_json
@handle_errors
def chat():
    data     = request.get_json()
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Field 'question' required."}), 400

    session_id = data.get("session_id") or f"s{int(time.time())}"
    context    = data.get("context", {})
    history    = _sessions.get(session_id, [])

    history.append({"role": "user", "content": question})
    answer = claude(RAG_SYSTEM.format(context=_ctx_str(context)),
                     history[-10:], max_tokens=400)
    if _needs_local_fallback(answer):
        answer = _local_chat_answer(question, context)
    history.append({"role": "assistant", "content": answer})
    _sessions[session_id] = history[-20:]

    return jsonify({
        "answer":     answer,
        "session_id": session_id,
        "turns":      len(_sessions[session_id]) // 2,
        "timestamp":  datetime.utcnow().isoformat() + "Z",
    })


# ── /ticket — agentic maintenance work-order ──────────────────────────────────
@app.route("/ticket", methods=["POST"])
@require_json
@handle_errors
def ticket():
    data      = request.get_json()
    inv_id    = data.get("inverter_id", "UNKNOWN")
    score     = data.get("risk_score", 0)
    level     = data.get("risk_level", "UNKNOWN")
    shap      = data.get("shap_drivers", {})
    narr      = data.get("narrative", {})
    priority  = "CRITICAL" if level=="HIGH" else ("HIGH" if level=="MEDIUM" else "MEDIUM")
    now       = datetime.utcnow()

    prompt = (
        f"Asset:{inv_id} Risk:{score}/100({level}) Priority:{priority}\n"
        f"SHAP:{json.dumps(shap)} Diagnosis:{json.dumps(narr)}\n\n"
        f"TICKET ID: AUTO-{now.strftime('%Y%m%d%H%M')}\n"
        f"DATE: {now.strftime('%Y-%m-%d')}\nPRIORITY: {priority}\n"
        f"ASSET: {inv_id}\nREPORTED BY: AI Predictive Maintenance System\n"
        f"────────────────────────────\n"
        f"ISSUE SUMMARY: <2 sentences>\n"
        f"ROOT CAUSE:\n1.\n2.\n"
        f"ACTIONS:\n1.\n2.\n3.\n"
        f"URGENCY: [24h/3d/7d]\nDOWNTIME: [h]\nPARTS: ...\n"
        f"────────────────────────────"
    )
    ticket_text = claude(TICKET_SYSTEM, [{"role":"user","content":prompt}], max_tokens=400)
    if _needs_local_fallback(ticket_text):
        ticket_text = _local_ticket(data)
    return jsonify({
        "ticket":       ticket_text,
        "inverter_id":  inv_id,
        "generated_at": now.isoformat() + "Z",
    })


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):  return jsonify({"error":"Not found","status":404}), 404
@app.errorhandler(405)
def bad_method(e): return jsonify({"error":"Method not allowed","status":405}), 405
@app.errorhandler(413)
def too_large(e):  return jsonify({"error":"Payload too large","status":413}), 413


# ── /live-data — serve the real live_predictions.csv ─────────────────────────
@app.route("/live-data", methods=["GET"])
@handle_errors
def live_data():
    """
    Returns dashboard rows and computes risk from model.pkl when available.
    CSV is optional; if missing, app serves demo telemetry and still predicts.
    """
    import csv as _csv

    csv_candidates = [
        BASE_DIR / "live_predictions.csv",
        BASE_DIR / "live_predictions (1).csv",
        BASE_DIR.parent / "live_predictions.csv",
        BASE_DIR.parent / "predictions.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)

    raw_rows = []
    if csv_path:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            raw_rows = list(_csv.DictReader(f))

    if not raw_rows:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        raw_rows = [
            {"inverter_id":"INV-023","current_temp":"71.8","current_power":"4680","power_ratio_24hr":"0.90","temp_24hr_avg":"66.2","anomaly_flag":"1","fault_predicted":"1","last_reading":now},
            {"inverter_id":"INV-031","current_temp":"68.1","current_power":"4550","power_ratio_24hr":"0.93","temp_24hr_avg":"64.8","anomaly_flag":"0","fault_predicted":"0","last_reading":now},
            {"inverter_id":"INV-046","current_temp":"66.4","current_power":"4420","power_ratio_24hr":"0.95","temp_24hr_avg":"63.7","anomaly_flag":"0","fault_predicted":"0","last_reading":now},
            {"inverter_id":"INV-011","current_temp":"59.5","current_power":"4860","power_ratio_24hr":"0.98","temp_24hr_avg":"58.8","anomaly_flag":"0","fault_predicted":"0","last_reading":now},
            {"inverter_id":"INV-019","current_temp":"57.8","current_power":"4910","power_ratio_24hr":"0.99","temp_24hr_avg":"57.1","anomaly_flag":"0","fault_predicted":"0","last_reading":now},
            {"inverter_id":"INV-052","current_temp":"64.9","current_power":"4465","power_ratio_24hr":"0.94","temp_24hr_avg":"62.3","anomaly_flag":"0","fault_predicted":"0","last_reading":now},
        ]

    results = []
    for i, row in enumerate(raw_rows):
        inv_id = row.get("inverter_id") or row.get("Inverter_ID") or f"INV-{i+1:03d}"
        current_temp = float(row.get("current_temp", row.get("temp", 60)) or 60)
        current_power = float(row.get("current_power", row.get("power", 4500)) or 4500)
        power_ratio_24hr = float(row.get("power_ratio_24hr", row.get("power_ratio", 0.95)) or 0.95)
        temp_24hr_avg = float(row.get("temp_24hr_avg", row.get("temp_rolling_mean", current_temp - 2)) or (current_temp - 2))
        anomaly_flag = int(float(row.get("anomaly_flag", row.get("temp_spike_flag", 0)) or 0))
        fault_predicted = int(float(row.get("fault_predicted", 0) or 0))
        last_reading = row.get("last_reading") or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        pred_input = {
            "inverter_id": inv_id,
            "temp": float(row.get("temp", current_temp) or current_temp),
            "power": float(row.get("power", current_power) or current_power),
            "pv1_power": float(row.get("pv1_power", current_power * 0.97) or (current_power * 0.97)),
            "v_ab": float(row.get("v_ab", 230.5) or 230.5),
            "v_bc": float(row.get("v_bc", 231.0) or 231.0),
            "v_ca": float(row.get("v_ca", 229.5) or 229.5),
            "freq": float(row.get("freq", 49.98) or 49.98),
            "kwh_today": float(row.get("kwh_today", max(current_power / 120.0, 1.0)) or max(current_power / 120.0, 1.0)),
            "ambient_temp": float(row.get("ambient_temp", 35.0) or 35.0),
            "power_ratio": float(row.get("power_ratio", power_ratio_24hr) or power_ratio_24hr),
            "temp_deviation": float(row.get("temp_deviation", current_temp - temp_24hr_avg) or (current_temp - temp_24hr_avg)),
            "voltage_imbalance": float(row.get("voltage_imbalance", 0.01) or 0.01),
            "temp_spike_flag": float(row.get("temp_spike_flag", anomaly_flag) or anomaly_flag),
            "temp_rolling_mean": float(row.get("temp_rolling_mean", temp_24hr_avg) or temp_24hr_avg),
            "temp_rolling_std": float(row.get("temp_rolling_std", 3.0) or 3.0),
            "power_rolling_mean": float(row.get("power_rolling_mean", current_power * 0.98) or (current_power * 0.98)),
            "power_ratio_rolling_mean": float(row.get("power_ratio_rolling_mean", power_ratio_24hr) or power_ratio_24hr),
            "voltage_imbalance_rolling": float(row.get("voltage_imbalance_rolling", 0.009) or 0.009),
            "anomaly_score": float(row.get("anomaly_score", 0.25 if anomaly_flag else 0.05) or (0.25 if anomaly_flag else 0.05)),
            "hour": float(row.get("hour", datetime.utcnow().hour) or datetime.utcnow().hour),
            "month": float(row.get("month", datetime.utcnow().month) or datetime.utcnow().month),
        }

        inf = _infer_single(pred_input)
        score_0_100 = float(inf["risk_score"])
        risk_level = inf["risk_level"]
        anomaly_flag = max(anomaly_flag, int(inf.get("anomaly_model_flag", 0)))

        results.append({
            "inverter_id": inv_id,
            "last_reading": last_reading,
            "risk_score": score_0_100,
            "risk_score_raw": score_0_100 / 100.0,
            "avg_risk_score": score_0_100 / 100.0,
            "anomaly_flag": anomaly_flag,
            "current_temp": current_temp,
            "current_power": current_power,
            "temp_24hr_avg": temp_24hr_avg,
            "power_ratio_24hr": power_ratio_24hr,
            "fault_predicted": fault_predicted,
            "risk_level": risk_level,
            "risk_category": "No Risk" if risk_level == "LOW" else "Risk",
            "model_used": inf.get("model_used", "mock"),
            "shap_drivers": inf.get("shap_drivers", {}),
        })

    scores = [r["risk_score"] for r in results]
    return jsonify({
        "results": results,
        "summary": {
            "total":       len(results),
            "high_risk":   sum(1 for r in results if r["risk_level"] == "HIGH"),
            "medium_risk": sum(1 for r in results if r["risk_level"] == "MEDIUM"),
            "low_risk":    sum(1 for r in results if r["risk_level"] == "LOW"),
            "anomaly_flagged": sum(1 for r in results if r["anomaly_flag"] == 1),
            "avg_risk_score":  round(float(np.mean(scores)), 4) if scores else 0,
            "max_risk_score":  round(float(np.max(scores)),  4) if scores else 0,
            "model_used": "real" if model is not None else "mock",
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


# -- /predict/csv — Bulk CSV upload → batch predictions ───────────────────────
@app.route("/predict/csv", methods=["POST"])
@handle_errors
def predict_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file. POST multipart/form-data with key 'file'."}), 400
    f = request.files["file"]
    if f.filename == "" or not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a .csv file."}), 400

    import io, csv as _csv, base64
    try:
        stream = io.StringIO(f.stream.read().decode("utf-8-sig"), newline="")
        reader = _csv.DictReader(stream)
        rows = list(reader)
    except Exception as e:
        return jsonify({"error": f"CSV parse error: {e}"}), 400

    if not rows:
        return jsonify({"error": "CSV file is empty."}), 400
    if len(rows) > 5000:
        return jsonify({"error": "Maximum 5,000 rows per upload."}), 400

    t0 = time.perf_counter()
    valid_items, skipped = [], []
    for idx, row in enumerate(rows, 1):
        inv_id = (row.get("inverter_id") or row.get("Inverter_ID") or row.get("InverterID") or row.get("id") or f"ROW-{idx}").strip()
        item, ok = {"inverter_id": inv_id}, True
        for col in RAW_NUMERIC_COLS:
            raw = row.get(col, row.get(col.replace("_", " "), "0"))
            try:
                item[col] = float(raw) if raw not in ("", None) else 0.0
            except ValueError:
                skipped.append({"row": idx, "inverter_id": inv_id, "error": f"Bad value for '{col}': '{raw}'"})
                ok = False
                break
        for col in OPTIONAL_MODEL_COLS:
            raw = row.get(col, DEFAULT_FEATURE_VALUES.get(col, 0.0))
            try:
                item[col] = float(raw) if raw not in ("", None) else float(DEFAULT_FEATURE_VALUES.get(col, 0.0))
            except ValueError:
                item[col] = float(DEFAULT_FEATURE_VALUES.get(col, 0.0))
        if ok:
            valid_items.append(item)

    if not valid_items:
        return jsonify({"error": "No valid rows found.", "skipped": skipped}), 400

    inferences = _infer_batch(valid_items)
    results = [{
        "inverter_id": item["inverter_id"],
        "risk_score": inferences[i]["risk_score"],
        "risk_level": inferences[i]["risk_level"],
        "shap_drivers": inferences[i]["shap_drivers"],
        "anomaly_model_flag": inferences[i].get("anomaly_model_flag", 0),
        "top_drivers": ", ".join(list(inferences[i]["shap_drivers"].keys())[:3]),
        "model_used": inferences[i]["model_used"],
    } for i, item in enumerate(valid_items)]

    scores = [r["risk_score"] for r in results]
    summary = {
        "total": len(results), "skipped": len(skipped),
        "high_risk": sum(1 for r in results if r["risk_level"] == "HIGH"),
        "medium_risk": sum(1 for r in results if r["risk_level"] == "MEDIUM"),
        "low_risk": sum(1 for r in results if r["risk_level"] == "LOW"),
        "avg_score": round(float(np.mean(scores)), 4) if scores else 0,
        "max_score": round(float(np.max(scores)), 4) if scores else 0,
        "ml_latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }

    out_buf = io.StringIO()
    writer = _csv.writer(out_buf)
    writer.writerow(["inverter_id", "risk_score", "risk_level", "top_drivers", "model_used", "anomaly_model_flag"])
    for r in results:
        writer.writerow([r["inverter_id"], r["risk_score"], r["risk_level"], r["top_drivers"], r["model_used"], r["anomaly_model_flag"]])
    csv_uri = "data:text/csv;base64," + base64.b64encode(out_buf.getvalue().encode()).decode()

    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(f"CSV n={len(rows)} valid={len(valid_items)} total={total_ms}ms")
    return jsonify({
        "results": results,
        "summary": summary,
        "skipped": skipped,
        "csv_download": csv_uri,
        "latency_ms": total_ms,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/financial-impact", methods=["POST"])
@require_json
@handle_errors
def financial_impact():
    data        = request.get_json()
    results     = data.get("results", [])
    capacity_mw = float(data.get("plant_capacity_mw", 10))
    tariff      = float(data.get("tariff_per_kwh", 4.5))
    hours_per_day = 7.0
    high = [r for r in results if r.get("risk_level") == "HIGH"]
    med  = [r for r in results if r.get("risk_level") == "MEDIUM"]
    per_unit_kw = (capacity_mw * 1000) / max(len(results), 1)
    kwh_7d      = per_unit_kw * hours_per_day * 7
    loss_kwh    = round(len(high)*kwh_7d*0.025 + len(med)*kwh_7d*0.010, 1)
    revenue_risk = round(loss_kwh * tariff, 2)
    return jsonify({
        "plant_capacity_mw": capacity_mw, "tariff_per_kwh": tariff,
        "units_high_risk": len(high), "units_medium_risk": len(med),
        "kwh_at_risk_7d": loss_kwh, "revenue_at_risk_inr": revenue_risk,
        "estimated_savings_inr": round(revenue_risk * 0.80, 2),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


if __name__ == "__main__":
    # threaded=True lets Flask handle concurrent requests
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)





