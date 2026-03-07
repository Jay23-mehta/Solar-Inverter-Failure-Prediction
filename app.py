from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Allows Person 3 (Frontend) to call this API from browser

# ── Load model ONCE at startup ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
else:
    print("⚠️  model.pkl not found — using mock mode (returns dummy score)")


# ── Required telemetry fields ────────────────────────────────────────────────
REQUIRED_FIELDS = {
    "inverter_id":        str,
    "dc_voltage":         (int, float),
    "ac_power_output":    (int, float),
    "temperature":        (int, float),
    "alarm_code_encoded": (int, float),
    "rolling_mean_7d":    (int, float),
    "rolling_std_7d":     (int, float),
    "voltage_drop_ratio": (int, float),
}


# ── /predict ─────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON with inverter_id + telemetry features.
    Returns risk_score (0–100) + risk_level label.
    
    Example request body:
    {
        "inverter_id":        "INV-042",
        "dc_voltage":         380.5,
        "ac_power_output":    4800.0,
        "temperature":        68.0,
        "alarm_code_encoded": 3,
        "rolling_mean_7d":    4900.0,
        "rolling_std_7d":     120.0,
        "voltage_drop_ratio": 0.18
    }
    """
    data = request.get_json(silent=True)

    # ── 1. Check Content-Type / body exists ──────────────────────────────────
    if data is None:
        return jsonify({
            "error": "Request body must be valid JSON with Content-Type: application/json"
        }), 400

    # ── 2. Validate required fields & types ──────────────────────────────────
    errors = []
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in data:
            errors.append(f"Missing field: '{field}'")
        elif not isinstance(data[field], expected_type):
            errors.append(
                f"Wrong type for '{field}': expected {expected_type}, got {type(data[field]).__name__}"
            )

    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    # ── 3. Run inference (or mock if model not loaded yet) ───────────────────
    feature_cols = [
        "dc_voltage", "ac_power_output", "temperature",
        "alarm_code_encoded", "rolling_mean_7d",
        "rolling_std_7d", "voltage_drop_ratio",
    ]

    if model:
        df = pd.DataFrame([{col: data[col] for col in feature_cols}])
        prob = model.predict_proba(df)[0][1]   # probability of failure
        risk_score = round(prob * 100, 1)
    else:
        # ── MOCK: remove this block once Person 1 gives you model.pkl ────────
        risk_score = round(
            min(100, data["temperature"] * 0.8 + data["voltage_drop_ratio"] * 50), 1
        )

    # ── 4. Label ─────────────────────────────────────────────────────────────
    if risk_score >= 75:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return jsonify({
        "inverter_id": data["inverter_id"],
        "risk_score":  risk_score,          # 0–100
        "risk_level":  risk_level,          # LOW / MEDIUM / HIGH
        "model_used":  "real" if model else "mock",
    }), 200


# ── /health ──────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model":  "loaded" if model else "mock"
    }), 200


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)

@app.route("/inverters", methods=["GET"])
def get_all_inverters():
    """
    Returns risk score + level for all inverters.
    Frontend calls this once to load the full map.
    """
    # Replace this with real model predictions once model.pkl is ready
    inverters = [
        {"inverter_id": "INV-001", "zone": "A", "risk_score": 32.0, "risk_level": "LOW"},
        {"inverter_id": "INV-002", "zone": "A", "risk_score": 61.5, "risk_level": "MEDIUM"},
        {"inverter_id": "INV-003", "zone": "A", "risk_score": 84.2, "risk_level": "HIGH"},
        # ... add all your inverters here
    ]
    return jsonify({"inverters": inverters}), 200