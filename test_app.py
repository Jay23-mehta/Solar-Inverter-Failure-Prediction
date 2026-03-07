import pytest
import json
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ── Valid payload used across tests ──────────────────────────────────────────
VALID_PAYLOAD = {
    "inverter_id":        "INV-042",
    "dc_voltage":         380.5,
    "ac_power_output":    4800.0,
    "temperature":        68.0,
    "alarm_code_encoded": 3,
    "rolling_mean_7d":    4900.0,
    "rolling_std_7d":     120.0,
    "voltage_drop_ratio": 0.18,
}


# ── Test 1: Valid input → 200 + correct fields in response ───────────────────
def test_predict_valid_input(client):
    response = client.post(
        "/predict",
        data=json.dumps(VALID_PAYLOAD),
        content_type="application/json",
    )
    assert response.status_code == 200

    body = response.get_json()
    assert "risk_score"  in body
    assert "risk_level"  in body
    assert "inverter_id" in body
    assert body["inverter_id"] == "INV-042"
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert 0 <= body["risk_score"] <= 100


# ── Test 2: Missing field → 400 + error message ───────────────────────────────
def test_predict_missing_field(client):
    payload = VALID_PAYLOAD.copy()
    del payload["temperature"]          # remove one required field

    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400

    body = response.get_json()
    assert "error"   in body
    assert "details" in body
    # Check the error message mentions the missing field
    assert any("temperature" in detail for detail in body["details"])


# ── Test 3: Wrong data type → 400 + error message ────────────────────────────
def test_predict_wrong_type(client):
    payload = VALID_PAYLOAD.copy()
    payload["dc_voltage"] = "not-a-number"   # string instead of float

    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400

    body = response.get_json()
    assert "error"   in body
    assert "details" in body
    assert any("dc_voltage" in detail for detail in body["details"])


# ── Test 4: Empty body → 400 ──────────────────────────────────────────────────
def test_predict_empty_body(client):
    response = client.post(
        "/predict",
        data="",
        content_type="application/json",
    )
    assert response.status_code == 400


# ── Test 5: Health check → 200 + status ok ───────────────────────────────────
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.get_json()
    assert body["status"] == "ok"
    assert "model" in body
