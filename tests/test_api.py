"""AI Service test suite — T1 through T11 from Bloque C2."""
import os

os.environ.setdefault("AI_SERVICE_KEY", "internal-dev-key")

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "age": 55, "sex": 1, "currentSmoker": 0, "cigsPerDay": 0, "BPMeds": 1,
    "diabetes": 0, "totChol": 230.5, "sysBP": 145.0, "diaBP": 92.0,
    "BMI": 28.4, "glucose": 105.0,
}
HEADERS = {"X-Internal-Key": "internal-dev-key"}


def test_t1_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_t2_valid_key_valid_prediction():
    r = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    for field in ("risk_score", "risk_level", "anomaly_score", "is_anomaly",
                  "feature_importance", "model_version"):
        assert field in body


def test_t3_missing_key():
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code in (401, 403)


def test_t4_wrong_key():
    r = client.post("/predict", json=VALID_PAYLOAD, headers={"X-Internal-Key": "wrong"})
    assert r.status_code in (401, 403)


def test_t5_invalid_payload():
    r = client.post("/predict", json={"age": "not-a-number", "sex": 1}, headers=HEADERS)
    assert 400 <= r.status_code < 500


def test_t6_risk_score_bounds():
    r = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
    score = r.json()["risk_score"]
    assert 0.0 <= score <= 1.0


def test_t7_risk_level_boundaries():
    from app.inference import SkorpPredictor
    p = SkorpPredictor()
    assert p.risk_level(0.0) == "low"
    assert p.risk_level(0.19) == "low"
    assert p.risk_level(0.20) == "moderate"
    assert p.risk_level(0.3499) == "moderate"
    assert p.risk_level(0.35) == "high"
    assert p.risk_level(1.0) == "high"


def test_t8_anomaly_consistency():
    r = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
    body = r.json()
    if body["anomaly_score"] < 0:
        assert body["is_anomaly"] is True
    else:
        assert body["is_anomaly"] is False


def test_t9_feature_importance_shape():
    r = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
    fi = r.json()["feature_importance"]
    expected = [
        "male", "age", "currentSmoker", "cigsPerDay", "BPMeds", "diabetes",
        "totChol", "sysBP", "diaBP", "BMI", "glucose",
    ]
    assert list(fi) == expected
    assert len(fi) == 11
    assert fi["currentSmoker"] == 0.0
    assert fi["BPMeds"] == 0.0
    assert fi["totChol"] == 0.0
    assert abs(sum(fi.values()) - 100.0) < 0.05


def test_t10_model_version():
    r = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS)
    assert r.json()["model_version"] == "Skorp-Beta-0.2"


def test_t11_determinism():
    r1 = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS).json()
    r2 = client.post("/predict", json=VALID_PAYLOAD, headers=HEADERS).json()
    assert r1["risk_score"] == r2["risk_score"]
    assert r1["anomaly_score"] == r2["anomaly_score"]


def test_t12_health_exposes_age_eligibility():
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["model_version"] == "Skorp-Beta-0.2"
    assert body["eligible_age_range"] == {"min": 32, "max": 81}
    assert body["training_age_range"] == {"min": 32, "max": 81}


def test_t13_sanity_range_is_independent_of_eligibility():
    # The AI keeps its 18-120 input sanity validation; model eligibility (32-81)
    # is enforced by the backend before calling /predict, not by this schema.
    for age, expected in ((25, 200), (80, 200), (17, 422), (121, 422)):
        r = client.post("/predict", json={**VALID_PAYLOAD, "age": age}, headers=HEADERS)
        assert r.status_code == expected, (age, r.status_code)


def test_t14_obsolete_heart_rate_is_rejected():
    payload = {**VALID_PAYLOAD, "heartRate": 78}
    r = client.post("/predict", json=payload, headers=HEADERS)
    assert r.status_code == 422


def test_t15_model_age_boundary_contract():
    support = client.get("/health").json()["eligible_age_range"]
    lo, hi = support["min"], support["max"]
    expected = {31: False, 32: True, 70: True, 71: True, 81: True, 82: False}
    assert {age: lo <= age <= hi for age in expected} == expected
