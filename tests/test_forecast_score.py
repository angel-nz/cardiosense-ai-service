"""NEW S2C — POST /forecast/score.

Contract + math tests over the REAL deployed Skorp artifact and the frozen
R4C engine (imported read-only). Synthetic payloads only.
"""
import itertools
import logging
import math
import os
import random

import numpy as np
import pytest

os.environ.setdefault("AI_SERVICE_KEY", "internal-dev-key")

from fastapi.testclient import TestClient

import app.main as M
from app.forecast import scoring as FS
from app.personalization import constants as C
from app.personalization import engine as E
from tests.test_personalization import PS, STABLE, XT, ev, hist, st, unit

H = {"X-Internal-Key": "internal-dev-key"}
client = TestClient(M.app)
client_nr = TestClient(M.app, raise_server_exceptions=False)
PRED = M._predictor
MV = "Skorp-Beta-0.2"


def fs(targets, adj=None, version=MV, c=client, **extra):
    body = {"expected_model_version": version, "targets": targets, **extra}
    if adj is not None:
        body["individualized_adjustment"] = {"applied_logit": adj}
    return c.post("/forecast/score", json=body, headers=H)


def rand_state(rng):
    sys_ = rng.uniform(95, 210)
    return dict(age=rng.randint(32, 81), sex=rng.randint(0, 1), currentSmoker=(s := rng.randint(0, 1)),
                cigsPerDay=rng.randint(1, 40) if s else 0, BPMeds=rng.randint(0, 1), diabetes=rng.randint(0, 1),
                totChol=round(rng.uniform(150, 350), 2), sysBP=round(sys_, 2), diaBP=round(rng.uniform(55, sys_ - 5), 2),
                BMI=round(rng.uniform(18, 42), 2), glucose=round(rng.uniform(65, 260), 2))


# ── FS1 / global equivalence ───────────────────────────────────────────────
def test_FS1_global_equals_existing_predict_path_unrounded(monkeypatch):
    """Capture the UNROUNDED probability each path takes from the classifier."""
    rng = random.Random(20261003)
    seen = []
    real = PRED.classifier.predict_proba
    monkeypatch.setattr(PRED.classifier, "predict_proba", lambda X: (r := real(X), seen.append(float(r[0, 1])))[0])
    for _ in range(60):
        x = rand_state(rng)
        seen.clear()
        p = client.post("/predict", json=x, headers=H).json()
        p_predict = min(max(seen[-1], 0.0), 1.0)
        seen.clear()
        s = fs([x]).json()["scores"][0]
        p_forecast = min(max(seen[-1], 0.0), 1.0)
        assert p_forecast.hex() == p_predict.hex()                     # bit-identical global probability
        assert s["globalRiskScore"] == p["risk_score"] == s["finalRiskScore"]
        assert s["riskLevel"] == p["risk_level"].upper() and s["interpretation"] == "GLOBAL"
        assert p_forecast == PRED.global_probability_unrounded(x)      # same helper R uses


def test_FS1_score_targets_global_is_identity():
    rng = random.Random(7)
    for _ in range(40):
        x = rand_state(rng)
        it = FS.score_targets(PRED.global_probability_unrounded, PRED.risk_level, [x], None)[0]
        assert it["_p_final"] is it["_p_global"] or it["_p_final"] == it["_p_global"]
        assert it["_p_global"] == PRED.global_probability_unrounded(x)


# ── FS2 batch order ───────────────────────────────────────────────────────
@pytest.mark.parametrize("n", [1, 2, 3])
def test_FS2_batches_preserve_order(n):
    rng = random.Random(n)
    xs = [rand_state(rng) for _ in range(n)]
    out = fs(xs).json()["scores"]
    assert [s["targetIndex"] for s in out] == list(range(n))
    for x, s in zip(xs, out):
        assert s["globalRiskScore"] == fs([x]).json()["scores"][0]["globalRiskScore"]


def test_FS2_zero_or_four_targets_rejected():
    assert fs([]).status_code == 422
    assert fs([XT] * 4).status_code == 422


# ── FS3 unrounded classification ─────────────────────────────────────────
def _force(monkeypatch, p):
    monkeypatch.setattr(PRED.classifier, "predict_proba", lambda X: np.array([[1.0 - p, p]]))


@pytest.mark.parametrize("p,score,level", [
    (0.199996, 0.2, "LOW"), (0.19999, 0.2, "LOW"), (0.2, 0.2, "MODERATE"),
    (0.349996, 0.35, "MODERATE"), (0.35, 0.35, "HIGH"), (math.nextafter(0.35, 0), 0.35, "MODERATE"),
    (math.nextafter(0.2, 0), 0.2, "LOW"), (0.0, 0.0, "LOW"), (1.0, 1.0, "HIGH"),
])
def test_FS3_level_from_unrounded_global(monkeypatch, p, score, level):
    _force(monkeypatch, p)
    s = fs([XT]).json()["scores"][0]
    assert (s["globalRiskScore"], s["finalRiskScore"], s["riskLevel"]) == (score, score, level)


def test_FS3_level_from_unrounded_final_with_adjustment(monkeypatch):
    # choose p_g so that the adjusted p_final lands just below 0.35 but rounds to 0.35
    target = 0.349996
    L = 0.25
    p_g = 1 / (1 + math.exp(-(math.log(target / (1 - target)) - L)))
    _force(monkeypatch, p_g)
    s = fs([XT], adj=L).json()["scores"][0]
    p_f = FS.final_probability(p_g, L)
    assert round(p_f, 4) == 0.35 and p_f < 0.35
    assert s["finalRiskScore"] == 0.35 and s["riskLevel"] == "MODERATE" and s["interpretation"] == "INDIVIDUALIZED_BRIDGE"


# ── FS4 – FS6 rejections ─────────────────────────────────────────────────
def test_FS4_model_version_mismatch_409():
    r = fs([XT], version="Skorp-Beta-0.1")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "MODEL_VERSION_MISMATCH"


@pytest.mark.parametrize("bad", [
    dict(XT, sysBP=320), dict(XT, age=17), dict(XT, sex=2), dict(XT, heartRate=78.5),
    {k: v for k, v in XT.items() if k != "glucose"}, dict(XT, patientId="p-1"),
])
def test_FS5_invalid_target_422(bad):
    assert fs([bad]).status_code == 422


def test_FS5_cross_field_dia_ge_sys_422():
    r = fs([dict(XT, sysBP=120, diaBP=120)])
    assert r.status_code == 422 and r.json()["detail"]["code"] == "TARGET_INVALID"


@pytest.mark.parametrize("raw", ['NaN', 'Infinity', '-Infinity'])
def test_FS6_non_finite_adjustment_422(raw):
    body = '{"expected_model_version":"%s","targets":[%s],"individualized_adjustment":{"applied_logit":%s}}' % (
        MV, __import__("json").dumps(XT), raw)
    r = client.post("/forecast/score", content=body, headers=dict(H, **{"content-type": "application/json"}))
    assert r.status_code == 422


def test_FS6_adjustment_rejects_extra_keys_and_evidence():
    assert fs([XT], individualized_adjustment={"applied_logit": 0.1, "gamma": 0.5}).status_code == 422
    assert fs([XT], personalization=ev(hist(STABLE))).status_code == 422


def test_auth_and_model_not_loaded(monkeypatch):
    body = {"expected_model_version": MV, "targets": [XT]}
    assert client.post("/forecast/score", json=body).status_code == 401
    assert client.post("/forecast/score", json=body, headers={"X-Internal-Key": "x"}).status_code == 403
    monkeypatch.setattr(M, "_predictor", None)
    assert client_nr.post("/forecast/score", json=body, headers=H).status_code == 503


def test_non_finite_global_is_500_not_a_level(monkeypatch):
    monkeypatch.setattr(PRED.classifier, "predict_proba", lambda X: np.array([[math.nan, math.nan]]))
    r = client_nr.post("/forecast/score", json={"expected_model_version": MV, "targets": [XT]}, headers=H)
    assert r.status_code == 500 and b"HIGH" not in r.content and b"LOW" not in r.content


def test_response_has_no_forbidden_fields():
    r = fs([XT], adj=-0.1).json()
    assert set(r) == {"model_version", "scores"}
    assert set(r["scores"][0]) == {"targetIndex", "globalRiskScore", "finalRiskScore", "riskLevel", "interpretation"}


def test_logging_is_aggregate_only(caplog):
    with caplog.at_level(logging.DEBUG, logger="skorp.forecast"):
        fs([dict(XT, sysBP=171.37)], adj=-0.123456789)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "targets=1" in text and "171.37" not in text and "0.123456789" not in text


# ── INV1: bridge reproduces the REAL R individualized probability ───────
def _individualized_cases():
    """Real R4C runs with varied histories → INDIVIDUALIZED results incl. positive,
    negative, zero, bounded and unbounded applied_logit."""
    cases = []
    lows = (130.0, 80.0, 200.0, 90.0)
    for cur, hist_vals in [
        (XT, STABLE),                                                       # bounded negative (−0.30)
        (dict(XT, sysBP=130.0, diaBP=80.0, totChol=200.0, glucose=90.0),
         [(165, 95, 240, 110), (166, 96, 241, 111), (164, 94, 239, 109), (165, 95, 240, 110)]),  # positive
        (XT, [(140, 88, 225, 100), (175, 100, 260, 140), (132, 82, 205, 90), (170, 99, 255, 135)]),  # γ = 0
    ] + [
        (dict(XT, sysBP=165.0 + d, glucose=110.0 + d / 2), [(165 + d + e, 95, 240, 110 + e) for e in (-2, 1, -1, 2)])
        for d in (-6, -3, 3, 6)                                              # small / unbounded
    ]:
        res = E.personalize(cur, ev(hist(hist_vals)), PRED.global_probability_unrounded, PRED.risk_thresholds)
        if res.status == C.INDIVIDUALIZED:
            cases.append((cur, hist_vals, res))
    return cases


def test_INV1_anchor_plus_applied_logit_reproduces_R_bit_exactly():
    cases = _individualized_cases()
    signs = {(-1 if r.applied_logit < 0 else 1 if r.applied_logit > 0 else 0) for _, _, r in cases}
    assert {-1, 0, 1} <= signs, signs
    assert any(r.bounded for _, _, r in cases) and any(not r.bounded and r.applied_logit != 0 for _, _, r in cases)
    for cur, _, r in cases:
        it = FS.score_targets(PRED.global_probability_unrounded, PRED.risk_level, [cur], r.applied_logit)[0]
        assert it["_p_global"].hex() == float(r.global_risk_score).hex()
        assert it["_p_final"].hex() == float(r.final_risk_score).hex()          # bit-exact
        assert it["riskLevel"].lower() == r.final_risk_level


def test_INV1_over_http_wire_applied_logit_reproduces_predict_block():
    """applied_logit as it travels in the REAL /predict JSON (what the backend
    captures) is bit-identical to the engine value and reproduces the block."""
    for cur, hv, r in _individualized_cases():
        block = client.post("/predict", json=dict(cur, personalization=ev(hist(hv))), headers=H).json()["personalization"]
        assert block["status"] == "INDIVIDUALIZED"
        assert float(block["applied_logit"]).hex() == float(r.applied_logit).hex()
        s = fs([cur], adj=block["applied_logit"]).json()["scores"][0]
        assert (s["finalRiskScore"], s["riskLevel"].lower()) == (block["final_risk_score"], block["final_risk_level"])


@pytest.mark.parametrize("shift", [-12.0, -9.0, 9.0, 12.0])   # p_g pushed to the logit clamp edges
def test_INV1_near_clamp_edges(shift):
    def evaluate(x):
        p = PRED.global_probability_unrounded(x)
        return E.sigmoid(math.log(p / (1 - p)) + shift)
    res = E.personalize(XT, ev(hist(STABLE)), evaluate, PRED.risk_thresholds)
    if res.status != C.INDIVIDUALIZED:
        pytest.skip(f"engine not individualized at shift {shift}: {res.reason}")
    it = FS.score_targets(evaluate, PRED.risk_level, [XT], res.applied_logit)[0]
    assert it["_p_final"].hex() == float(res.final_risk_score).hex()


@pytest.mark.parametrize("p_g", [1e-9, 0.00005, math.nextafter(0.00005, 1), 0.2, 0.35, 0.99995, 1 - 1e-9, 0.0, 1.0])
@pytest.mark.parametrize("L", [-0.30, -0.0001, 0.0, 0.0001, 0.30])
def test_INV1_combination_identity_against_engine_formula(p_g, L):
    # mirror of engine.personalize's final step, line for line
    expected = p_g if L == 0.0 else E.sigmoid(E.clamped_logit(p_g) + L)
    assert FS.final_probability(p_g, L).hex() == expected.hex()


def test_INV1_zero_adjustment_returns_global_object_exactly():
    p = 0.3123456789
    assert FS.final_probability(p, 0.0) is p and FS.final_probability(p, -0.0) is p


# ── INV2: simulated targets never enter personalize() ─────────────────────
def test_INV2_forecast_never_calls_personalize(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("personalize() must not be called by forecast scoring")
    monkeypatch.setattr(E, "personalize", boom)
    import app.personalization.wire as W
    monkeypatch.setattr(W, "personalize", boom)
    assert fs([XT, dict(XT, age=59)], adj=-0.2).status_code == 200
    import ast
    for f in (FS.__file__, FS.__file__.replace("scoring.py", "router.py")):
        tree = ast.parse(open(f).read())
        called = {getattr(n.func, "id", getattr(n.func, "attr", None)) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        assert "personalize" not in called and "personalize" not in imported and "personalization_block" not in imported


def test_predict_route_unchanged_by_forecast_registration():
    r = client.post("/predict", json=XT, headers=H)
    assert r.status_code == 200 and "personalization" not in r.json()
    paths = M.app.openapi()["paths"]
    assert "/forecast/score" in paths and set(paths["/forecast/score"]) == {"post"} and "/predict" in paths
