"""NEW R (R4D) — POST /predict optional personalization integration tests.

Endpoint-level contract tests over the real deployed Skorp artifact and the
frozen R4C engine. NOT clinical-accuracy tests. All payloads are synthetic.
"""
import json
import math
import os

import pytest

os.environ.setdefault("AI_SERVICE_KEY", "internal-dev-key")

from fastapi.testclient import TestClient

import app.main as M
from app.personalization import constants as C
from app.personalization import engine as E
from app.personalization import wire as W
from tests.test_personalization import PS, STABLE, XT, counts, ev, hist, st, unit

H = {"X-Internal-Key": "internal-dev-key"}
client = TestClient(M.app)
client_nr = TestClient(M.app, raise_server_exceptions=False)

STATUSES = {"INDIVIDUALIZED", "GLOBAL_INSUFFICIENT_HISTORY",
            "GLOBAL_INCOMPATIBLE_HISTORY", "GLOBAL_ERROR_FALLBACK"}
TOP_KEYS = ["risk_score", "risk_level", "anomaly_score", "is_anomaly",
            "feature_importance", "model_version"]
BLOCK_KEYS = ["version", "status", "reason", "final_risk_score", "final_risk_level",
              "Q", "gamma", "delta_logit", "raw_adjustment", "applied_logit", "bounded",
              "bp_reverted", "sigma_exp", "effective_clinical_units", "pi_units",
              "comparable_transitions", "excluded_transitions", "per_feature"]
ERRATIC = [(140, 88, 225, 100), (175, 100, 260, 140), (132, 82, 205, 90), (170, 99, 255, 135)]


def r4b_counts(raw_records):
    """Exactly the R4B EvidenceCounts key sets (wire = toPersonalizationRequest)."""
    return {
        "clinical": {"rawRecords": raw_records, "droppedOutOfScope": 0, "excludedUnmeasuredClinicalTime": 0, "excludedInvalid": 0,
                     "excludedOutOfDomain": 0, "excludedAmbiguousTimestamp": 0,
                     "excludedDuplicateOfAnchor": 0, "sameTimestampCollapsed": 0,
                     "burstCollapsedDuplicates": 0, "effectiveUnits": raw_records},
        "prediction": {"candidateRows": raw_records, "droppedOutOfScope": 0,
                       "excludedRecordNotEffective": 0, "excludedVersion": 0,
                       "excludedMissingGlobalScore": 0, "ambiguousGroups": 0,
                       "canonicalUnits": raw_records},
    }


def r4b_ev(units):
    return {"version": "R-BPBC-3", "raw_counts": r4b_counts(len(units)), "units": units}


def post(body, c=client, **kw):
    return c.post("/predict", json=body, headers=H, **kw)


def legacy():
    r = post(XT)
    assert r.status_code == 200
    return r


def mode_b(block, current=XT):
    r = post(dict(current, personalization=block))
    assert r.status_code == 200, r.text
    body = r.json()
    assert "personalization" in body
    return body


def assert_block_shape(b):
    assert list(b) == BLOCK_KEYS
    assert b["version"] == "R-BPBC-3" and b["status"] in STATUSES


def assert_global_fallback(body, status, reason):
    """Any non-INDIVIDUALIZED block == the GLOBAL result of the same response."""
    p = body["personalization"]
    assert_block_shape(p)
    assert (p["status"], p["reason"]) == (status, reason)
    assert p["final_risk_score"] == body["risk_score"]
    assert p["final_risk_level"] == body["risk_level"]
    for k in ("Q", "gamma", "delta_logit", "raw_adjustment", "applied_logit",
              "bounded", "bp_reverted", "sigma_exp", "per_feature"):
        assert p[k] is None


def top(body):
    return {k: body[k] for k in TOP_KEYS}


# ═══ Endpoint matrix A–J ══════════════════════════════════════════════════════
def test_A_no_block_is_legacy_and_key_absent():
    r = legacy()
    body = r.json()
    assert list(body) == TOP_KEYS                                    # key ABSENT, not null
    assert b'"personalization"' not in r.content
    # PRE-T-B makes the 11-feature global contract strict: unknown or obsolete
    # top-level fields are rejected rather than silently ignored.
    r2 = post(dict(XT, someOtherKey=1))
    assert r2.status_code == 422


def test_B_valid_individualized_r4b_wire_shape():
    body = mode_b(r4b_ev(hist(STABLE)))
    p = body["personalization"]
    assert_block_shape(p)
    assert p["status"] == "INDIVIDUALIZED" and p["reason"] is None
    assert top(body) == legacy().json()                              # top-level stays GLOBAL
    assert p["final_risk_score"] == round(p["final_risk_score"], 4)
    assert p["final_risk_score"] != body["risk_score"]
    assert p["bounded"] is True and p["applied_logit"] == -0.30
    assert (p["effective_clinical_units"], p["pi_units"], p["comparable_transitions"]) == (4, 4, 3)
    assert p["excluded_transitions"] == {}
    assert set(p["per_feature"]) == set(C.CONDITIONED_FEATURES)
    for d in p["per_feature"].values():
        assert list(d) == ["z", "s", "K_anchor", "K_anchor_prime"]  # no μ, P, x̃
    assert "global_risk_score" not in p and "conditioned_risk_score" not in p
    assert "gradients" not in p
    assert body["model_version"] == "Skorp-Beta-0.2" and p["version"] == "R-BPBC-3"


def test_B_engine_result_matches_frozen_r4c():
    block = r4b_ev(hist(STABLE))
    res = E.personalize(XT, block, M._predictor.global_probability_unrounded,
                        M._predictor.risk_thresholds)
    p = mode_b(block)["personalization"]
    assert p["final_risk_score"] == round(res.final_risk_score, 4)
    assert p["final_risk_level"] == res.final_risk_level
    for k in ("Q", "gamma", "delta_logit", "raw_adjustment", "applied_logit", "sigma_exp"):
        assert p[k] == getattr(res, k)


def test_C_gamma_zero_is_individualized_and_exactly_global():
    body = mode_b(r4b_ev(hist(ERRATIC)))
    p = body["personalization"]
    assert p["status"] == "INDIVIDUALIZED" and p["reason"] is None
    assert p["gamma"] == 0.0 and p["applied_logit"] == 0.0 and p["Q"] >= 4
    assert p["final_risk_score"] == body["risk_score"]
    assert p["final_risk_level"] == body["risk_level"]


def test_D_insufficient_history():
    body = mode_b(r4b_ev([]) | {"raw_counts": r4b_counts(0)})
    assert_global_fallback(body, "GLOBAL_INSUFFICIENT_HISTORY", "NO_CLINICAL_HISTORY")
    assert body["personalization"]["effective_clinical_units"] == 0
    body2 = mode_b(r4b_ev(hist(STABLE[:2], ts=(-14, -7))))
    assert_global_fallback(body2, "GLOBAL_INSUFFICIENT_HISTORY", "PREDICTION_HISTORY_INSUFFICIENT")
    assert body2["personalization"]["pi_units"] == 2


def test_E_incompatible_history():
    body = mode_b(r4b_ev(hist(STABLE, ps=dict(PS, sexUsed=None))))
    assert_global_fallback(body, "GLOBAL_INCOMPATIBLE_HISTORY", "SEX_PROVENANCE_UNKNOWN")
    assert body["personalization"]["excluded_transitions"] == {"SEX_PROVENANCE_UNKNOWN": 3}
    excluded = r4b_ev([])
    excluded["raw_counts"]["clinical"].update(rawRecords=3, excludedInvalid=3, effectiveUnits=0)
    assert_global_fallback(mode_b(excluded), "GLOBAL_INCOMPATIBLE_HISTORY", "CLINICAL_HISTORY_EXCLUDED")


@pytest.mark.parametrize("version", ["R-BPBC-1", "R-BPBC-2", "r-bpbc-1", "", None, 1, ["R-BPBC-3"]])
def test_F_unsupported_version(version):
    block = r4b_ev(hist(STABLE)) | {"version": version}
    assert_global_fallback(mode_b(block), "GLOBAL_ERROR_FALLBACK", "UNSUPPORTED_VERSION")
    missing = r4b_ev(hist(STABLE))
    del missing["version"]
    assert_global_fallback(mode_b(missing), "GLOBAL_ERROR_FALLBACK", "UNSUPPORTED_VERSION")


@pytest.mark.parametrize("block,reason", [
    (None, "INVALID_PAYLOAD"),                                   # explicit null = NEW-R, invalid block
    ("R-BPBC-3", "INVALID_PAYLOAD"),
    (42, "INVALID_PAYLOAD"),
    ([], "INVALID_PAYLOAD"),
    ({"version": "R-BPBC-3"}, "INVALID_PAYLOAD"),                # raw_counts missing
    ({"version": "R-BPBC-3", "raw_counts": r4b_counts(0), "units": "x"}, "INVALID_PAYLOAD"),
    ({"version": "R-BPBC-3", "raw_counts": r4b_counts(1),
      "units": [{"t_days": -1, "states": []}]}, "INVALID_PAYLOAD"),
    ({"version": "R-BPBC-3", "raw_counts": r4b_counts(1),
      "units": [unit(-1, [st(-1, 140, 88, 225, "high")])]}, "INVALID_PAYLOAD"),
    ({"version": "R-BPBC-3", "raw_counts": r4b_counts(1),
      "units": [unit(-1, [st(-1, 140, 88, 225, 100)], pi=1.5, ps=dict(PS))]}, "INVALID_PAYLOAD"),
    ({"version": "R-BPBC-3", "raw_counts": r4b_counts(2),
      "units": [unit(-1, [st(-1, 140, 88, 225, 100)]), unit(-5, [st(-5, 140, 88, 225, 100)])]},
     "INVALID_CHRONOLOGY"),                                      # never reordered
    ({"version": "R-BPBC-3", "raw_counts": r4b_counts(1),
      "units": [unit(0, [st(0, 140, 88, 225, 100)])]}, "INVALID_CHRONOLOGY"),
])
def test_G_malformed_block_is_200_global_fallback(block, reason):
    assert_global_fallback(mode_b(block), "GLOBAL_ERROR_FALLBACK", reason)


def test_G_non_finite_json_in_block_falls_back():
    raw = json.dumps(dict(XT, personalization=r4b_ev(hist(STABLE))))
    raw = raw.replace('"sysBP": 140', '"sysBP": NaN', 1)
    assert "NaN" in raw
    r = client.post("/predict", content=raw, headers=dict(H, **{"Content-Type": "application/json"}))
    assert r.status_code == 200
    assert_global_fallback(r.json(), "GLOBAL_ERROR_FALLBACK", "INVALID_PAYLOAD")
    assert b"NaN" not in r.content and b"Infinity" not in r.content


def test_H_gradient_error_falls_back(monkeypatch):
    monkeypatch.setitem(E.MODEL_INPUT_RANGES, "BMI", (27.95, 28.05))   # neither BMI ± 0.1 valid
    body = mode_b(r4b_ev(hist(STABLE)))
    assert_global_fallback(body, "GLOBAL_ERROR_FALLBACK", "GRADIENT_UNDEFINED")
    assert top(body) == legacy().json()


@pytest.mark.parametrize("bad", [
    {"age": "not-a-number"}, {"age": 17}, {"sysBP": 301.0}, {"glucose": None}, {"sex": 2},
])
def test_I_invalid_global_input_still_422_with_block(bad):
    for block in (r4b_ev(hist(STABLE)), "garbage", None):
        r = post(dict(XT, **bad, personalization=block))
        assert r.status_code == 422
        r0 = post(dict(XT, **bad))
        assert r0.status_code == 422
    missing = dict(XT, personalization=r4b_ev(hist(STABLE)))
    del missing["age"]
    assert post(missing).status_code == 422


def test_I_cross_field_invalid_current_is_global_plus_fallback():
    cur = dict(XT, sysBP=90.0, diaBP=95.0)                        # passes PredictRequest, dia ≥ sys
    legacy_r = post(cur)
    assert legacy_r.status_code == 200 and "personalization" not in legacy_r.json()
    body = mode_b(r4b_ev(hist(STABLE)), current=cur)
    assert top(body) == legacy_r.json()
    assert_global_fallback(body, "GLOBAL_ERROR_FALLBACK", "CURRENT_INPUT_INVALID")


def test_J_forced_global_failure_is_not_a_fallback(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("forced global failure")
    monkeypatch.setattr(M._predictor.classifier, "predict_proba", boom)
    for body in (XT, dict(XT, personalization=r4b_ev(hist(STABLE)))):
        r = post(body, c=client_nr)
        assert r.status_code == 500
        assert b"GLOBAL_ERROR_FALLBACK" not in r.content


# ═══ R4D-FIX1 — PHASE B failures preserve the PHASE A global result ════════
def _phase_b_fallback(body_bytes_or_resp, reason):
    r = body_bytes_or_resp
    assert r.status_code == 200
    assert b"NaN" not in r.content and b"Infinity" not in r.content
    body = r.json()
    assert top(body) == legacy().json()                          # exactly the Phase-A result
    assert_global_fallback(body, "GLOBAL_ERROR_FALLBACK", reason)
    return body


@pytest.mark.parametrize("bad_eval", [
    lambda x: (_ for _ in ()).throw(RuntimeError("engine re-eval failed")),   # raises
    lambda x: float("nan"),                                                   # invalid probability
    lambda x: 1.5,                                                            # out of range
])
def test_FIX1_r4c_global_inference_error_after_phase_a_is_fallback(monkeypatch, bad_eval):
    # Phase A (_predictor.predict → classifier) still succeeds; only the R4C
    # engine's internal global evaluation fails → GlobalInferenceError in R4C.
    with pytest.raises(E.GlobalInferenceError):
        E.personalize(XT, r4b_ev(hist(STABLE)), bad_eval, M._predictor.risk_thresholds)
    monkeypatch.setattr(M._predictor, "global_probability_unrounded", bad_eval)
    r = post(dict(XT, personalization=r4b_ev(hist(STABLE))), c=client_nr)
    _phase_b_fallback(r, "MODEL_EVALUATION_FAILED")
    assert post(XT).status_code == 200                           # legacy path unaffected


def test_FIX1_global_inference_error_raised_directly_by_engine(monkeypatch):
    def gie(*a, **k):
        raise E.GlobalInferenceError("forced inside personalization")
    monkeypatch.setattr(W, "personalize", gie)
    _phase_b_fallback(post(dict(XT, personalization=r4b_ev(hist(STABLE))), c=client_nr),
                      "MODEL_EVALUATION_FAILED")


@pytest.mark.parametrize("target", ["personalize", "to_block", "precheck"])
def test_FIX1_unexpected_personalization_exception_is_fallback(monkeypatch, target):
    def boom(*a, **k):
        raise KeyError("unexpected")
    monkeypatch.setattr(W, target, boom)
    _phase_b_fallback(post(dict(XT, personalization=r4b_ev(hist(STABLE))), c=client_nr),
                      "UNEXPECTED_ERROR")


def test_FIX1_boundary_log_is_safe_metadata_only(monkeypatch, caplog):
    def gie(*a, **k):
        raise E.GlobalInferenceError("SECRET-DETAIL 165.0 95.0")
    monkeypatch.setattr(W, "personalize", gie)
    with caplog.at_level("INFO", logger="skorp.personalization"):
        post(dict(XT, personalization=r4b_ev(hist(STABLE))))
    msgs = [rec.getMessage() for rec in caplog.records if rec.name == "skorp.personalization"]
    assert len(msgs) == 1
    m = msgs[0]
    assert "status=GLOBAL_ERROR_FALLBACK" in m and "reason=MODEL_EVALUATION_FAILED" in m
    assert "units=4" in m and "duration_ms=" in m and "error_type=GlobalInferenceError" in m
    assert "SECRET-DETAIL" not in m and "165" not in m and "0.2888" not in m


def test_FIX1_route_errors_outside_boundary_are_not_hidden(monkeypatch):
    # The boundary protects only the personalization attempt: a failure of
    # the endpoint itself (here: response construction) still surfaces as 500.
    monkeypatch.setattr(M, "PredictResponse", lambda **kw: (_ for _ in ()).throw(RuntimeError("route")))
    r = post(dict(XT, personalization=r4b_ev(hist(STABLE))), c=client_nr)
    assert r.status_code == 500 and b"GLOBAL_ERROR_FALLBACK" not in r.content


def test_J_model_not_loaded_is_503_in_both_modes(monkeypatch):
    monkeypatch.setattr(M, "_predictor", None)
    assert post(XT).status_code == 503
    assert post(dict(XT, personalization=r4b_ev(hist(STABLE)))).status_code == 503


# ═══ strict unknown-key policy (FORBID) ═══════════════════════════════════════
def _with(path_setter):
    block = r4b_ev(hist(STABLE))
    path_setter(block)
    return block


@pytest.mark.parametrize("name,mutate", [
    ("top patientId", lambda b: b.update(patientId="p-1")),
    ("top doctorId", lambda b: b.update(doctorId="d-1")),
    ("top evidenceFingerprint", lambda b: b.update(evidenceFingerprint="ab" * 32)),
    ("top anchor", lambda b: b.update(anchor={"recordId": "r"})),
    ("raw_counts extra group", lambda b: b["raw_counts"].update(identity={})),
    ("clinical extra key", lambda b: b["raw_counts"]["clinical"].update(patientId=1)),
    ("prediction extra key", lambda b: b["raw_counts"]["prediction"].update(predictionIds=[])),
    ("unit recordIds", lambda b: b["units"][0].update(recordIds=["r1"])),
    ("unit source", lambda b: b["units"][0].update(source={"recordIds": []})),
    ("state recordedAt", lambda b: b["units"][0]["states"][0].update(recordedAt="2026-01-01")),
    ("state BMI", lambda b: b["units"][0]["states"][0].update(BMI=28.0)),
    ("pi predictionId", lambda b: b["units"][0]["pi"].update(predictionId="x")),
    ("pi scoreBp", lambda b: b["units"][0]["pi"].update(scoreBp=3775)),
    ("pi_state name", lambda b: b["units"][0]["pi_state"].update(name="Jane")),
    ("pi_state sex", lambda b: b["units"][0]["pi_state"].update(sex=1)),
])
def test_unknown_keys_are_forbidden(name, mutate):
    assert_global_fallback(mode_b(_with(mutate)), "GLOBAL_ERROR_FALLBACK", "INVALID_PAYLOAD")


def test_precheck_precedence():
    assert W.precheck("x") == "INVALID_PAYLOAD"
    assert W.precheck({"version": "R-BPBC-9", "patientId": 1}) == "UNSUPPORTED_VERSION"
    assert W.precheck({"version": "R-BPBC-3", "patientId": 1}) == "INVALID_PAYLOAD"
    assert W.precheck(r4b_ev(hist(STABLE))) is None
    assert W.precheck(ev(hist(STABLE))) is None                      # R4C subset counts accepted
    assert W.RAW_COUNTS_GROUP_KEYS["clinical"] >= set(counts()["clinical"])
    assert W.RAW_COUNTS_GROUP_KEYS["prediction"] >= set(counts()["prediction"])


def test_block_passed_to_engine_unchanged(monkeypatch):
    seen = {}
    orig = W.personalize

    def spy(current, evidence, evaluate, thr):
        seen["current"], seen["evidence"] = current, evidence
        return orig(current, evidence, evaluate, thr)
    monkeypatch.setattr(W, "personalize", spy)
    block = r4b_ev(hist(STABLE))
    mode_b(block)
    assert seen["evidence"] == block                                 # raw_counts + order untouched
    assert [u["t_days"] for u in seen["evidence"]["units"]] == [-28, -21, -14, -7]
    assert set(seen["current"]) == set(XT) and "personalization" not in seen["current"]


# ═══ rounding edges (final_risk_score 4 dp, level from UNROUNDED R4C value) ═══
def _forced(monkeypatch, p_final, p_g=0.3775):
    def fake(current, evidence, evaluate, thr):
        return E.PersonalizationResult(
            version="R-BPBC-3", status="INDIVIDUALIZED", reason=None,
            global_risk_score=p_g, final_risk_score=p_final,
            final_risk_level=E.risk_level(p_final, thr),
            conditioned_risk_score=p_final, Q=0.5, gamma=1.0, delta_logit=-0.1,
            raw_adjustment=-0.1, applied_logit=-0.1, bounded=False, bp_reverted=False,
            sigma_exp=0.1, effective_clinical_units=4, pi_units=4, comparable_transitions=3,
            per_feature={k: {"mu": 1.0, "P": 1.0, "P_anchor": 1.0, "z": 0.1, "s": 0.0,
                             "K_anchor": 0.5, "K_anchor_prime": 0.5, "conditioned": 1.0}
                         for k in C.CONDITIONED_FEATURES})
    monkeypatch.setattr(W, "personalize", fake)
    return mode_b(r4b_ev(hist(STABLE)))["personalization"]


@pytest.mark.parametrize("p_final,score,level", [
    (0.199996, 0.2, "low"),          # rounds up to the threshold, level stays low
    (0.199951, 0.2, "low"),
    (0.19995, round(0.19995, 4), "low"),   # binary tie: same built-in round() as top-level risk_score
    (0.19994, 0.1999, "low"),
    (0.2, 0.2, "moderate"),
    (0.200004, 0.2, "moderate"),
    (0.349996, 0.35, "moderate"),    # rounds up to 0.35, level stays moderate
    (0.349951, 0.35, "moderate"),
    (0.34995, round(0.34995, 4), "moderate"),
    (0.35, 0.35, "high"),
    (0.350001, 0.35, "high"),
])
def test_rounding_edges(monkeypatch, p_final, score, level):
    p = _forced(monkeypatch, p_final)
    assert p["final_risk_score"] == score and p["final_risk_level"] == level
    assert p["per_feature"]["sysBP"] == {"z": 0.1, "s": 0.0, "K_anchor": 0.5, "K_anchor_prime": 0.5}


@pytest.mark.parametrize("field,value", [("sigma_exp", math.inf), ("Q", math.nan), ("gamma", -math.inf)])
def test_non_finite_never_serialized(monkeypatch, field, value):
    orig = W.personalize

    def bad(*a):
        res = orig(*a)
        setattr(res, field, value)
        return res
    monkeypatch.setattr(W, "personalize", bad)
    r = post(dict(XT, personalization=r4b_ev(hist(STABLE))))
    assert r.status_code == 200
    assert b"NaN" not in r.content and b"Infinity" not in r.content
    assert_global_fallback(r.json(), "GLOBAL_ERROR_FALLBACK", "NUMERICAL_FAILURE")


# ═══ immutability / determinism / OpenAPI / health ════════════════════════════
def test_top_level_immutable_across_blocks():
    ref = legacy().json()
    blocks = [r4b_ev(hist(STABLE)), r4b_ev(hist(ERRATIC)), r4b_ev([]),
              r4b_ev(hist(STABLE, ps=dict(PS, sexUsed=None))), {"version": "zzz"}, None, "x",
              r4b_ev(hist([(185, 104, 262, 128), (183, 103, 260, 126),
                           (186, 105, 263, 129), (184, 104, 261, 127)]))]
    finals = set()
    for b in blocks:
        body = mode_b(b)
        assert top(body) == ref
        finals.add(body["personalization"]["final_risk_score"])
    assert len(finals) >= 3                                         # history changes the block only


def test_determinism_10_identical_requests():
    body = dict(XT, personalization=r4b_ev(hist(STABLE)))
    raws = {post(body).content for _ in range(10)}
    assert len(raws) == 1
    legacy_raws = {post(XT).content for _ in range(10)}
    assert len(legacy_raws) == 1


def test_openapi_optional_block():
    spec = client.get("/openapi.json").json()
    op = spec["paths"]["/predict"]["post"]
    ref = op["requestBody"]["content"]["application/json"]["schema"]["$ref"].split("/")[-1]
    req = spec["components"]["schemas"][ref]
    assert ref == "PredictWithPersonalizationRequest"
    assert "personalization" in req["properties"]
    assert "personalization" not in req["required"]
    assert set(req["required"]) == set(XT)
    resp = spec["components"]["schemas"]["PredictResponse"]
    assert "personalization" in resp["properties"]
    assert "personalization" not in resp.get("required", [])
    blk = spec["components"]["schemas"]["PersonalizationBlock"]
    assert blk["properties"]["status"]["enum"] == sorted(STATUSES, key=[
        "INDIVIDUALIZED", "GLOBAL_INSUFFICIENT_HISTORY",
        "GLOBAL_INCOMPATIBLE_HISTORY", "GLOBAL_ERROR_FALLBACK"].index)


def test_health_unchanged():
    body = client.get("/health").json()
    assert list(body) == ["status", "model_loaded", "model_version",
                          "eligible_age_range", "training_age_range"]
