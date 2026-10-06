"""PRE-S — canonical risk-threshold contract (AI service).

    LOW       risk <  0.20
    MODERATE  0.20 <= risk < 0.35
    HIGH      risk >= 0.35

The level is always derived from the UNROUNDED probability; the 4-decimal
risk_score is rounded afterwards and never re-classified.
"""
import json
import math
import os

import numpy as np
import pytest

os.environ.setdefault("AI_SERVICE_KEY", "internal-dev-key")

from fastapi.testclient import TestClient

import app.main as M
from app.config import ARTIFACT_DIR
from app.personalization import engine as E

H = {"X-Internal-Key": "internal-dev-key"}
client = TestClient(M.app)
client_nr = TestClient(M.app, raise_server_exceptions=False)
PRED = M._predictor
XT = dict(age=58, sex=1, currentSmoker=1, cigsPerDay=15, BPMeds=0, diabetes=0,
          totChol=240.0, sysBP=165.0, diaBP=95.0, BMI=28.0, glucose=110.0)

below = lambda x: math.nextafter(x, -math.inf)
above = lambda x: math.nextafter(x, math.inf)

MATRIX = [
    (0.0, "low"), (0.1999, "low"), (below(0.20), "low"),
    (0.20, "moderate"), (above(0.20), "moderate"),
    (0.3499, "moderate"), (below(0.35), "moderate"),
    (0.35, "high"), (above(0.35), "high"), (1.0, "high"),
]


def test_runtime_loaded_thresholds_are_canonical():
    # the values the RUNNING predictor actually uses (not just the file)
    assert PRED.loaded and PRED.risk_thresholds == {"low_max": 0.20, "moderate_max": 0.35}
    with open(ARTIFACT_DIR / "risk_thresholds.json") as f:
        assert json.load(f) == PRED.risk_thresholds
    with open(ARTIFACT_DIR / "metadata.json") as f:
        assert json.load(f)["risk_thresholds"] == PRED.risk_thresholds


@pytest.mark.parametrize("p,level", MATRIX)
def test_global_classifier_boundaries(p, level):
    assert PRED.risk_level(p) == level


@pytest.mark.parametrize("p,level", MATRIX)
def test_personalization_classifier_same_boundaries(p, level):
    # R4C final-level helper uses the SAME runtime thresholds object
    assert E.risk_level(p, PRED.risk_thresholds) == level


def test_global_and_personalization_classifiers_never_diverge():
    grid = np.unique(np.concatenate([np.linspace(0.0, 1.0, 100_001),
                                     [below(0.2), 0.2, above(0.2), below(0.35), 0.35, above(0.35)]]))
    for p in grid:
        assert PRED.risk_level(float(p)) == E.risk_level(float(p), PRED.risk_thresholds)


@pytest.mark.parametrize("bad", [-1e-12, -0.1, 1.0000001, 2.0, float("nan"), float("inf"), float("-inf"), True, None, "0.3"])
def test_out_of_domain_is_rejected_not_classified(bad):
    with pytest.raises(ValueError):
        PRED.risk_level(bad)


def _force_proba(monkeypatch, p):
    monkeypatch.setattr(PRED.classifier, "predict_proba", lambda X: np.array([[1.0 - p, p]]))


@pytest.mark.parametrize("p,score,level", [
    (0.199996, 0.2, "low"),         # rounds UP to 0.2000, level stays LOW (unrounded < 0.20)
    (0.19999, 0.2, "low"),
    (0.19994, 0.1999, "low"),
    (0.2, 0.2, "moderate"),
    (0.200004, 0.2, "moderate"),
    (0.349996, 0.35, "moderate"),   # rounds UP to 0.3500, level stays MODERATE
    (0.34996, 0.35, "moderate"),
    (0.3499, 0.3499, "moderate"),
    (0.35, 0.35, "high"),
    (0.350004, 0.35, "high"),
    (0.0, 0.0, "low"),
    (1.0, 1.0, "high"),
])
def test_predict_classifies_unrounded_then_rounds(monkeypatch, p, score, level):
    _force_proba(monkeypatch, p)
    body = client.post("/predict", json=XT, headers=H).json()
    assert body["risk_score"] == score and body["risk_level"] == level


@pytest.mark.parametrize("p,score,level", [(0.349996, 0.35, "moderate"), (0.199996, 0.2, "low"), (0.35, 0.35, "high")])
def test_personalized_fallback_uses_same_unrounded_level(monkeypatch, p, score, level):
    _force_proba(monkeypatch, p)
    block = {"version": "R-BPBC-3", "units": [],
             "raw_counts": {"clinical": {"rawRecords": 0, "excludedInvalid": 0, "excludedOutOfDomain": 0, "excludedUnmeasuredClinicalTime": 0,
                                         "excludedAmbiguousTimestamp": 0, "excludedDuplicateOfAnchor": 0},
                            "prediction": {"excludedVersion": 0, "excludedMissingGlobalScore": 0,
                                           "excludedRecordNotEffective": 0, "ambiguousGroups": 0, "canonicalUnits": 0}}}
    body = client.post("/predict", json=dict(XT, personalization=block), headers=H).json()
    pz = body["personalization"]
    assert (body["risk_score"], body["risk_level"]) == (score, level)
    assert pz["status"] == "GLOBAL_INSUFFICIENT_HISTORY"
    assert (pz["final_risk_score"], pz["final_risk_level"]) == (score, level)


def test_nan_probability_is_a_global_failure_not_a_level(monkeypatch):
    monkeypatch.setattr(PRED.classifier, "predict_proba", lambda X: np.array([[math.nan, math.nan]]))
    r = client_nr.post("/predict", json=XT, headers=H)
    assert r.status_code == 500
    assert b"high" not in r.content and b"low" not in r.content
