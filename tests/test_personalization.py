"""NEW R (R4C) — pure personalization engine (R-BPBC-3) tests.

Uses the real deployed Skorp artifact. These are contract / mathematical tests,
NOT clinical-accuracy tests.
"""
import json
import math
import os

import pytest

os.environ.setdefault("AI_SERVICE_KEY", "internal-dev-key")

from app.inference import SkorpPredictor
from app.personalization import constants as C
from app.personalization import engine as E

PRED = SkorpPredictor()
THR = PRED.risk_thresholds
F = PRED.global_probability_unrounded

XT = dict(age=58, sex=1, currentSmoker=1, cigsPerDay=15, BPMeds=0, diabetes=0,
          totChol=240.0, sysBP=165.0, diaBP=95.0, BMI=28.0, glucose=110.0)
PS = dict(age=58, currentSmoker=1, cigsPerDay=15, BPMeds=0, diabetes=0, BMI=28.0, sexUsed=1)
ZERO_COUNTS = {
    "clinical": {"rawRecords": 0, "excludedUnmeasuredClinicalTime": 0, "excludedInvalid": 0, "excludedOutOfDomain": 0,
                 "excludedAmbiguousTimestamp": 0, "excludedDuplicateOfAnchor": 0},
    "prediction": {"excludedVersion": 0, "excludedMissingGlobalScore": 0,
                   "excludedRecordNotEffective": 0, "ambiguousGroups": 0, "canonicalUnits": 0},
}


def counts(**kw):
    c = json.loads(json.dumps(ZERO_COUNTS))
    for k, v in kw.items():
        group = "clinical" if k in c["clinical"] else "prediction"
        c[group][k] = v
    return c


def st(t, sys, dia, chol, glu):
    return {"t_days": t, "sysBP": sys, "diaBP": dia, "totChol": chol, "glucose": glu}


def unit(t, states, pi=None, ps=None):
    return {"t_days": t, "states": states, "pi": None if pi is None else {"score": pi},
            "pi_state": ps if pi is not None else None}


def ev(units, raw=None):
    return {"version": C.PERSONALIZATION_VERSION,
            "raw_counts": raw or counts(rawRecords=sum(len(u["states"]) for u in units)), "units": units}


def hist(vals, ts=(-28, -21, -14, -7), ps=PS, pis=None):
    """Single-state units; pi = rounded global score of that historical state (realistic)."""
    out = []
    for i, ((sy, di, ch, gl), t) in enumerate(zip(vals, ts)):
        x = dict(XT, sysBP=sy, diaBP=di, totChol=ch, glucose=gl)
        pi = round(F(x), 4) if pis is None else pis[i]
        out.append(unit(t, [st(t, sy, di, ch, gl)], pi, dict(ps)))
    return out


STABLE = [(140, 88, 225, 100), (138, 86, 228, 98), (142, 89, 222, 101), (139, 87, 226, 99)]


def run(current=XT, evidence=None, evaluate=F):
    return E.personalize(current, evidence, evaluate, THR)


def parsed_units(units):
    return E._parse_evidence(ev(units))[0]


# ═══ parameters ══════════════════════════════════════════════════════════════
def test_frozen_parameters():
    assert C.PERSONALIZATION_VERSION == "R-BPBC-3"
    assert C.PARAMETER_SET == "R-BPBC-3 / 2026-10-05 measuredAt-causal-ordering"
    assert C.CONDITIONED_FEATURES == ("sysBP", "diaBP", "totChol", "glucose")
    assert C.CV_I == {"totChol": 0.053, "glucose": 0.047}
    assert C.R["sysBP"] == 182.25 and C.R["diaBP"] == 59.29
    assert C.R["totChol"] == pytest.approx(0.00280506213209, rel=1e-12)
    assert C.R["glucose"] == pytest.approx(0.00220656374663, rel=1e-12)
    exp_q = {"sysBP": 0.499315068493, "diaBP": 0.162438356164,
             "totChol": 7.68510173175e-6, "glucose": 6.04538012775e-6}
    exp_floor = {"sysBP": 45.5625, "diaBP": 14.8225,
                 "totChol": 0.000701265533023, "glucose": 0.000551640936657}
    for k in C.CONDITIONED_FEATURES:
        assert C.Q_PER_DAY[k] == pytest.approx(exp_q[k], rel=1e-11)
        assert C.P_FLOOR[k] == pytest.approx(exp_floor[k], rel=1e-11)
    assert C.MAX_DRIFT_DAYS == 3650.0 and C.QUANT_STEP == 1e-4
    assert (C.PI_CLAMP_LO, C.PI_CLAMP_HI) == (0.00005, 0.99995)
    assert C.B_LOGIT == 0.30 and C.GUARD_START == 2.0
    assert (C.MIN_PI_UNITS, C.MIN_COMPARABLE_TRANSITIONS) == (3, 2)
    assert C.FD_STEPS == {"sysBP": 1.0, "diaBP": 1.0, "totChol": 1.0, "glucose": 1.0, "BMI": 0.1}
    assert THR == {"low_max": 0.2, "moderate_max": 0.35}           # artifact thresholds unchanged
    assert not hasattr(C, "K_MIN")                                  # no explicit K_min exists


def test_model_input_ranges_come_from_predict_request():
    assert E.MODEL_INPUT_RANGES["sysBP"] == (60.0, 300.0)
    assert E.MODEL_INPUT_RANGES["diaBP"] == (40.0, 200.0)
    assert set(E.MODEL_INPUT_RANGES) == set(XT)


# ═══ R3-FIX3 burst information cap ═══════════════════════════════════════════
@pytest.mark.parametrize("feature", C.CONDITIONED_FEATURES)
@pytest.mark.parametrize("m", [1, 2, 4, 10])
def test_first_burst_carries_exactly_one_reading(feature, m):
    base = {"sysBP": 140, "diaBP": 88, "totChol": 220, "glucose": 100}
    states = []
    for i in range(m):
        v = dict(base)
        v[feature] = base[feature] + i
        states.append(st(-10.0, v["sysBP"], v["diaBP"], v["totChol"], v["glucose"]))   # no drift
    fs = E.filter_feature(feature, parsed_units([unit(-10.0, states)]), XT[feature])
    assert fs.P == pytest.approx(C.R[feature], rel=1e-12)          # exactly one ordinary reading


def test_first_burst_m4_not_stronger_than_m1_and_old_init_was():
    one = E.filter_feature("sysBP", parsed_units([unit(-10.0, [st(-10.0, 140, 88, 220, 100)])]), 165.0)
    four = E.filter_feature("sysBP", parsed_units([unit(-10.0, [st(-10.0, 140 + i, 88, 220, 100) for i in range(4)])]), 165.0)
    assert four.P == pytest.approx(one.P, rel=1e-12)
    # the pre-FIX3 initialization P = r would have given 1.75x precision:
    P = C.R["sysBP"]
    for _ in range(3):
        K = P / (P + 4 * C.R["sysBP"]); P = (1 - K) * P
    assert C.R["sysBP"] / P == pytest.approx(1.75)


def test_two_states_both_influence_mu_symmetric():
    fs = E.filter_feature("sysBP", parsed_units([unit(-10.0, [st(-10.0, 140, 88, 220, 100), st(-10.0, 150, 88, 220, 100)])]), 165.0)
    assert fs.mu == pytest.approx(145.0, abs=1e-12) and fs.P == pytest.approx(C.R["sysBP"], rel=1e-12)


@pytest.mark.parametrize("feature", C.CONDITIONED_FEATURES)
def test_within_burst_drift_never_stronger_than_one_reading(feature):
    states = [st(-10.0 + i * 20 / 1440, 140 + i, 88, 220 + i, 100 + i) for i in range(4)]
    fs = E.filter_feature(feature, parsed_units([unit(states[-1]["t_days"], states)]), XT[feature])
    assert fs.P >= C.R[feature]


def test_log_scale_features_use_log_state():
    fs = E.filter_feature("totChol", parsed_units([unit(-10.0, [st(-10.0, 140, 88, 200, 100), st(-10.0, 140, 88, 250, 100)])]), 240.0)
    assert fs.mu == pytest.approx((math.log(200) + math.log(250)) / 2, abs=1e-12)


# ═══ state filter ═════════════════════════════════════════════════════════════
def test_separated_bursts_each_add_information_and_floor():
    one = E.filter_feature("sysBP", parsed_units([unit(-60.0, [st(-60.0, 140, 88, 220, 100)])]), 140.0)
    many = E.filter_feature("sysBP", parsed_units([unit(-60.0 + i, [st(-60.0 + i, 140, 88, 220, 100)]) for i in range(40)]), 140.0)
    assert many.P < one.P
    assert many.P >= C.P_FLOOR["sysBP"] and many.P == pytest.approx(C.P_FLOOR["sysBP"])


def test_drift_cap_3650_days():
    a = E.filter_feature("sysBP", parsed_units([unit(-3650.0, [st(-3650.0, 140, 88, 220, 100)])]), 140.0)
    b = E.filter_feature("sysBP", parsed_units([unit(-9000.0, [st(-9000.0, 140, 88, 220, 100)])]), 140.0)
    assert a.P_anchor == b.P_anchor == pytest.approx(C.R["sysBP"] + C.Q_PER_DAY["sysBP"] * 3650)


@pytest.mark.parametrize("seq,direction", [((140, 140, 140), 0), ((130, 140, 150), 1), ((150, 140, 130), -1)])
def test_stable_rising_falling_history(seq, direction):
    us = [unit(-30.0 + 10 * i, [st(-30.0 + 10 * i, v, 80, 220, 100)]) for i, v in enumerate(seq)]
    fs = E.filter_feature("sysBP", parsed_units(us), 140.0)
    if direction == 0:
        assert fs.mu == pytest.approx(140.0)
    else:
        assert (fs.mu - 140.0) * direction > 0        # recent values weigh more


def test_filter_deterministic():
    us = parsed_units(hist(STABLE))
    assert all(E.filter_feature("glucose", us, 110.0) == E.filter_feature("glucose", us, 110.0) for _ in range(5))


# ═══ change guard ═════════════════════════════════════════════════════════════
@pytest.mark.parametrize("z", [1.0, 2.0, 2.5, 3.0, 4.0])
def test_change_guard(z):
    us = parsed_units([unit(-7.0, [st(-7.0, 140, 88, 220, 100)])])
    P_a = C.R["sysBP"] + C.Q_PER_DAY["sysBP"] * 7.0
    current = 140.0 + z * math.sqrt(P_a + C.R["sysBP"])
    fs = E.filter_feature("sysBP", us, current)
    K_a = P_a / (P_a + C.R["sysBP"])
    assert fs.z == pytest.approx(z, abs=1e-12)
    exp_s = min(max(z - 2.0, 0.0), 1.0)
    assert fs.s == pytest.approx(exp_s, abs=1e-12)
    assert fs.K_anchor_prime == pytest.approx(K_a + (1 - K_a) * exp_s, abs=1e-12)
    if z < 2.0:
        assert fs.s == 0.0 and fs.conditioned == pytest.approx(140.0 + K_a * (current - 140.0))
    if 2.0 < z < 3.0:
        assert 0.0 < fs.s < 1.0
    if fs.s == 1.0:                                     # z ≥ 3 (incl. the z = 3 boundary)
        assert fs.K_anchor_prime == 1.0 and fs.conditioned == current      # current wins EXACTLY
    if z > 3.0:
        assert fs.s == 1.0


# ═══ conditioned-input validity ══════════════════════════════════════════════
def test_bp_cross_constraint_reverts_both_bp_features():
    x = dict(XT, sysBP=105.0, diaBP=100.0)
    units = hist([(200, 110, 225, 100)] * 3, ts=(-21, -14, -7), pis=[0.4, 0.4, 0.4])
    res = run(x, ev(units))
    assert res.status == C.INDIVIDUALIZED and res.bp_reverted is True
    assert res.per_feature["sysBP"]["conditioned"] == 105.0          # guard: s=1
    assert res.per_feature["diaBP"]["conditioned"] >= 105.0          # would violate dia < sys → reverted


def test_invalid_conditioned_vector_never_evaluated(monkeypatch):
    seen = []

    def spy(x):
        seen.append(dict(x))
        return F(x)
    orig = E.filter_feature

    def bad(feature, units, cur):
        fs = orig(feature, units, cur)
        if feature == "glucose":
            fs.conditioned = 9999.0            # outside model contract
        return fs
    monkeypatch.setattr(E, "filter_feature", bad)
    res = run(XT, ev(hist(STABLE)), evaluate=spy)
    assert res.status == C.GLOBAL_ERROR_FALLBACK and res.reason == C.Reason.INVALID_CONDITIONED_INPUT
    assert res.final_risk_score is res.global_risk_score
    assert all(E.is_valid_model_input(x) for x in seen)


# ═══ gradients ════════════════════════════════════════════════════════════════
def test_gradient_methods_at_bounds_and_cross_constraint():
    L0 = lambda x: E.clamped_logit(F(x))
    x = dict(XT)
    assert E.gradient(F, x, "sysBP", L0(x))[1] == "central"
    lo = dict(XT, sysBP=60.0, diaBP=55.0)
    assert E.gradient(F, lo, "sysBP", L0(lo))[1] == "forward"
    hi = dict(XT, sysBP=300.0)
    assert E.gradient(F, hi, "sysBP", L0(hi))[1] == "backward"
    cross = dict(XT, sysBP=100.0, diaBP=99.5)             # dia+1 ≥ sys → invalid up
    assert E.gradient(F, cross, "diaBP", L0(cross))[1] == "backward"
    assert E.gradient(F, cross, "sysBP", L0(cross))[1] == "forward"    # sys−1 ≤ dia → invalid down
    bmi = dict(XT, BMI=10.0)
    assert E.gradient(F, bmi, "BMI", L0(bmi))[1] == "forward"


def test_gradient_undefined_falls_back_exactly(monkeypatch):
    monkeypatch.setitem(E.MODEL_INPUT_RANGES, "BMI", (27.95, 28.05))   # neither BMI ± 0.1 valid
    res = run(XT, ev(hist(STABLE)))
    assert res.status == C.GLOBAL_ERROR_FALLBACK and res.reason == C.Reason.GRADIENT_UNDEFINED
    assert res.final_risk_score is res.global_risk_score


def test_nan_in_personalization_path_falls_back():
    calls = {"n": 0}

    def flaky(x):
        calls["n"] += 1
        return F(x) if calls["n"] == 1 else float("nan")
    res = run(XT, ev(hist(STABLE)), evaluate=flaky)
    assert res.status == C.GLOBAL_ERROR_FALLBACK and res.reason == C.Reason.NUMERICAL_FAILURE
    assert res.final_risk_score is res.global_risk_score


# ═══ prediction history / comparability ══════════════════════════════════════
@pytest.mark.parametrize("n,status", [(0, C.GLOBAL_INSUFFICIENT_HISTORY), (1, C.GLOBAL_INSUFFICIENT_HISTORY),
                                      (2, C.GLOBAL_INSUFFICIENT_HISTORY), (3, C.INDIVIDUALIZED)])
def test_pi_minimum(n, status):
    us = hist(STABLE[:3], ts=(-21, -14, -7))
    for i in range(3 - n):
        us[i]["pi"], us[i]["pi_state"] = None, None
    res = run(XT, ev(us))
    assert res.status == status and res.pi_units == n


def test_legacy_sex_null_not_comparable_incompatible():
    res = run(XT, ev(hist(STABLE, ps=dict(PS, sexUsed=None))))
    assert res.status == C.GLOBAL_INCOMPATIBLE_HISTORY and res.reason == C.Reason.SEX_PROVENANCE_UNKNOWN
    assert res.excluded_transitions == {C.Reason.SEX_PROVENANCE_UNKNOWN: 3}
    assert res.final_risk_score is res.global_risk_score


@pytest.mark.parametrize("field,new", [("sexUsed", 0), ("age", 59), ("currentSmoker", 0), ("cigsPerDay", 10),
                                       ("BPMeds", 1), ("diabetes", 1)])
def test_changed_covariate_breaks_comparability(field, new):
    us = hist(STABLE)
    for u in us[1::2]:                       # alternate → every transition changes the field
        u["pi_state"] = dict(u["pi_state"], **{field: new})
    res = run(XT, ev(us))
    assert res.comparable_transitions == 0
    assert res.status == C.GLOBAL_INSUFFICIENT_HISTORY and res.reason == C.Reason.COMPARABLE_TRANSITIONS_INSUFFICIENT


def test_bmi_changes_keep_comparability_and_apply_correction():
    us = hist(STABLE)
    us[1]["pi_state"] = dict(us[1]["pi_state"], BMI=29.0)
    res = run(XT, ev(us))
    assert res.status == C.INDIVIDUALIZED and res.comparable_transitions == 3
    g_bmi = res.gradients["BMI"]["g"]
    # recompute Q by hand from the frozen equations
    s2 = res.sigma_exp ** 2
    drift = sum(res.gradients[k]["g"] ** 2 * E._q_hat(k, XT[k]) for k in C.CONDITIONED_FEATURES)
    terms = []
    for a, b in zip(us, us[1:]):
        d = (E.clamped_logit(b["pi"]["score"]) - E.clamped_logit(a["pi"]["score"])
             - g_bmi * (b["pi_state"]["BMI"] - a["pi_state"]["BMI"]))
        v = 2 * s2 + drift * (b["t_days"] - a["t_days"]) + E.quantization_variance(a["pi"]["score"]) + E.quantization_variance(b["pi"]["score"])
        terms.append(d * d / v)
    assert res.Q == pytest.approx(sum(terms) / 3, rel=1e-12)


def test_incompatible_vs_insufficient_from_counts():
    us = hist(STABLE[:2], ts=(-14, -7))
    assert run(XT, ev(us)).status == C.GLOBAL_INSUFFICIENT_HISTORY
    r = run(XT, ev(us, counts(rawRecords=4, excludedVersion=2)))
    assert r.status == C.GLOBAL_INCOMPATIBLE_HISTORY and r.reason == C.Reason.PREDICTION_HISTORY_EXCLUDED
    r = run(XT, ev([], counts(rawRecords=3, excludedOutOfDomain=3)))
    assert r.status == C.GLOBAL_INCOMPATIBLE_HISTORY and r.reason == C.Reason.CLINICAL_HISTORY_EXCLUDED
    r = run(XT, ev([], counts()))
    assert r.status == C.GLOBAL_INSUFFICIENT_HISTORY and r.reason == C.Reason.NO_CLINICAL_HISTORY


# ═══ quantization ═════════════════════════════════════════════════════════════
def test_quantization_clamp_and_variance():
    assert E.clamped_logit(0.0) == pytest.approx(math.log(0.00005 / 0.99995))
    assert E.clamped_logit(1.0) == pytest.approx(math.log(0.99995 / 0.00005))
    for p in (0.0, 0.00005, 0.0001, 0.5, 0.9999, 1.0):
        v = E.quantization_variance(p)
        assert math.isfinite(v) and v > 0
    assert E.quantization_variance(0.0) == E.quantization_variance(0.00005)


# ═══ Q / gamma ════════════════════════════════════════════════════════════════
def test_gamma_mapping():
    assert E.gamma_from_q(0.0) == 1.0 and E.gamma_from_q(1.0) == 1.0
    assert E.gamma_from_q(2.5) == pytest.approx(0.5)
    assert E.gamma_from_q(4.0) == 0.0 and E.gamma_from_q(9.0) == 0.0
    qs = [i / 10 for i in range(0, 60)]
    gs = [E.gamma_from_q(q) for q in qs]
    assert all(a >= b for a, b in zip(gs, gs[1:]))


# ═══ bound / final ════════════════════════════════════════════════════════════
def test_bound_applied_and_probability_range():
    lower = run(XT, ev(hist(STABLE)))                                # Δ ≈ −0.32 → clipped
    assert lower.status == C.INDIVIDUALIZED and lower.bounded is True and lower.applied_logit == -0.30
    assert lower.final_risk_score == pytest.approx(E.sigmoid(E.clamped_logit(lower.global_risk_score) - 0.30))
    higher = run(XT, ev(hist([(185, 104, 262, 128), (183, 103, 260, 126), (186, 105, 263, 129), (184, 104, 261, 127)])))
    assert higher.applied_logit > 0 and abs(higher.applied_logit) <= 0.30
    mild = run(XT, ev(hist([(152, 91, 232, 104), (150, 90, 234, 105), (153, 92, 231, 103), (151, 91, 233, 104)])))
    assert mild.bounded is False and mild.applied_logit == mild.raw_adjustment
    for r in (lower, higher, mild):
        assert 0.0 <= r.final_risk_score <= 1.0
        assert r.final_risk_level == E.risk_level(r.final_risk_score, THR)


# ═══ exact fallback ══════════════════════════════════════════════════════════
@pytest.mark.parametrize("evidence,status", [
    (ev([], counts()), C.GLOBAL_INSUFFICIENT_HISTORY),
    (ev(hist(STABLE[:2], ts=(-14, -7))), C.GLOBAL_INSUFFICIENT_HISTORY),
    (ev(hist(STABLE[:2], ts=(-14, -7)), counts(rawRecords=5, ambiguousGroups=2)), C.GLOBAL_INCOMPATIBLE_HISTORY),
    (ev(hist(STABLE, ps=dict(PS, sexUsed=None))), C.GLOBAL_INCOMPATIBLE_HISTORY),
    (dict(ev(hist(STABLE)), version="R-BPBC-0"), C.GLOBAL_ERROR_FALLBACK),
    ({"version": "R-BPBC-3", "units": "x"}, C.GLOBAL_ERROR_FALLBACK),
    (ev([unit(-1.0, [st(2.0, 140, 88, 220, 100)])]), C.GLOBAL_ERROR_FALLBACK),     # future state
    (ev([unit(-7.0, [st(-7.0, 140, 88, 220, 100)]), unit(-14.0, [st(-14.0, 140, 88, 220, 100)])]), C.GLOBAL_ERROR_FALLBACK),
    (None, C.GLOBAL_ERROR_FALLBACK),
])
def test_exact_global_fallback(evidence, status):
    res = run(XT, evidence)
    assert res.status == status
    assert res.final_risk_score is res.global_risk_score
    assert res.global_risk_score == F(XT)                                 # bit-exact unrounded global
    assert res.final_risk_level == E.risk_level(F(XT), THR)


def test_global_failure_is_not_a_fallback():
    with pytest.raises(E.GlobalInferenceError):
        run(XT, ev(hist(STABLE)), evaluate=lambda x: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(E.GlobalInferenceError):
        run(dict(XT, diaBP=170.0), ev(hist(STABLE)))                       # invalid anchor: fail closed


def test_gamma_zero_is_individualized_not_fallback():
    erratic = hist([(140, 88, 225, 100), (175, 100, 260, 140), (132, 82, 205, 90), (170, 99, 255, 135)])
    res = run(XT, ev(erratic))
    assert res.status == C.INDIVIDUALIZED and res.reason is None
    assert res.Q >= 4 and res.gamma == 0.0 and res.applied_logit == 0.0
    assert res.final_risk_score is res.global_risk_score                   # bit-exact


# ═══ counterfactual / exchangeability / ablation / no-drift / responsiveness ═
def test_counterfactual_same_current_different_history():
    a = run(XT, ev(hist(STABLE)))
    b = run(XT, ev(hist([(185, 104, 262, 128), (183, 103, 260, 126), (186, 105, 263, 129), (184, 104, 261, 127)])))
    assert a.status == b.status == C.INDIVIDUALIZED
    assert a.global_risk_score == b.global_risk_score
    assert a.final_risk_score != b.final_risk_score
    assert a.final_risk_score < a.global_risk_score < b.final_risk_score


def test_exchangeability_no_identity():
    e1, e2 = ev(hist(STABLE)), ev(hist(STABLE))
    assert run(XT, e1).to_dict() == run(XT, e2).to_dict()
    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)
    assert not any(k == "id" or k.endswith("Id") or k.endswith("_id") for k in keys(e1))


def test_prediction_history_ablation():
    clinical = [(140, 88, 225, 100), (140 + 12.5, 88, 225, 100), (140 - 12.5, 88, 225, 100), (140 + 12.5, 88, 225, 100)]
    consistent = hist(STABLE)
    for u, (sy, di, ch, gl) in zip(consistent, clinical):
        u["states"] = [st(u["t_days"], sy, di, ch, gl)]
    a = run(XT, ev(consistent))                                           # consistent Π (STABLE scores)
    jumpy = json.loads(json.dumps(consistent))
    for u, s in zip(jumpy, (0.30, 0.42, 0.29, 0.43)):
        u["pi"]["score"] = s
    b = run(XT, ev(jumpy))                                                # inconsistent Π, same clinical
    short = json.loads(json.dumps(consistent))
    for u in short[:2]:
        u["pi"], u["pi_state"] = None, None
    c = run(XT, ev(short))                                                # Π below minimum
    assert a.status == b.status == C.INDIVIDUALIZED and a.gamma == 1.0 and b.gamma < 1.0
    assert [a.per_feature[k] for k in C.CONDITIONED_FEATURES] == [b.per_feature[k] for k in C.CONDITIONED_FEATURES]
    assert abs(b.applied_logit) < abs(a.applied_logit)
    assert c.status == C.GLOBAL_INSUFFICIENT_HISTORY and c.final_risk_score is c.global_risk_score


def test_no_drift_repeated_calls():
    e = ev(hist(STABLE))
    first = run(XT, e).to_dict()
    assert all(run(XT, e).to_dict() == first for _ in range(10))


def test_current_state_responsiveness():
    x = dict(XT, sysBP=200.0)
    res = run(x, ev(hist([(120, 78, 225, 100)] * 4)))
    assert res.per_feature["sysBP"]["z"] >= 3.0
    assert res.per_feature["sysBP"]["conditioned"] == 200.0


# ═══ global endpoint unchanged ═══════════════════════════════════════════════
def test_unrounded_helper_matches_predict_rounding():
    from app.schemas import PredictRequest
    for x in (XT, dict(XT, sex=0, age=40, sysBP=118.5, diaBP=76.0), dict(XT, glucose=250.0, diabetes=1)):
        assert round(F(x), 4) == PRED.predict(PredictRequest(**x))["risk_score"]


# ═══ R4C-FIX1 — exact status contract (literal values, status ≠ reason) ══════
EXACT = ("INDIVIDUALIZED", "GLOBAL_INSUFFICIENT_HISTORY", "GLOBAL_INCOMPATIBLE_HISTORY", "GLOBAL_ERROR_FALLBACK")


def test_status_vocabulary_is_exactly_the_prisma_enum():
    assert C.PERSONALIZATION_STATUSES == EXACT
    assert (C.INDIVIDUALIZED, C.GLOBAL_INSUFFICIENT_HISTORY, C.GLOBAL_INCOMPATIBLE_HISTORY, C.GLOBAL_ERROR_FALLBACK) == EXACT
    with pytest.raises(ValueError):
        E.PersonalizationResult("R-BPBC-3", "INSUFFICIENT", None, 0.1, 0.1, "low")   # shorthand rejected


def _gradient_undefined(monkeypatch):
    monkeypatch.setitem(E.MODEL_INPUT_RANGES, "BMI", (27.95, 28.05))
    return run(XT, ev(hist(STABLE)))


def _invalid_conditioned(monkeypatch):
    orig = E.filter_feature

    def bad(feature, units, cur):
        fs = orig(feature, units, cur)
        if feature == "glucose":
            fs.conditioned = 9999.0
        return fs
    monkeypatch.setattr(E, "filter_feature", bad)
    return run(XT, ev(hist(STABLE)))


@pytest.mark.parametrize("case,make,status,reason", [
    ("A no clinical history", lambda mp: run(XT, ev([], counts())),
     "GLOBAL_INSUFFICIENT_HISTORY", "NO_CLINICAL_HISTORY"),
    ("B n=2 prediction history", lambda mp: run(XT, ev(hist(STABLE[:2], ts=(-14, -7)))),
     "GLOBAL_INSUFFICIENT_HISTORY", "PREDICTION_HISTORY_INSUFFICIENT"),
    ("C legacy sexUsed NULL", lambda mp: run(XT, ev(hist(STABLE, ps=dict(PS, sexUsed=None)))),
     "GLOBAL_INCOMPATIBLE_HISTORY", "SEX_PROVENANCE_UNKNOWN"),
    ("D gradient undefined", _gradient_undefined, "GLOBAL_ERROR_FALLBACK", "GRADIENT_UNDEFINED"),
    ("E invalid conditioned input", _invalid_conditioned, "GLOBAL_ERROR_FALLBACK", "INVALID_CONDITIONED_INPUT"),
    ("F gamma=0 valid evidence", lambda mp: run(XT, ev(hist([(140, 88, 225, 100), (175, 100, 260, 140), (132, 82, 205, 90), (170, 99, 255, 135)]))),
     "INDIVIDUALIZED", None),
])
def test_status_literal_mapping(monkeypatch, case, make, status, reason):
    res = make(monkeypatch)
    assert res.status == status and type(res.status) is str
    assert res.reason == reason
    assert res.to_dict()["status"] == status                      # serialized literal, no translation
    if status != "INDIVIDUALIZED":
        assert res.final_risk_score is res.global_risk_score
    else:
        assert res.gamma == 0.0 and res.final_risk_score is res.global_risk_score


def test_global_failure_never_returns_global_error_fallback():
    for bad in (lambda x: float("nan"), lambda x: 1.5):
        with pytest.raises(E.GlobalInferenceError):
            run(XT, ev(hist(STABLE)), evaluate=bad)
