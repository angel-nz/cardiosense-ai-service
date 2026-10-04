"""NEW R (R4C) — pure, deterministic patient-personalization engine (R-BPBC-1).

    p_g      = f_theta(x_t)                                   (unrounded, global Skorp)
    clinical : per conditioned feature, a local-level filter over the patient's
               own prior states (R3 + R3-FIX3 burst information cap)
               → x_tilde (anchor update with change guard)
    p_c      = f_theta(x_tilde)                               (direct model evaluation)
    gamma    : consistency of the persisted GLOBAL risk history (Π) with the
               variation expected from observation variability + latent drift
    p_final  = sigmoid(logit p_g + clip(gamma · (logit p_c − logit p_g), ±B))

Properties:
  * Pure: no I/O, no clock, no randomness, no caching, no patient identity.
    Same (current input, evidence, version) → identical result.
  * Only GLOBAL historical predictions enter Π (selected by the backend, R4B).
    Individualized historical outputs never feed back.
  * Any personalization failure returns the EXACT global probability object
    (bit-exact) with a GLOBAL_* status and a stable reason. A failure of the
    GLOBAL inference itself is not a personalization fallback: it raises.
  * gamma is a deterministic reliability control. It is NOT a probability,
    confidence, calibration score or outcome likelihood. Nothing here claims
    improved clinical accuracy.
  * Not wired into /predict (R4D owns integration).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Optional

from annotated_types import Ge, Le

from app.schemas import PredictRequest
from app.personalization import constants as C

Evaluate = Callable[[Mapping[str, float]], float]


class PersonalizationError(Exception):
    """Personalization-only failure → GLOBAL_ERROR_FALLBACK with `reason`."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class GlobalInferenceError(RuntimeError):
    """The GLOBAL path itself cannot produce a valid probability (not a fallback)."""


# ─── Model-input contract (source of truth: PredictRequest field constraints) ─
def _contract_ranges() -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for name, f in PredictRequest.model_fields.items():
        lo = next((m.ge for m in f.metadata if isinstance(m, Ge)), None)
        hi = next((m.le for m in f.metadata if isinstance(m, Le)), None)
        if lo is None or hi is None:
            raise RuntimeError(f"PredictRequest.{name} lacks ge/le bounds")
        ranges[name] = (float(lo), float(hi))
    return ranges


MODEL_INPUT_RANGES = _contract_ranges()
MODEL_INPUT_FIELDS = tuple(MODEL_INPUT_RANGES)


def _is_real(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def is_valid_model_input(x: Mapping[str, Any]) -> bool:
    """All PredictRequest ranges + the cross-feature constraint diaBP < sysBP."""
    for k, (lo, hi) in MODEL_INPUT_RANGES.items():
        v = x.get(k)
        if not _is_real(v) or v < lo or v > hi:
            return False
    return x["diaBP"] < x["sysBP"]


# ─── Small numeric helpers ──────────────────────────────────────────────────
def clamped_logit(p: float) -> float:
    """logit with the frozen numerical clamp [δ/2, 1 − δ/2]."""
    q = min(max(p, C.PI_CLAMP_LO), C.PI_CLAMP_HI)
    return math.log(q / (1.0 - q))


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def quantization_variance(p: float) -> float:
    """(δ²/12) / [p(1−p)]² on the clamped p — persisted-score rounding variance."""
    q = min(max(p, C.PI_CLAMP_LO), C.PI_CLAMP_HI)
    return (C.QUANT_STEP ** 2 / 12.0) / (q * (1.0 - q)) ** 2


def gamma_from_q(q: float) -> float:
    return min(max((C.GAMMA_Q_ZERO - q) / (C.GAMMA_Q_ZERO - C.GAMMA_Q_FULL), 0.0), 1.0)


def risk_level(p: float, thresholds: Mapping[str, float]) -> str:
    """Existing thresholds, applied to the UNROUNDED probability."""
    if p < thresholds["low_max"]:
        return "low"
    if p < thresholds["moderate_max"]:
        return "moderate"
    return "high"


def _check_prob(p: Any, reason: str) -> float:
    if not _is_real(p) or p < 0.0 or p > 1.0:
        raise PersonalizationError(reason, f"probability {p!r}")
    return float(p)


# ─── Evidence parsing (R4B → AI contract; no ids, no absolute timestamps) ───
@dataclass(frozen=True)
class _State:
    t_days: float
    values: dict[str, float]


@dataclass(frozen=True)
class _Unit:
    t_days: float
    states: tuple[_State, ...]
    pi: Optional[float]
    pi_state: Optional[dict[str, Any]]


_RAW_COUNT_KEYS = {
    "clinical": ("rawRecords", "excludedInvalid", "excludedOutOfDomain",
                 "excludedAmbiguousTimestamp", "excludedDuplicateOfAnchor"),
    "prediction": ("excludedVersion", "excludedMissingGlobalScore",
                   "excludedRecordNotEffective", "ambiguousGroups", "canonicalUnits"),
}
_PI_STATE_KEYS = ("age", "currentSmoker", "cigsPerDay", "BPMeds", "diabetes", "BMI", "heartRate", "sexUsed")


def _parse_evidence(ev: Mapping[str, Any]) -> tuple[list[_Unit], dict[str, dict[str, int]]]:
    if not isinstance(ev, Mapping):
        raise PersonalizationError(C.Reason.INVALID_PAYLOAD, "evidence is not an object")
    if ev.get("version") != C.PERSONALIZATION_VERSION:
        raise PersonalizationError(C.Reason.UNSUPPORTED_VERSION, repr(ev.get("version")))

    raw = ev.get("raw_counts")
    counts: dict[str, dict[str, int]] = {}
    if not isinstance(raw, Mapping):
        raise PersonalizationError(C.Reason.INVALID_PAYLOAD, "raw_counts missing")
    for group, keys in _RAW_COUNT_KEYS.items():
        g = raw.get(group)
        if not isinstance(g, Mapping):
            raise PersonalizationError(C.Reason.INVALID_PAYLOAD, f"raw_counts.{group} missing")
        counts[group] = {}
        for k in keys:
            v = g.get(k)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise PersonalizationError(C.Reason.INVALID_PAYLOAD, f"raw_counts.{group}.{k}")
            counts[group][k] = v

    units_in = ev.get("units")
    if not isinstance(units_in, list):
        raise PersonalizationError(C.Reason.INVALID_PAYLOAD, "units is not a list")
    units: list[_Unit] = []
    for u in units_in:
        if not isinstance(u, Mapping):
            raise PersonalizationError(C.Reason.INVALID_PAYLOAD, "unit is not an object")
        t_u = u.get("t_days")
        states_in = u.get("states")
        if not _is_real(t_u):
            raise PersonalizationError(C.Reason.INVALID_CHRONOLOGY, "unit t_days not finite")
        if not isinstance(states_in, list) or len(states_in) == 0:
            raise PersonalizationError(C.Reason.INVALID_PAYLOAD, "unit without states")
        states: list[_State] = []
        for s in states_in:
            if not isinstance(s, Mapping) or not _is_real(s.get("t_days")):
                raise PersonalizationError(C.Reason.INVALID_CHRONOLOGY, "state t_days not finite")
            vals: dict[str, float] = {}
            for k in C.CONDITIONED_FEATURES:
                v = s.get(k)
                lo, hi = MODEL_INPUT_RANGES[k]
                if not _is_real(v) or v < lo or v > hi:
                    raise PersonalizationError(C.Reason.INVALID_PAYLOAD, f"state.{k}={v!r}")
                vals[k] = float(v)
            states.append(_State(float(s["t_days"]), vals))
        pi_in, ps_in = u.get("pi"), u.get("pi_state")
        pi: Optional[float] = None
        pi_state: Optional[dict[str, Any]] = None
        if pi_in is not None:
            score = pi_in.get("score") if isinstance(pi_in, Mapping) else None
            if not _is_real(score) or score < 0.0 or score > 1.0:
                raise PersonalizationError(C.Reason.INVALID_PAYLOAD, "pi.score")
            if not isinstance(ps_in, Mapping):
                raise PersonalizationError(C.Reason.INVALID_PAYLOAD, "pi without pi_state")
            pi_state = {}
            for k in _PI_STATE_KEYS:
                v = ps_in.get(k)
                if k == "sexUsed":
                    if v is not None and v not in (0, 1) or isinstance(v, bool):
                        raise PersonalizationError(C.Reason.INVALID_PAYLOAD, "pi_state.sexUsed")
                elif not _is_real(v):
                    raise PersonalizationError(C.Reason.INVALID_PAYLOAD, f"pi_state.{k}")
                pi_state[k] = v
            pi = float(score)
        units.append(_Unit(float(t_u), tuple(states), pi, pi_state))

    # Chronology: every state strictly before the anchor (t < 0), states in
    # non-decreasing real time within and across units, unit time = latest
    # member time (≥ its last distinct state). Never silently reordered.
    prev_t = -math.inf
    prev_unit_t = -math.inf
    for u in units:
        if not (u.t_days < 0.0) or u.t_days < prev_unit_t:
            raise PersonalizationError(C.Reason.INVALID_CHRONOLOGY, "unit time")
        for s in u.states:
            if not (s.t_days < 0.0) or s.t_days < prev_t or s.t_days > u.t_days:
                raise PersonalizationError(C.Reason.INVALID_CHRONOLOGY, "state time")
            prev_t = s.t_days
        prev_unit_t = u.t_days
    return units, counts


# ─── Clinical state channel ─────────────────────────────────────────────────
@dataclass
class FeatureState:
    mu: float              # posterior mean before the anchor (state scale: raw or ln)
    P: float               # posterior variance after the last historical state
    P_anchor: float        # predicted variance at the anchor (P⁻_anchor)
    z: float
    s: float
    K_anchor: float
    K_anchor_prime: float
    conditioned: float     # x_tilde in ORIGINAL units


def _to_state(feature: str, v: float) -> float:
    if feature in C.LOG_SCALE_FEATURES:
        if not (v > 0.0):
            raise PersonalizationError(C.Reason.LOG_DOMAIN_ERROR, feature)
        return math.log(v)
    return v


def filter_feature(feature: str, units: list[_Unit], current_value: float) -> FeatureState:
    """R3 local-level filter with the R3-FIX3 burst information cap."""
    r, q, floor = C.R[feature], C.Q_PER_DAY[feature], C.P_FLOOR[feature]
    mu: Optional[float] = None
    P = 0.0
    t_prev = 0.0
    for u in units:
        R_b = r * len(u.states)                       # m_b distinct states → R_b = r · m_b
        for s in u.states:
            y = _to_state(feature, s.values[feature])
            if mu is None:                            # first usable observation of the history
                mu, P, t_prev = y, R_b, s.t_days      # R3-FIX3: P = R_b (not r)
                continue
            dt = s.t_days - t_prev
            if dt < 0:
                raise PersonalizationError(C.Reason.INVALID_CHRONOLOGY, "negative Δt")
            P_minus = P + q * min(dt, C.MAX_DRIFT_DAYS)
            K = P_minus / (P_minus + R_b)
            mu = mu + K * (y - mu)
            P = max((1.0 - K) * P_minus, floor)
            t_prev = s.t_days
    if mu is None:
        raise PersonalizationError(C.Reason.INVALID_PAYLOAD, "no historical state")

    dt_anchor = 0.0 - t_prev
    P_anchor = P + q * min(dt_anchor, C.MAX_DRIFT_DAYS)
    y_t = _to_state(feature, current_value)
    z = abs(y_t - mu) / math.sqrt(P_anchor + r)
    s_guard = min(max(z - C.GUARD_START, 0.0), 1.0)
    K_a = P_anchor / (P_anchor + r)
    # s = 1 ⇒ K' = 1 exactly (K_a + (1 − K_a)·1 can round to 1 − ε in floats)
    K_p = 1.0 if s_guard >= 1.0 else K_a + (1.0 - K_a) * s_guard
    if K_p >= 1.0:
        conditioned = float(current_value)            # full guard: current measurement wins exactly
    else:
        y_tilde = mu + K_p * (y_t - mu)
        conditioned = math.exp(y_tilde) if feature in C.LOG_SCALE_FEATURES else y_tilde
    vals = (mu, P, P_anchor, z, s_guard, K_a, K_p, conditioned)
    if not all(math.isfinite(v) for v in vals):
        raise PersonalizationError(C.Reason.NUMERICAL_FAILURE, f"state {feature}")
    return FeatureState(mu, P, P_anchor, z, s_guard, K_a, K_p, conditioned)


# ─── Model evaluation / gradients ───────────────────────────────────────────
def _eval_personalization(evaluate: Evaluate, x: Mapping[str, float]) -> float:
    if not is_valid_model_input(x):     # never evaluate an invalid synthetic vector
        raise PersonalizationError(C.Reason.INVALID_CONDITIONED_INPUT, "invalid synthetic input")
    try:
        p = evaluate(x)
    except Exception as exc:  # noqa: BLE001 — personalization-only path
        raise PersonalizationError(C.Reason.MODEL_EVALUATION_FAILED, type(exc).__name__) from exc
    return _check_prob(p, C.Reason.NUMERICAL_FAILURE)


def gradient(evaluate: Evaluate, x: Mapping[str, float], feature: str, l0: float) -> tuple[float, str]:
    """d logit(f)/d feature at x — central / forward / backward by validity."""
    h = C.FD_STEPS[feature]
    up, dn = dict(x), dict(x)
    up[feature] = x[feature] + h
    dn[feature] = x[feature] - h
    up_ok, dn_ok = is_valid_model_input(up), is_valid_model_input(dn)
    if up_ok and dn_ok:
        g, method = (clamped_logit(_eval_personalization(evaluate, up))
                     - clamped_logit(_eval_personalization(evaluate, dn))) / (2.0 * h), "central"
    elif up_ok:
        g, method = (clamped_logit(_eval_personalization(evaluate, up)) - l0) / h, "forward"
    elif dn_ok:
        g, method = (l0 - clamped_logit(_eval_personalization(evaluate, dn))) / h, "backward"
    else:
        raise PersonalizationError(C.Reason.GRADIENT_UNDEFINED, feature)
    if not math.isfinite(g):
        raise PersonalizationError(C.Reason.NUMERICAL_FAILURE, f"gradient {feature}")
    return g, method


def _r_hat(feature: str, x_t: float) -> float:
    if feature in C.LOG_SCALE_FEATURES:
        return (C.CV_I[feature] * x_t) ** 2          # (c · x_t)²
    return C.R[feature]


def _q_hat(feature: str, x_t: float) -> float:
    if feature in C.LOG_SCALE_FEATURES:
        # Same delta-method mapping that gives r_hat = x²(e^{r_log} − 1) = (c·x)²
        return x_t ** 2 * math.expm1(C.Q_PER_DAY[feature])
    return C.Q_PER_DAY[feature]


# ─── Result ─────────────────────────────────────────────────────────────────
@dataclass
class PersonalizationResult:
    version: str
    status: C.PersonalizationStatus          # exactly the Prisma PersonalizationStatus values
    reason: Optional[str]                    # separate machine-readable cause (None when INDIVIDUALIZED)
    global_risk_score: float                 # unrounded p_g
    final_risk_score: float                  # unrounded p_final (IS p_g on any fallback)
    final_risk_level: str
    conditioned_risk_score: Optional[float] = None
    Q: Optional[float] = None
    gamma: Optional[float] = None
    delta_logit: Optional[float] = None
    raw_adjustment: Optional[float] = None
    applied_logit: Optional[float] = None
    bounded: Optional[bool] = None
    bp_reverted: Optional[bool] = None
    sigma_exp: Optional[float] = None
    effective_clinical_units: int = 0
    pi_units: int = 0
    comparable_transitions: int = 0
    excluded_transitions: dict[str, int] = field(default_factory=dict)
    per_feature: dict[str, dict[str, float]] = field(default_factory=dict)
    gradients: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in C.PERSONALIZATION_STATUSES:
            raise ValueError(f"invalid personalization status {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─── Engine ─────────────────────────────────────────────────────────────────
def personalize(
    current: Mapping[str, Any],
    evidence: Optional[Mapping[str, Any]],
    evaluate: Evaluate,
    risk_thresholds: Mapping[str, float],
) -> PersonalizationResult:
    """`current`: the 12 PredictRequest-named inputs of the anchor.
    `evidence`: the R4B personalization block. `evaluate(x)`: unrounded
    deployed-Skorp probability for PredictRequest-named inputs."""
    if not is_valid_model_input(current):
        # Fail closed: an invalid / ineligible anchor is a GLOBAL-path error.
        raise GlobalInferenceError("current input violates the model input contract")
    x_t = {k: current[k] for k in MODEL_INPUT_FIELDS}
    try:
        p_g = evaluate(x_t)
    except Exception as exc:  # noqa: BLE001
        raise GlobalInferenceError(f"global inference failed: {exc}") from exc
    if not _is_real(p_g) or p_g < 0.0 or p_g > 1.0:
        raise GlobalInferenceError(f"invalid global probability {p_g!r}")

    base = dict(global_risk_score=p_g, final_risk_score=p_g,
                final_risk_level=risk_level(p_g, risk_thresholds))

    def fallback(status: C.PersonalizationStatus, reason: str, **extra: Any) -> PersonalizationResult:
        # p_final IS p_g (same float object) — bit-exact global result.
        return PersonalizationResult(C.PERSONALIZATION_VERSION, status, reason, **base, **extra)

    try:
        if evidence is None:
            return fallback(C.GLOBAL_ERROR_FALLBACK, C.Reason.EVIDENCE_UNAVAILABLE)
        units, counts = _parse_evidence(evidence)

        # ── Π and comparability (pure bookkeeping, no model calls) ──────────
        pis = [u for u in units if u.pi is not None]
        excluded: dict[str, int] = {}
        comparable: list[tuple[_Unit, _Unit]] = []
        for a, b in zip(pis, pis[1:]):
            sa, sb = a.pi_state["sexUsed"], b.pi_state["sexUsed"]
            if sa is None or sb is None:
                why = C.Reason.SEX_PROVENANCE_UNKNOWN
            elif sa != sb:
                why = C.Reason.SEX_CHANGED
            elif any(a.pi_state[k] != b.pi_state[k] for k in C.COMPARABILITY_FIELDS):
                why = C.Reason.COVARIATE_CHANGED
            else:
                comparable.append((a, b))
                continue
            excluded[why] = excluded.get(why, 0) + 1
        info = dict(effective_clinical_units=len(units), pi_units=len(pis),
                    comparable_transitions=len(comparable), excluded_transitions=excluded)

        # ── Minimums: INSUFFICIENT (cold start) vs INCOMPATIBLE (exclusions) ─
        cl, pr = counts["clinical"], counts["prediction"]
        if len(units) == 0:
            if cl["rawRecords"] == 0:
                return fallback(C.GLOBAL_INSUFFICIENT_HISTORY, C.Reason.NO_CLINICAL_HISTORY, **info)
            return fallback(C.GLOBAL_INCOMPATIBLE_HISTORY, C.Reason.CLINICAL_HISTORY_EXCLUDED, **info)
        if len(pis) < C.MIN_PI_UNITS:
            pi_depth = (len(pis) + pr["ambiguousGroups"] + pr["excludedVersion"]
                        + pr["excludedMissingGlobalScore"] + pr["excludedRecordNotEffective"])
            if pi_depth >= C.MIN_PI_UNITS:
                return fallback(C.GLOBAL_INCOMPATIBLE_HISTORY, C.Reason.PREDICTION_HISTORY_EXCLUDED, **info)
            return fallback(C.GLOBAL_INSUFFICIENT_HISTORY, C.Reason.PREDICTION_HISTORY_INSUFFICIENT, **info)
        if len(comparable) < C.MIN_COMPARABLE_TRANSITIONS:
            if len(comparable) + excluded.get(C.Reason.SEX_PROVENANCE_UNKNOWN, 0) >= C.MIN_COMPARABLE_TRANSITIONS:
                return fallback(C.GLOBAL_INCOMPATIBLE_HISTORY, C.Reason.SEX_PROVENANCE_UNKNOWN, **info)
            return fallback(C.GLOBAL_INSUFFICIENT_HISTORY, C.Reason.COMPARABLE_TRANSITIONS_INSUFFICIENT, **info)

        # ── Clinical channel → conditioned input ─────────────────────────────
        states = {k: filter_feature(k, units, float(x_t[k])) for k in C.CONDITIONED_FEATURES}
        x_tilde = dict(x_t)
        for k, st in states.items():
            x_tilde[k] = st.conditioned
        bp_reverted = False
        if not (x_tilde["diaBP"] < x_tilde["sysBP"]):
            x_tilde["sysBP"], x_tilde["diaBP"] = x_t["sysBP"], x_t["diaBP"]
            bp_reverted = True
        if not is_valid_model_input(x_tilde):
            return fallback(C.GLOBAL_ERROR_FALLBACK, C.Reason.INVALID_CONDITIONED_INPUT, **info)
        p_c = _eval_personalization(evaluate, x_tilde)

        # ── Prediction-history channel → gamma ───────────────────────────────
        L_g = clamped_logit(p_g)
        grads: dict[str, tuple[float, str]] = {}
        for k in (*C.CONDITIONED_FEATURES, "BMI", "heartRate"):
            grads[k] = gradient(evaluate, x_t, k, L_g)
        sigma_exp_sq = sum(grads[k][0] ** 2 * _r_hat(k, x_t[k]) for k in C.CONDITIONED_FEATURES)
        drift_rate = sum(grads[k][0] ** 2 * _q_hat(k, x_t[k]) for k in C.CONDITIONED_FEATURES)
        g_bmi, g_hr = grads["BMI"][0], grads["heartRate"][0]
        terms = []
        for a, b in comparable:
            d = (clamped_logit(b.pi) - clamped_logit(a.pi)
                 - (g_bmi * (b.pi_state["BMI"] - a.pi_state["BMI"])
                    + g_hr * (b.pi_state["heartRate"] - a.pi_state["heartRate"])))
            dt = b.t_days - a.t_days
            if dt < 0:
                raise PersonalizationError(C.Reason.INVALID_CHRONOLOGY, "Π time")
            v = (2.0 * sigma_exp_sq + drift_rate * min(dt, C.MAX_DRIFT_DAYS)
                 + quantization_variance(a.pi) + quantization_variance(b.pi))
            if not (math.isfinite(v) and v > 0.0 and math.isfinite(d)):
                raise PersonalizationError(C.Reason.NUMERICAL_FAILURE, "transition variance")
            terms.append(d * d / v)
        Q = sum(terms) / len(terms)
        gamma = gamma_from_q(Q)

        # ── Bounded combination ──────────────────────────────────────────────
        L_c = clamped_logit(p_c)
        delta = L_c - L_g
        raw_adj = gamma * delta
        applied = min(max(raw_adj, -C.B_LOGIT), C.B_LOGIT)
        bounded = abs(raw_adj) > C.B_LOGIT
        p_final = p_g if applied == 0.0 else sigmoid(L_g + applied)   # -0.0 == 0.0 → exact p_g
        for v in (Q, gamma, delta, raw_adj, applied, p_final):
            if not math.isfinite(v):
                raise PersonalizationError(C.Reason.NUMERICAL_FAILURE, "combination")
        p_final = _check_prob(p_final, C.Reason.NUMERICAL_FAILURE)

        return PersonalizationResult(
            version=C.PERSONALIZATION_VERSION, status=C.INDIVIDUALIZED, reason=None,
            global_risk_score=p_g, final_risk_score=p_final,
            final_risk_level=risk_level(p_final, risk_thresholds),
            conditioned_risk_score=p_c, Q=Q, gamma=gamma, delta_logit=delta,
            raw_adjustment=raw_adj, applied_logit=applied, bounded=bounded,
            bp_reverted=bp_reverted, sigma_exp=math.sqrt(sigma_exp_sq),
            per_feature={k: {kk: vv for kk, vv in asdict(st).items()} for k, st in states.items()},
            gradients={k: {"g": g, "method": m} for k, (g, m) in grads.items()},
            **info,
        )
    except PersonalizationError as exc:
        return fallback(C.GLOBAL_ERROR_FALLBACK, exc.reason)
    except Exception:  # noqa: BLE001 — unexpected personalization-only failure
        return fallback(C.GLOBAL_ERROR_FALLBACK, C.Reason.UNEXPECTED_ERROR)


def personalize_with_predictor(predictor, current: Mapping[str, Any],
                               evidence: Optional[Mapping[str, Any]]) -> PersonalizationResult:
    """Convenience wrapper over the deployed SkorpPredictor (not used by /predict yet)."""
    return personalize(current, evidence, predictor.global_probability_unrounded,
                       predictor.risk_thresholds)
