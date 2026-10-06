"""NEW R (R4D) — /predict adapter for the frozen R4C engine (R-BPBC-3).

Responsibilities (and nothing else):
  * strict wire validation of the optional `personalization` block BEFORE the
    engine: unknown keys anywhere in the R4B wire shape are FORBIDDEN
    (→ GLOBAL_ERROR_FALLBACK / INVALID_PAYLOAD); identity-like keys
    (patientId, recordId, …) are therefore rejected deterministically;
  * calling the R4C engine with the deployed predictor;
  * mapping the engine result to the public response block:
      - final_risk_score rounded to 4 decimals (same rounding as risk_score);
      - final_risk_level = the engine's level from the UNROUNDED value;
      - on every non-INDIVIDUALIZED status the block carries the GLOBAL result
        (final_risk_score == top-level risk_score, same level);
      - only non-clinical diagnostics are exposed (no x̃, μ, P, gradients);
      - never NaN / Infinity (non-finite → GLOBAL_ERROR_FALLBACK /
        NUMERICAL_FAILURE).

Failure boundary (R4D-FIX1). /predict has two phases:
  * PHASE A — authoritative GLOBAL inference (the route, not this module).
    Its failures keep the existing semantics (500 / 503 / 422) and never
    reach this module, so they can never become a fallback.
  * PHASE B — this module, called ONLY after Phase A produced a valid global
    result. Any failure confined to the personalization attempt (including
    R4C's GlobalInferenceError from its own internal global re-evaluation)
    preserves that result: GLOBAL_ERROR_FALLBACK + a stable existing reason.
    The boundary wraps only the personalization attempt; route-level
    serialization / programming errors are not hidden by it.
Evidence order is never changed; raw_counts are passed through to the engine
unchanged.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Mapping, Optional

from app.personalization import constants as C
from app.personalization.engine import (
    GlobalInferenceError,
    PersonalizationResult,
    is_valid_model_input,
    personalize,
)

logger = logging.getLogger("skorp.personalization")

# ─── Strict wire allowlists (exactly R4B toPersonalizationRequest) ──────────
TOP_LEVEL_KEYS = frozenset({"version", "raw_counts", "units"})
RAW_COUNTS_KEYS = frozenset({"clinical", "prediction"})
RAW_COUNTS_GROUP_KEYS = {
    "clinical": frozenset({
        "rawRecords", "droppedOutOfScope", "excludedUnmeasuredClinicalTime", "excludedInvalid", "excludedOutOfDomain",
        "excludedAmbiguousTimestamp", "excludedDuplicateOfAnchor",
        "sameTimestampCollapsed", "burstCollapsedDuplicates", "effectiveUnits",
    }),
    "prediction": frozenset({
        "candidateRows", "droppedOutOfScope", "excludedRecordNotEffective",
        "excludedVersion", "excludedMissingGlobalScore", "ambiguousGroups",
        "canonicalUnits",
    }),
}
UNIT_KEYS = frozenset({"t_days", "states", "pi", "pi_state"})
STATE_KEYS = frozenset({"t_days", *C.CONDITIONED_FEATURES})
PI_KEYS = frozenset({"score"})
PI_STATE_KEYS = frozenset({"age", "currentSmoker", "cigsPerDay", "BPMeds",
                           "diabetes", "BMI", "sexUsed"})

PER_FEATURE_EXPOSED = ("z", "s", "K_anchor", "K_anchor_prime")
_NUMERIC_FIELDS = ("Q", "gamma", "delta_logit", "raw_adjustment", "applied_logit", "sigma_exp")


def _has_unknown_keys(block: Mapping[str, Any]) -> bool:
    """True if any object in the wire shape carries a key outside its
    allowlist. Only key sets are checked here; types / ranges / chronology
    stay the engine's job (unchanged R4C reasons)."""
    if set(block) - TOP_LEVEL_KEYS:
        return True
    raw = block.get("raw_counts")
    if isinstance(raw, Mapping):
        if set(raw) - RAW_COUNTS_KEYS:
            return True
        for group, allowed in RAW_COUNTS_GROUP_KEYS.items():
            g = raw.get(group)
            if isinstance(g, Mapping) and set(g) - allowed:
                return True
    units = block.get("units")
    if isinstance(units, list):
        for u in units:
            if not isinstance(u, Mapping):
                continue
            if set(u) - UNIT_KEYS:
                return True
            states = u.get("states")
            if isinstance(states, list):
                for s in states:
                    if isinstance(s, Mapping) and set(s) - STATE_KEYS:
                        return True
            pi = u.get("pi")
            if isinstance(pi, Mapping) and set(pi) - PI_KEYS:
                return True
            ps = u.get("pi_state")
            if isinstance(ps, Mapping) and set(ps) - PI_STATE_KEYS:
                return True
    return False


def precheck(block: Any) -> Optional[str]:
    """Deterministic precedence: not an object → INVALID_PAYLOAD; then
    version ≠ R-BPBC-3 (incl. missing) → UNSUPPORTED_VERSION; then any
    unknown key → INVALID_PAYLOAD. None = hand over to the engine."""
    if not isinstance(block, Mapping):
        return C.Reason.INVALID_PAYLOAD
    if block.get("version") != C.PERSONALIZATION_VERSION:
        return C.Reason.UNSUPPORTED_VERSION
    if _has_unknown_keys(block):
        return C.Reason.INVALID_PAYLOAD
    return None


def _global_block(status: str, reason: str, risk_score: float, risk_level: str,
                  counts: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """The GLOBAL result as a personalization block (every key explicit)."""
    c = counts or {}
    return {
        "version": C.PERSONALIZATION_VERSION,
        "status": status,
        "reason": reason,
        "final_risk_score": risk_score,
        "final_risk_level": risk_level,
        "Q": None, "gamma": None, "delta_logit": None, "raw_adjustment": None,
        "applied_logit": None, "bounded": None, "bp_reverted": None, "sigma_exp": None,
        "effective_clinical_units": c.get("effective_clinical_units"),
        "pi_units": c.get("pi_units"),
        "comparable_transitions": c.get("comparable_transitions"),
        "excluded_transitions": c.get("excluded_transitions"),
        "per_feature": None,
    }


def _finite(v: Any) -> bool:
    return v is None or isinstance(v, bool) or (isinstance(v, (int, float)) and math.isfinite(v))


def to_block(result: PersonalizationResult, risk_score: float, risk_level: str) -> dict[str, Any]:
    """Map an engine result to the public block. `risk_score` / `risk_level`
    are the top-level GLOBAL values of the same response."""
    if result.status != C.INDIVIDUALIZED:
        counts = None
        if result.status != C.GLOBAL_ERROR_FALLBACK:
            counts = dict(effective_clinical_units=result.effective_clinical_units,
                          pi_units=result.pi_units,
                          comparable_transitions=result.comparable_transitions,
                          excluded_transitions=dict(result.excluded_transitions))
        return _global_block(result.status, result.reason, risk_score, risk_level, counts)

    per_feature = {k: {f: float(v[f]) for f in PER_FEATURE_EXPOSED}
                   for k, v in result.per_feature.items()}
    block = {
        "version": result.version,
        "status": result.status,
        "reason": result.reason,
        "final_risk_score": round(result.final_risk_score, 4),
        "final_risk_level": result.final_risk_level,
        "Q": result.Q, "gamma": result.gamma, "delta_logit": result.delta_logit,
        "raw_adjustment": result.raw_adjustment, "applied_logit": result.applied_logit,
        "bounded": result.bounded, "bp_reverted": result.bp_reverted,
        "sigma_exp": result.sigma_exp,
        "effective_clinical_units": result.effective_clinical_units,
        "pi_units": result.pi_units,
        "comparable_transitions": result.comparable_transitions,
        "excluded_transitions": dict(result.excluded_transitions),
        "per_feature": per_feature,
    }
    numerics = [block[k] for k in _NUMERIC_FIELDS] + [block["final_risk_score"]]
    numerics += [v for d in per_feature.values() for v in d.values()]
    if not all(_finite(v) for v in numerics):
        return _global_block(C.GLOBAL_ERROR_FALLBACK, C.Reason.NUMERICAL_FAILURE, risk_score, risk_level)
    return block


def _attempt(predictor, current: Mapping[str, Any], block: Any,
             risk_score: float, risk_level: str) -> dict[str, Any]:
    reason = precheck(block)
    if reason is not None:
        return _global_block(C.GLOBAL_ERROR_FALLBACK, reason, risk_score, risk_level)
    if not is_valid_model_input(current):
        # Passed PredictRequest (11-field ranges) but violates the engine's
        # cross-feature contract (diaBP < sysBP). The GLOBAL result is still
        # returned top-level exactly as today; personalization is refused.
        return _global_block(C.GLOBAL_ERROR_FALLBACK, C.Reason.CURRENT_INPUT_INVALID, risk_score, risk_level)
    result = personalize(current, block, predictor.global_probability_unrounded,
                         predictor.risk_thresholds)
    return to_block(result, risk_score, risk_level)


def personalization_block(predictor, current: Mapping[str, Any], block: Any,
                          risk_score: float, risk_level: str) -> dict[str, Any]:
    """PHASE B of /predict. Precondition: PHASE A already produced the valid
    GLOBAL result (`risk_score` / `risk_level`). `current` = the 11 validated
    core inputs. Never raises for a personalization-only failure: the GLOBAL
    result is preserved as GLOBAL_ERROR_FALLBACK with a stable reason."""
    t0 = time.perf_counter()
    error_type: Optional[str] = None
    try:
        out = _attempt(predictor, current, block, risk_score, risk_level)
    except GlobalInferenceError:
        # R4C's own global (re-)evaluation failed AFTER Phase A succeeded: a
        # personalization-path failure here, not an endpoint-global one.
        error_type = "GlobalInferenceError"
        out = _global_block(C.GLOBAL_ERROR_FALLBACK, C.Reason.MODEL_EVALUATION_FAILED,
                            risk_score, risk_level)
    except Exception as exc:  # noqa: BLE001 — scoped to the personalization attempt only
        error_type = type(exc).__name__
        out = _global_block(C.GLOBAL_ERROR_FALLBACK, C.Reason.UNEXPECTED_ERROR,
                            risk_score, risk_level)
    units = block.get("units") if isinstance(block, Mapping) else None
    # Non-PHI operational log only: no values, no vectors, no history, no
    # identity, no exception message (only the exception class name).
    log = logger.warning if error_type else logger.info
    log("personalization version=%s status=%s reason=%s units=%s duration_ms=%.1f%s",
        C.PERSONALIZATION_VERSION, out["status"], out["reason"],
        len(units) if isinstance(units, list) else "n/a",
        (time.perf_counter() - t0) * 1000.0,
        f" error_type={error_type}" if error_type else "")
    return out
