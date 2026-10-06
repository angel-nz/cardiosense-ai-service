"""NEW R (R4C) — the ONE canonical module for the frozen personalization
contract constants.

personalizationVersion "R-BPBC-3", parameter set "R-BPBC-3 / 2026-10-05 measuredAt-causal-ordering"
(NEW R3 + R3-FIX1 + R3-FIX2 + R3-FIX3).

r_k are versioned OBSERVATION-VARIABILITY PROXIES used by the patient-state
filter — NOT pure instrument measurement error:
  * sysBP / diaBP: real-world visit-to-visit within-subject SD over the first
    7 routine-care measurements (Muntner et al., J Hypertens 2011: 13.5 / 7.7
    mmHg; median 11.5 months for those 7; underlying chart data median 2.8 y).
    Technique error, short-term biology, real change and treatment change are
    mixed and inseparable.
  * totChol / glucose: EFLM Biological Variation Database within-subject CV_I
    meta-estimates (5.3 %, updated 2026-09-09; 4.7 %, updated 2026-06-05),
    held as natural-log variance ln(1 + CV_I^2). Biological variation only.

Every other constant here (q_k, P_floor, change guard, gamma mapping, B,
minimums, finite-difference steps) is an ENGINEERING / safety parameter. PRE-T-R1
changes none of their numerical values; only the R causal-time contract changed.
They remain non-clinical constants.
"""
import math
from typing import Literal, get_args

PERSONALIZATION_VERSION = "R-BPBC-3"
PARAMETER_SET = "R-BPBC-3 / 2026-10-05 measuredAt-causal-ordering"

# The only four conditioned features, in a fixed canonical order.
CONDITIONED_FEATURES = ("sysBP", "diaBP", "totChol", "glucose")
LOG_SCALE_FEATURES = frozenset({"totChol", "glucose"})

# Lab within-subject CV_I (only used for the log-scale features).
CV_I = {"totChol": 0.053, "glucose": 0.047}

# r_k — observation-variability proxy, in state units (mmHg^2 or log-variance).
R = {
    "sysBP": 182.25,                             # 13.5² mmHg² (exact literal)
    "diaBP": 59.29,                              # 7.7² mmHg² (exact literal; 7.7**2 in floats is 59.290000000000006)
    "totChol": math.log(1.0 + CV_I["totChol"] ** 2),   # 0.00280506213209...
    "glucose": math.log(1.0 + CV_I["glucose"] ** 2),   # 0.00220656374663...
}
# q_k — latent drift per day (engineering): r_k / 365.
Q_PER_DAY = {k: v / 365.0 for k, v in R.items()}
# P_floor — posterior variance floor (engineering): r_k / 4. No explicit K_min.
P_FLOOR = {k: v / 4.0 for k, v in R.items()}

MAX_DRIFT_DAYS = 3650.0          # elapsed-time cap for drift accumulation

# Change guard: s = clip(z - GUARD_START, 0, 1)  (z = 2 → 0, z = 3 → 1)
GUARD_START = 2.0

# Prediction-history channel (gamma).
QUANT_STEP = 1e-4                                   # Decimal(5,4) persistence step δ
PI_CLAMP_LO = QUANT_STEP / 2.0                      # 0.00005
PI_CLAMP_HI = 1.0 - QUANT_STEP / 2.0                # 0.99995
MIN_PI_UNITS = 3                                    # safety/product minimum (not a d.o.f. claim)
MIN_COMPARABLE_TRANSITIONS = 2
COMPARABILITY_FIELDS = ("age", "currentSmoker", "cigsPerDay", "BPMeds", "diabetes")
GAMMA_Q_FULL = 1.0                                  # Q <= 1 → gamma 1
GAMMA_Q_ZERO = 4.0                                  # Q >= 4 → gamma 0   (gamma = clip((4 - Q)/3, 0, 1))

# Bounded combination.
B_LOGIT = 0.30

# Local finite-difference steps for logit(f(x)) at the current input.
FD_STEPS = {"sysBP": 1.0, "diaBP": 1.0, "totChol": 1.0, "glucose": 1.0, "BMI": 0.1}

# Status values — EXACTLY the backend Prisma enum PersonalizationStatus
# (R3 / R4A). One vocabulary end to end: no aliases, no shortened forms, no
# translation layer. The fallback cause lives in the separate `reason` field.
INDIVIDUALIZED = "INDIVIDUALIZED"
GLOBAL_INSUFFICIENT_HISTORY = "GLOBAL_INSUFFICIENT_HISTORY"
GLOBAL_INCOMPATIBLE_HISTORY = "GLOBAL_INCOMPATIBLE_HISTORY"
GLOBAL_ERROR_FALLBACK = "GLOBAL_ERROR_FALLBACK"
PersonalizationStatus = Literal[
    "INDIVIDUALIZED",
    "GLOBAL_INSUFFICIENT_HISTORY",
    "GLOBAL_INCOMPATIBLE_HISTORY",
    "GLOBAL_ERROR_FALLBACK",
]
PERSONALIZATION_STATUSES: tuple[str, ...] = get_args(PersonalizationStatus)


class Reason:
    """Stable machine-readable fallback / exclusion reasons (never UI text)."""
    # GLOBAL_INSUFFICIENT_HISTORY
    NO_CLINICAL_HISTORY = "NO_CLINICAL_HISTORY"
    PREDICTION_HISTORY_INSUFFICIENT = "PREDICTION_HISTORY_INSUFFICIENT"
    COMPARABLE_TRANSITIONS_INSUFFICIENT = "COMPARABLE_TRANSITIONS_INSUFFICIENT"
    # GLOBAL_INCOMPATIBLE_HISTORY
    CLINICAL_HISTORY_EXCLUDED = "CLINICAL_HISTORY_EXCLUDED"
    PREDICTION_HISTORY_EXCLUDED = "PREDICTION_HISTORY_EXCLUDED"
    SEX_PROVENANCE_UNKNOWN = "SEX_PROVENANCE_UNKNOWN"
    # GLOBAL_ERROR_FALLBACK
    EVIDENCE_UNAVAILABLE = "EVIDENCE_UNAVAILABLE"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    INVALID_PAYLOAD = "INVALID_PAYLOAD"
    INVALID_CHRONOLOGY = "INVALID_CHRONOLOGY"
    CURRENT_INPUT_INVALID = "CURRENT_INPUT_INVALID"
    LOG_DOMAIN_ERROR = "LOG_DOMAIN_ERROR"
    GRADIENT_UNDEFINED = "GRADIENT_UNDEFINED"
    INVALID_CONDITIONED_INPUT = "INVALID_CONDITIONED_INPUT"
    MODEL_EVALUATION_FAILED = "MODEL_EVALUATION_FAILED"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"
    # transition-level comparability exclusions (provenance only)
    SEX_CHANGED = "SEX_CHANGED"
    COVARIATE_CHANGED = "COVARIATE_CHANGED"
