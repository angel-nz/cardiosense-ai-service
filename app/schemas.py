from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Matches the exact payload built by backend/src/lib/aiClient.ts
    (buildFeatures). Field names/casing are dictated by that function, not
    by this service — do not rename without updating aiClient.ts too."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=120)
    sex: int = Field(ge=0, le=1, description="0=female, 1=male (backend semantics)")
    currentSmoker: int = Field(ge=0, le=1)
    cigsPerDay: int = Field(ge=0, le=100)
    BPMeds: int = Field(ge=0, le=1)
    diabetes: int = Field(ge=0, le=1)
    totChol: float = Field(ge=50, le=800)
    sysBP: float = Field(ge=60, le=300)
    diaBP: float = Field(ge=40, le=200)
    BMI: float = Field(ge=10, le=80)
    glucose: float = Field(ge=30, le=500)


class PredictWithPersonalizationRequest(PredictRequest):
    """NEW R (R4D) — /predict request body. The 11 core fields are inherited
    unchanged from PredictRequest (same strict validation, same 422s).

    `personalization` is OPTIONAL and deliberately untyped at this layer: a
    malformed / unsupported block must NEVER turn into a 422 — it is parsed
    by app.personalization.wire and degrades to the GLOBAL result with
    status GLOBAL_ERROR_FALLBACK and a stable reason. Omitting the key keeps
    the legacy (GLOBAL-only) contract byte-identical. An explicit `null` is a
    NEW-R request with an invalid block (INVALID_PAYLOAD)."""

    personalization: Any = Field(
        default=None,
        description=(
            "Optional NEW-R personalization evidence (R-BPBC-3): "
            "{version, raw_counts{clinical, prediction}, units[{t_days, "
            "states[{t_days, sysBP, diaBP, totChol, glucose}], pi{score}|null, "
            "pi_state|null}]}. Unknown keys are rejected (GLOBAL_ERROR_FALLBACK / "
            "INVALID_PAYLOAD). Contains no patient identity."
        ),
    )


class PersonalizationFeatureDiagnostics(BaseModel):
    z: float
    s: float
    K_anchor: float
    K_anchor_prime: float


class PersonalizationBlock(BaseModel):
    """NEW R (R4D) — appended ONLY when the request carried `personalization`.
    The top-level response fields stay the GLOBAL result. On every non-
    INDIVIDUALIZED status final_risk_score == top-level risk_score."""

    version: str
    status: Literal[
        "INDIVIDUALIZED",
        "GLOBAL_INSUFFICIENT_HISTORY",
        "GLOBAL_INCOMPATIBLE_HISTORY",
        "GLOBAL_ERROR_FALLBACK",
    ]
    reason: Optional[str]
    final_risk_score: float
    final_risk_level: Literal["low", "moderate", "high"]
    Q: Optional[float]
    gamma: Optional[float]
    delta_logit: Optional[float]
    raw_adjustment: Optional[float]
    applied_logit: Optional[float]
    bounded: Optional[bool]
    bp_reverted: Optional[bool]
    sigma_exp: Optional[float]
    effective_clinical_units: Optional[int]
    pi_units: Optional[int]
    comparable_transitions: Optional[int]
    excluded_transitions: Optional[Dict[str, int]]
    per_feature: Optional[Dict[str, PersonalizationFeatureDiagnostics]]


class PredictResponse(BaseModel):
    risk_score: float
    risk_level: Literal["low", "moderate", "high"]
    anomaly_score: float
    is_anomaly: bool
    feature_importance: Dict[str, float]
    model_version: str
    # NEW R (R4D): present ONLY for NEW-R requests (the route serializes
    # with exclude_unset, so legacy responses never contain this key).
    personalization: Optional[PersonalizationBlock] = None


class AgeRange(BaseModel):
    min: int
    max: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str
    # U8.6B — model-owned age support. eligible_age_range is the range the
    # current model is allowed to predict for (chosen = observed training
    # support); training_age_range is what the training data actually
    # contained. Neither is a clinically validated range. Independent of the
    # 18-120 input sanity validation on PredictRequest.age. None if the model
    # is not loaded or its metadata lacks the contract.
    eligible_age_range: Optional[AgeRange] = None
    training_age_range: Optional[AgeRange] = None
