"""NEW S2C — /forecast/score wire contract (internal, backend → AI only)."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import PredictRequest


class ForecastTarget(PredictRequest):
    """One SIMULATED 11-feature state: exactly the PredictRequest fields and
    validation; any other key (ids, history, …) is rejected."""
    model_config = ConfigDict(extra="forbid")


class IndividualizedAdjustment(BaseModel):
    """Typed REAL-cutoff R adjustment from a compatible RForecastBridgeState.
    Only the decisive scalar — never evidence, history or provenance."""
    model_config = ConfigDict(extra="forbid")
    applied_logit: float = Field(allow_inf_nan=False)


class ForecastScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_model_version: str = Field(min_length=1)
    targets: List[ForecastTarget] = Field(min_length=1, max_length=3)
    individualized_adjustment: Optional[IndividualizedAdjustment] = None


class ForecastScoreItem(BaseModel):
    targetIndex: int
    globalRiskScore: float
    finalRiskScore: float
    riskLevel: Literal["LOW", "MODERATE", "HIGH"]
    interpretation: Literal["GLOBAL", "INDIVIDUALIZED_BRIDGE"]


class ForecastScoreResponse(BaseModel):
    model_version: str
    scores: List[ForecastScoreItem]
