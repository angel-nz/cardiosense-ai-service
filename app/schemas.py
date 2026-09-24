from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Matches the exact payload built by backend/src/lib/aiClient.ts
    (buildFeatures). Field names/casing are dictated by that function, not
    by this service — do not rename without updating aiClient.ts too."""

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
    heartRate: int = Field(ge=30, le=250)
    glucose: float = Field(ge=30, le=500)


class PredictResponse(BaseModel):
    risk_score: float
    risk_level: Literal["low", "moderate", "high"]
    anomaly_score: float
    is_anomaly: bool
    feature_importance: Dict[str, float]
    model_version: str


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
