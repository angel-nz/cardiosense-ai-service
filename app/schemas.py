from typing import Dict, Literal
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


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str
