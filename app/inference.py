import json
import math
import logging

import joblib
import pandas as pd

from app.config import ARTIFACT_DIR, MODEL_VERSION
from app.schemas import PredictRequest

logger = logging.getLogger("skorp")

# Order matters — must exactly match scripts/train.py FEATURES.
FEATURES = [
    "male", "age", "currentSmoker", "cigsPerDay", "BPMeds", "diabetes",
    "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose",
]


def _read_age_range(metadata: dict, key: str) -> dict | None:
    """Return {'min': int, 'max': int} from metadata[key], or None if absent/malformed."""
    value = metadata.get(key)
    if not isinstance(value, dict):
        return None
    lo, hi = value.get("min"), value.get("max")
    if not (isinstance(lo, int) and isinstance(hi, int)) or isinstance(lo, bool) or isinstance(hi, bool) or lo > hi:
        return None
    return {"min": lo, "max": hi}


class ModelLoadError(RuntimeError):
    pass


class SkorpPredictor:
    """Loads Skorp-Beta-0.1 artifacts once at startup. Never trains, never
    silently falls back — if artifacts are missing or invalid this raises
    ModelLoadError and the service should be considered unhealthy/refuse to
    start (see main.py startup handler)."""

    def __init__(self, artifact_dir=ARTIFACT_DIR):
        self.artifact_dir = artifact_dir
        self.loaded = False
        self.classifier = None
        self.imputer = None
        self.anomaly_model = None
        self.feature_importance: dict[str, float] = {}
        self.risk_thresholds = {"low_max": 0.20, "moderate_max": 0.35}
        self.metadata: dict = {}
        # U8.6B — model-owned age support contract (see metadata.json).
        self.training_age_range: dict | None = None
        self.eligible_age_range: dict | None = None
        self._load()

    def _load(self):
        required = [
            "classifier.joblib", "imputer.joblib", "anomaly_model.joblib",
            "feature_importance.json", "risk_thresholds.json", "metadata.json",
        ]
        missing = [f for f in required if not (self.artifact_dir / f).exists()]
        if missing:
            raise ModelLoadError(
                f"Missing Skorp-Beta-0.1 artifacts in {self.artifact_dir}: {missing}. "
                f"Run scripts/train.py first — the AI Service does not train at startup."
            )
        try:
            self.classifier = joblib.load(self.artifact_dir / "classifier.joblib")
            self.imputer = joblib.load(self.artifact_dir / "imputer.joblib")
            self.anomaly_model = joblib.load(self.artifact_dir / "anomaly_model.joblib")
            with open(self.artifact_dir / "feature_importance.json") as f:
                fi = json.load(f)
            with open(self.artifact_dir / "risk_thresholds.json") as f:
                self.risk_thresholds = json.load(f)
            with open(self.artifact_dir / "metadata.json") as f:
                self.metadata = json.load(f)
        except Exception as exc:  # noqa: BLE001 — re-raise as ModelLoadError
            raise ModelLoadError(f"Failed to load Skorp-Beta-0.1 artifacts: {exc}") from exc

        if self.metadata.get("model_version") != MODEL_VERSION:
            raise ModelLoadError(
                f"Artifact model_version mismatch: expected {MODEL_VERSION}, "
                f"got {self.metadata.get('model_version')!r}"
            )

        top6 = fi.get("top_6") or {}
        if any(v is None for v in top6.values()):
            logger.warning("feature_importance.json has N/A values — permutation importance was non-positive at training time")
            self.feature_importance = {}
        else:
            self.feature_importance = {k: float(v) for k, v in top6.items()}

        self.training_age_range = _read_age_range(self.metadata, "training_age_range")
        self.eligible_age_range = _read_age_range(self.metadata, "eligible_age_range")
        if self.eligible_age_range is None:
            logger.warning("metadata.json has no valid eligible_age_range — consumers must treat eligibility as unknown")

        self.loaded = True
        logger.info("Skorp-Beta-0.1 artifacts loaded from %s", self.artifact_dir)

    def risk_level(self, score: float) -> str:
        """Canonical CardioSense classification of an UNROUNDED probability:
        LOW < low_max (0.20) <= MODERATE < moderate_max (0.35) <= HIGH.
        PRE-S threshold gate: a non-finite or out-of-[0, 1] value is a model
        failure, never a risk level (previously NaN/inf classified as HIGH and
        negatives as LOW)."""
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) \
                or score < 0.0 or score > 1.0:
            raise ValueError(f"risk score outside the probability domain: {score!r}")
        if score < self.risk_thresholds["low_max"]:
            return "low"
        if score < self.risk_thresholds["moderate_max"]:
            return "moderate"
        return "high"

    @staticmethod
    def _model_row(features) -> dict:
        """Request-named features (sex, age, ...) -> training column names."""
        return {
            "male": features["sex"],  # sex -> male mapping (0=female,1=male), see report divergence
            "age": features["age"],
            "currentSmoker": features["currentSmoker"],
            "cigsPerDay": features["cigsPerDay"],
            "BPMeds": features["BPMeds"],
            "diabetes": features["diabetes"],
            "totChol": features["totChol"],
            "sysBP": features["sysBP"],
            "diaBP": features["diaBP"],
            "BMI": features["BMI"],
            "heartRate": features["heartRate"],
            "glucose": features["glucose"],
        }

    def _imputed(self, features) -> pd.DataFrame:
        X = pd.DataFrame([self._model_row(features)], columns=FEATURES)
        return pd.DataFrame(self.imputer.transform(X), columns=FEATURES)

    def global_probability_unrounded(self, features) -> float:
        """NEW R (R4C) — the deployed Skorp classifier's probability, clipped to
        [0, 1] but NOT rounded. `features` uses the PredictRequest field names.
        This is the exact value /predict rounds to 4 decimals for risk_score;
        the personalization engine needs it unrounded. Never derived from a
        persisted/rounded score."""
        proba = float(self.classifier.predict_proba(self._imputed(features))[0, 1])
        return min(max(proba, 0.0), 1.0)

    def predict(self, req: PredictRequest) -> dict:
        features = req.model_dump()
        X_imp = self._imputed(features)

        proba = float(self.classifier.predict_proba(X_imp)[0, 1])
        risk_score = min(max(proba, 0.0), 1.0)
        level = self.risk_level(risk_score)

        anomaly_score = float(self.anomaly_model.decision_function(X_imp)[0])
        is_anomaly = anomaly_score < 0

        return {
            "risk_score": round(risk_score, 4),
            "risk_level": level,
            "anomaly_score": round(anomaly_score, 4),
            "is_anomaly": bool(is_anomaly),
            "feature_importance": self.feature_importance,
            "model_version": self.metadata.get("model_version", MODEL_VERSION),
        }
