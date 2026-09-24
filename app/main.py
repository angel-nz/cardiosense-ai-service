import logging

from fastapi import Depends, FastAPI, HTTPException

from app.config import MODEL_VERSION
from app.inference import SkorpPredictor, ModelLoadError
from app.schemas import HealthResponse, PredictRequest, PredictResponse
from app.security import verify_internal_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skorp")

app = FastAPI(title="CardioSense AI Service", version=MODEL_VERSION)

# Loaded once at process startup — never re-trained, never re-loaded per
# request. If this fails, the service stays up (so /health is reachable)
# but reports model_loaded=false and /predict refuses with 503 — no silent
# ML fallback is introduced here (see report for the backend-side mock,
# which is separate and out of this service's control).
_predictor: SkorpPredictor | None = None
_load_error: str | None = None
try:
    _predictor = SkorpPredictor()
except ModelLoadError as exc:
    _load_error = str(exc)
    logger.critical("Skorp-Beta-0.1 failed to load at startup: %s", _load_error)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = _predictor is not None and _predictor.loaded
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_version=MODEL_VERSION,
        eligible_age_range=_predictor.eligible_age_range if loaded else None,
        training_age_range=_predictor.training_age_range if loaded else None,
    )


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_internal_key)])
def predict(req: PredictRequest) -> PredictResponse:
    if _predictor is None or not _predictor.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded — see /health")
    result = _predictor.predict(req)
    return PredictResponse(**result)
