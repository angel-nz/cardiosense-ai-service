import logging

from fastapi import Depends, FastAPI, HTTPException

from app.config import MODEL_VERSION
from app.inference import SkorpPredictor, ModelLoadError
from app.personalization.wire import personalization_block
from app.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictWithPersonalizationRequest,
)
from app.security import verify_internal_key
from app.forecast.router import build_forecast_router  # NEW S2C

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
    logger.critical("Skorp-Beta-0.2 failed to load at startup: %s", _load_error)


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


_CORE_FIELDS = frozenset(PredictRequest.model_fields)


# NEW R (R4D): response_model_exclude_unset=True makes the `personalization`
# key ABSENT (not null) unless it was explicitly set — legacy responses stay
# byte-identical. Every other response field is always set explicitly.
@app.post(
    "/predict",
    response_model=PredictResponse,
    response_model_exclude_unset=True,
    dependencies=[Depends(verify_internal_key)],
)
def predict(req: PredictWithPersonalizationRequest) -> PredictResponse:
    if _predictor is None or not _predictor.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded — see /health")
    # GLOBAL inference exactly as before, on the 11 validated core fields only.
    core = PredictRequest.model_validate(req.model_dump(include=_CORE_FIELDS))
    result = _predictor.predict(core)
    if "personalization" not in req.model_fields_set:
        # MODE A — legacy / GLOBAL-only request: unchanged contract.
        return PredictResponse(**result)
    # MODE B — NEW-R request: top-level stays GLOBAL; block appended.
    # PHASE A (above) failures keep 500 / 503 / 422. PHASE B (below) only
    # runs after a valid global result and preserves it on any
    # personalization-only failure (GLOBAL_ERROR_FALLBACK, R4D-FIX1).
    block = personalization_block(
        _predictor, core.model_dump(), req.personalization,
        result["risk_score"], result["risk_level"],
    )
    return PredictResponse(**result, personalization=block)


# NEW S2C — internal forecast scoring (S). Separate route; /predict unchanged.
app.include_router(build_forecast_router(lambda: _predictor))
