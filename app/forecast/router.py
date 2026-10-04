"""NEW S2C — POST /forecast/score (internal service key, no persistence).

Errors: 401/403 key · 422 invalid request/target · 409 model-version
mismatch · 503 model not loaded · 500 non-finite scoring result.
Logs only aggregate metadata (target count, adjustment present) — never ids
or clinical values.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import ValidationError

from app.config import MODEL_VERSION
from app.forecast.schemas import ForecastScoreRequest, ForecastScoreResponse
from app.forecast.scoring import ForecastScoringError, score_targets
from app.personalization.engine import is_valid_model_input
from app.security import verify_internal_key

logger = logging.getLogger("skorp.forecast")


def build_forecast_router(get_predictor: Callable[[], Optional[object]]) -> APIRouter:
    router = APIRouter()

    @router.post("/forecast/score", response_model=ForecastScoreResponse,
                 dependencies=[Depends(verify_internal_key)])
    def forecast_score(body: Any = Body(...)) -> ForecastScoreResponse:
        # Validated here (not as a typed parameter) so a 422 never echoes the
        # submitted values (clinical state, non-finite numbers) back.
        try:
            req = ForecastScoreRequest.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=[
                {"loc": list(e["loc"]), "type": e["type"]} for e in exc.errors(include_input=False, include_url=False)
            ]) from None
        predictor = get_predictor()
        if predictor is None or not predictor.loaded:
            raise HTTPException(status_code=503, detail="Model not loaded — see /health")
        loaded_version = predictor.metadata.get("model_version", MODEL_VERSION)
        if req.expected_model_version != loaded_version:
            raise HTTPException(status_code=409, detail={"code": "MODEL_VERSION_MISMATCH", "loaded": loaded_version})
        targets = [t.model_dump() for t in req.targets]
        for i, x in enumerate(targets):
            if not is_valid_model_input(x):              # cross-field (diaBP < sysBP) + contract ranges
                raise HTTPException(status_code=422, detail={"code": "TARGET_INVALID", "targetIndex": i})
        adj = req.individualized_adjustment.applied_logit if req.individualized_adjustment else None
        try:
            items = score_targets(predictor.global_probability_unrounded, predictor.risk_level, targets, adj)
        except (ForecastScoringError, ValueError) as exc:
            logger.error("forecast scoring failed: %s", type(exc).__name__)
            raise HTTPException(status_code=500, detail={"code": "FORECAST_SCORING_FAILED"}) from None
        logger.info("forecast scored: targets=%d adjustment=%s", len(items), adj is not None)
        return ForecastScoreResponse(
            model_version=loaded_version,
            scores=[{k: v for k, v in it.items() if not k.startswith("_")} for it in items],
        )

    return router
