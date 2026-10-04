"""NEW S2C — pure forecast scoring math (no I/O, no persistence, no logging).

For each SIMULATED 12-feature target state:

    p_g     = unrounded GLOBAL Skorp probability (same path as /predict and R)
    p_final = p_g                                         (no adjustment → GLOBAL)
    p_final = p_g if L == 0 else sigmoid(clamped_logit(p_g) + L)
                                                          (typed bridge L → INDIVIDUALIZED_BRIDGE)

The bridge branch is EXACTLY R4C's bounded-combination step
(engine.personalize: `p_final = p_g if applied == 0.0 else sigmoid(L_g + applied)`
with `L_g = clamped_logit(p_g)`), using the engine's own helpers imported
read-only. L is the REAL-cutoff `applied_logit` R already produced (and
bounded) for the anchor; nothing here re-runs R, reads evidence, or feeds a
simulated state into `personalize()`.

The level is classified from the UNROUNDED p_final with the canonical
classifier; scores are rounded to 4 decimals only afterwards.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Optional, Sequence

from app.personalization.engine import clamped_logit, sigmoid  # read-only reuse of R4C math

GLOBAL = "GLOBAL"
INDIVIDUALIZED_BRIDGE = "INDIVIDUALIZED_BRIDGE"


class ForecastScoringError(RuntimeError):
    """Technical scoring failure (non-finite / out-of-domain probability)."""


def _prob(p: object, what: str) -> float:
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0.0 or p > 1.0:
        raise ForecastScoringError(f"{what} outside the probability domain")
    return float(p)


def final_probability(p_g: float, applied_logit: Optional[float]) -> float:
    """The forecast-time combination. `applied_logit=None` ⇒ GLOBAL."""
    if applied_logit is None:
        return p_g
    if not math.isfinite(applied_logit):
        raise ForecastScoringError("non-finite individualized adjustment")
    return p_g if applied_logit == 0.0 else sigmoid(clamped_logit(p_g) + applied_logit)


def score_targets(
    evaluate: Callable[[Mapping[str, float]], float],
    classify: Callable[[float], str],
    targets: Sequence[Mapping[str, float]],
    applied_logit: Optional[float],
) -> list[dict]:
    """`evaluate` = unrounded global probability (SkorpPredictor.global_probability_unrounded);
    `classify` = canonical level of an UNROUNDED probability (SkorpPredictor.risk_level).
    Returns one item per target, in input order."""
    interpretation = GLOBAL if applied_logit is None else INDIVIDUALIZED_BRIDGE
    out = []
    for i, x in enumerate(targets):
        p_g = _prob(evaluate(x), "global probability")
        p_f = _prob(final_probability(p_g, applied_logit), "final probability")
        out.append({
            "targetIndex": i,
            "globalRiskScore": round(p_g, 4),
            "finalRiskScore": round(p_f, 4),
            "riskLevel": classify(p_f).upper(),          # from the UNROUNDED p_final
            "interpretation": interpretation,
            # unrounded values stay internal (tests); never serialized
            "_p_global": p_g,
            "_p_final": p_f,
        })
    return out
