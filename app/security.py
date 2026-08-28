from fastapi import Header, HTTPException, status
from app.config import AI_SERVICE_KEY


async def verify_internal_key(x_internal_key: str | None = Header(default=None)) -> None:
    """Frontend never calls this service directly (Frontend -> Backend ->
    X-Internal-Key -> AI Service). Reject missing or incorrect keys."""
    if x_internal_key is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-Internal-Key")
    if x_internal_key != AI_SERVICE_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid X-Internal-Key")
