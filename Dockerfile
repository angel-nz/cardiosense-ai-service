FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Runtime only: application source and the approved Beta-0.2 artifacts.
COPY app ./app
COPY artifacts/skorp-beta-0.2 ./artifacts/skorp-beta-0.2

RUN addgroup --system cardiosense \
    && adduser --system --ingroup cardiosense --home /app cardiosense \
    && chown -R cardiosense:cardiosense /app
USER cardiosense

EXPOSE 8000

CMD ["sh", "-c", "if [ -z \"${AI_SERVICE_KEY:-}\" ] || [ \"$AI_SERVICE_KEY\" = \"internal-dev-key\" ]; then echo \"AI_SERVICE_KEY must be explicitly configured for deployment\"; exit 1; fi; exec python -m uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
