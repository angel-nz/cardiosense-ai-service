import os
from pathlib import Path
from dotenv import load_dotenv

# Load ai-service/.env into the real process environment — mirrors how the
# Node backend loads backend/.env via `import 'dotenv/config'`
# (backend/src/config/env.ts). Without this, a .env file here was inert:
# os.environ.get() only sees vars already in the process environment, so a
# custom AI_SERVICE_KEY set only in .env was silently ignored and the
# fallback default was used instead — the confirmed root cause of the
# 403 diagnosed for C2-3.
#
# override=False: real environment variables (e.g. exported by the shell,
# Docker, or a process manager) still take precedence over .env, matching
# python-dotenv's default and standard practice.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# Must match the backend's AI_SERVICE_KEY (see backend/src/config/env.ts and
# backend/.env.example). Never hardcode a real value here — only a dev
# fallback that mirrors the backend's own documented default so local
# docker-compose / dev setups work out of the box.
AI_SERVICE_KEY = os.environ.get("AI_SERVICE_KEY", "internal-dev-key")

ARTIFACT_DIR = Path(
    os.environ.get(
        "SKORP_ARTIFACT_DIR",
        Path(__file__).resolve().parent.parent / "artifacts" / "skorp-beta-0.1",
    )
)

MODEL_VERSION = "Skorp-Beta-0.1"
