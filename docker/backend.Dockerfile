# GestureFlow API: FastAPI + the PyTorch model, CPU only.
# Build context is the repository root (see docker-compose.yml).

FROM python:3.12-slim-bookworm

# OpenCV and MediaPipe need these shared libraries even without a display.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, from the lockfile, so code changes don't reinstall them.
# On Linux the lock resolves torch to the CPU build (see [tool.uv.sources] in pyproject.toml).
COPY pyproject.toml uv.lock README.md alembic.ini ./
RUN uv sync --frozen --no-dev --no-install-project

COPY gestureflow/ gestureflow/
COPY backend/ backend/
COPY models/gesture_cnn.pt models/gesture_cnn.pt
COPY docs/evaluation_report.json docs/evaluation_report.json

# uid 1000: Hugging Face Spaces runs containers as this user (works everywhere else too)
RUN useradd --create-home --uid 1000 gestureflow
USER gestureflow

EXPOSE 8000
# Bring the schema up to date, then serve. One worker: the login rate limiter is in-process.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn backend.main:app --host 0.0.0.0 --port 8000 --proxy-headers"]

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request, json, sys; sys.exit(0 if json.load(urllib.request.urlopen('http://localhost:8000/api/v1/health'))['status'] == 'ok' else 1)"
