"""
GestureFlow API.

    uvicorn backend.main:app --reload            # http://localhost:8000/docs
"""

import json
import logging
import time

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from backend.api import analytics, auth, calibration, gestures, recognition, sessions, ws
from backend.config import get_settings
from backend.db import database_ok
from backend.schemas import HealthOut
from backend.services.recognizer import get_recognizer

API_PREFIX = "/api/v1"
access_log = logging.getLogger("gestureflow.access")


def create_app() -> FastAPI:
    settings = get_settings()
    settings.resolved_jwt_secret()  # fail at startup, not first login, if production lacks one
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    app = FastAPI(title="GestureFlow API", version="0.4.0",
                  description="Landmark-based hand gesture recognition. Camera frames never "
                              "reach this server; clients send hand landmarks.")
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        # one JSON line per request: easy to grep, and to load into any log tool later
        t0 = time.perf_counter()
        response = await call_next(request)
        access_log.info(json.dumps({
            "method": request.method, "path": request.url.path, "status": response.status_code,
            "ms": round((time.perf_counter() - t0) * 1000, 1)}))
        return response

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_credentials=False, allow_methods=["*"],
                       allow_headers=["Authorization", "Content-Type"])

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")   # the API has no home page; send people to its docs

    api = APIRouter(prefix=API_PREFIX)

    @api.get("/health", response_model=HealthOut, tags=["health"])
    def health():
        try:
            classes = len(get_recognizer().labels)
        except Exception:
            classes = 0
        db = "ok" if database_ok() else "unavailable"
        return HealthOut(status="ok" if classes and db == "ok" else "degraded",
                         model_loaded=bool(classes), model_classes=classes, database=db,
                         sentence_provider=settings.sentence_provider)

    for module in (auth, recognition, gestures, sessions, calibration, analytics, ws):
        api.include_router(module.router)
    app.include_router(api)
    return app


app = create_app()
