"""FastAPI assembly: middleware, optional API-key auth, metrics, lifespan."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from app.agents.checkpointer import close_checkpointer, get_checkpointer
from app.api import routes
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db import session as db_session
from app.llm.client import LLMClient
from app.observability.metrics import (
    CONTENT_TYPE_LATEST,
    HTTP_LATENCY,
    generate_latest,
)

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create LangGraph checkpoint tables once at startup (non-fatal if DB is
    # not up yet: deferred lazily by get_checkpointer on first research call).
    try:
        await get_checkpointer()
    except Exception as exc:  # noqa: BLE001 - startup must not crash the API
        log.warning("checkpointer_setup_deferred", error=str(exc))
    yield
    await close_checkpointer()


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    app = FastAPI(title="OmniAgent", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def observability_middleware(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        HTTP_LATENCY.labels(request.method, request.url.path).observe(elapsed)
        response.headers["X-Process-Time-Ms"] = str(round(elapsed * 1000, 1))
        return response

    if settings.api_key:

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            # Only protects /api/v1*; /health and /metrics stay open for ops.
            if request.url.path.startswith("/api/v1"):
                if request.headers.get("X-API-Key") != settings.api_key:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Missing or invalid X-API-Key"},
                    )
            return await call_next(request)

    app.include_router(routes.router, prefix="/api/v1")

    @app.get("/health", tags=["ops"])
    async def health(deep: bool = False) -> dict:
        db_ok = await db_session.ping()
        if not deep:
            return {"status": "ok" if db_ok else "degraded", "db": db_ok}
        llm_ok = await LLMClient().ping()
        return {
            "status": "ok" if (db_ok and llm_ok) else "degraded",
            "db": db_ok,
            "llm": llm_ok,
        }

    @app.get("/metrics", tags=["ops"])
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
