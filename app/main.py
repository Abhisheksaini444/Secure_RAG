from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.middleware.auth import APIKeyAuthMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.validation import InputValidationMiddleware
from app.services.logging_service import logging_service

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.app_debug,
    docs_url="/docs" if not settings.is_production() else None,
    redoc_url="/redoc" if not settings.is_production() else None,
    openapi_url="/openapi.json" if not settings.is_production() else None,
)

# CORS is intentionally restrictive. The service is intended for controlled
# internship submission use, not public browser access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(InputValidationMiddleware)
app.add_middleware(APIKeyAuthMiddleware)

app.include_router(router)


@app.on_event("startup")
async def startup_event() -> None:
    logging_service.log_event("startup", status="ok", extra={"app_name": settings.app_name})


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logging_service.log_event("shutdown", status="ok", extra={"app_name": settings.app_name})
