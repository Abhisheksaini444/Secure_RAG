from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.services.logging_service import logging_service


class InMemoryRateLimitStore:
    def __init__(self) -> None:
        self._hits: dict[str, deque[datetime]] = defaultdict(deque)

    def allow(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int, int]:
        now = datetime.now(timezone.utc)
        window_start = now.timestamp() - window_seconds
        hits = self._hits[key]
        while hits and hits[0].timestamp() < window_start:
            hits.popleft()
        if len(hits) >= limit:
            retry_after = int(window_seconds - (now.timestamp() - hits[0].timestamp())) if hits else window_seconds
            return False, limit - len(hits), max(retry_after, 1)
        hits.append(now)
        return True, limit - len(hits), window_seconds


RATE_LIMIT_STORE = InMemoryRateLimitStore()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Very small, dependency-free rate limit for local and assessment usage.

    Azure production deployments should back this with a shared store such as
    Redis or a gateway policy. For this project we keep the implementation local
    to avoid introducing extra managed services beyond the assessment scope.
    """

    def __init__(self, app, exempt_paths: set[str] | None = None):
        super().__init__(app)
        self.settings = get_settings()
        self.exempt_paths = exempt_paths or {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        request_id = getattr(request.state, "request_id", None) or str(uuid4())
        client_ip = request.client.host if request.client else "unknown"
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        allowed, remaining, retry_after = RATE_LIMIT_STORE.allow(
            client_ip,
            self.settings.rate_limit_per_minute,
            60,
        )
        if not allowed:
            logging_service.log_rejection(
                "rate_limit_exceeded",
                "Too many requests from IP",
                request_id=request_id,
                client_ip=client_ip,
            )
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={"detail": "Rate limit exceeded. Try again later."},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.settings.rate_limit_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(remaining, 0))
        return response
