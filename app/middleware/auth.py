from __future__ import annotations

import hmac
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.services.azure_secrets import get_secret_manager
from app.services.logging_service import logging_service


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Reject unauthenticated access before the request reaches the router.

    The middleware is intentionally strict: if the service is deployed behind a
    public endpoint, the API key is the first line of defense against abuse.
    """

    def __init__(self, app, exempt_paths: set[str] | None = None):
        super().__init__(app)
        self.settings = get_settings()
        self.exempt_paths = exempt_paths or {"/health", "/docs", "/openapi.json", "/redoc"}
        self.secret_manager = get_secret_manager()
        self.configured_key = self.secret_manager.get_secret("API_KEY", env_var="API_KEY", required=False)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        request_id = request.headers.get("X-Request-Id") or str(uuid4())
        request.state.request_id = request_id
        client_ip = request.client.host if request.client else "unknown"
        request.state.client_ip = client_ip
        logging_service.bind_request(request_id=request_id, client_ip=client_ip)

        try:
            if request.url.path in self.exempt_paths:
                return await call_next(request)

            configured_key = self.configured_key.strip()
            if not configured_key:
                logging_service.log_rejection("auth_missing_configuration", "API key is not configured", request_id=request_id, client_ip=client_ip)
                return JSONResponse(status_code=500, content={"detail": "Server authentication is not configured"})

            presented_key = request.headers.get(self.settings.api_key_header)
            if not presented_key:
                logging_service.log_rejection("auth_missing_key", "API key header missing", request_id=request_id, client_ip=client_ip)
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

            if not hmac.compare_digest(presented_key.strip(), configured_key):
                logging_service.log_rejection("auth_invalid_key", "Invalid API key", request_id=request_id, client_ip=client_ip)
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

            response = await call_next(request)
            response.headers["X-Request-Id"] = request_id
            return response
        finally:
            logging_service.clear_request()
