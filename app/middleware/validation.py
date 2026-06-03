from __future__ import annotations

import json
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.security.prompt_injection import DEFAULT_PROMPT_INJECTION_DETECTOR
from app.services.logging_service import logging_service


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Validate request bodies before they reach business logic.

    This middleware rejects empty prompts, oversized payloads, and obvious
    prompt-injection attempts early to reduce compute cost and attack surface.
    """

    def __init__(self, app, exempt_paths: set[str] | None = None):
        super().__init__(app)
        self.settings = get_settings()
        self.exempt_paths = exempt_paths or {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        request_id = getattr(request.state, "request_id", None) or str(uuid4())
        client_ip = getattr(request.state, "client_ip", None)

        if request.method in {"POST", "PUT", "PATCH"} and request.url.path not in self.exempt_paths:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                body = await request.body()
                if not body:
                    logging_service.log_rejection("validation_empty_body", "Request body is empty", request_id=request_id, client_ip=client_ip)
                    return JSONResponse(status_code=400, content={"detail": "Request body cannot be empty"})
                if len(body) > self.settings.max_prompt_chars * 8:
                    logging_service.log_rejection("validation_body_too_large", "Request body exceeds size limit", request_id=request_id, client_ip=client_ip)
                    return JSONResponse(status_code=413, content={"detail": "Request body too large"})

                # The body is read here so we can stop obvious attacks before
                # they reach the handler. We then replay the cached bytes so the
                # downstream FastAPI route still receives the full payload.
                try:
                    parsed = json.loads(body.decode("utf-8", errors="ignore"))
                except json.JSONDecodeError:
                    logging_service.log_rejection("validation_malformed_json", "Malformed JSON payload", request_id=request_id, client_ip=client_ip)
                    return JSONResponse(status_code=400, content={"detail": "Malformed JSON"})

                normalized = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
                assessment = DEFAULT_PROMPT_INJECTION_DETECTOR.assess(normalized)
                if assessment.is_suspicious:
                    logging_service.log_rejection(
                        "validation_prompt_injection",
                        ",".join(assessment.matched_rules),
                        request_id=request_id,
                        client_ip=client_ip,
                    )
                    return JSONResponse(status_code=400, content={"detail": "Suspicious request rejected"})

                async def receive() -> dict[str, object]:
                    return {"type": "http.request", "body": body, "more_body": False}

                request = Request(request.scope, receive)

        response = await call_next(request)
        return response
