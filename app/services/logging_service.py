from __future__ import annotations

import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pythonjsonlogger.jsonlogger import JsonFormatter

from app.models.schemas import AppLogEvent

request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)
client_ip_context: ContextVar[str | None] = ContextVar("client_ip", default=None)


class SecureJsonFormatter(JsonFormatter):
    """Structured logging that avoids unbounded payloads and secret leakage."""

    def add_fields(self, log_record: dict[str, Any], record: logging.LogRecord, message_dict: dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.now(timezone.utc).isoformat()
        request_id = request_id_context.get()
        if request_id:
            log_record["request_id"] = request_id
        client_ip = client_ip_context.get()
        if client_ip:
            log_record["client_ip"] = client_ip
        log_record.setdefault("level", record.levelname)
        log_record.setdefault("logger", record.name)


class LoggingService:
    def __init__(self, logger_name: str = "secure_rag") -> None:
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                SecureJsonFormatter(
                    "%(timestamp)s %(level)s %(logger)s %(message)s %(request_id)s %(client_ip)s"
                )
            )
            self.logger.addHandler(handler)

    def bind_request(self, request_id: str | UUID | None, client_ip: str | None = None) -> None:
        if request_id is not None:
            request_id_context.set(str(request_id))
        if client_ip is not None:
            client_ip_context.set(client_ip)

    def clear_request(self) -> None:
        request_id_context.set(None)
        client_ip_context.set(None)

    def log_event(
        self,
        event: str,
        *,
        status: str | None = None,
        reason: str | None = None,
        token_usage: dict[str, int] | None = None,
        request_id: str | UUID | None = None,
        client_ip: str | None = None,
        extra: dict[str, Any] | None = None,
        level: int = logging.INFO,
    ) -> None:
        request_id_value = None if request_id is None else str(request_id)
        payload = AppLogEvent(
            request_id=request_id_value,
            event=event,
            status=status,
            reason=reason,
            token_usage=token_usage,
            client_ip=client_ip,
            extra=extra or {},
        ).model_dump(mode="json")
        self.logger.log(level, payload | {"event": event})

    def log_rejection(self, event: str, reason: str, **kwargs: Any) -> None:
        self.log_event(event, status="rejected", reason=reason, **kwargs)

    def log_response(self, event: str, status: str, token_usage: dict[str, int] | None = None, **kwargs: Any) -> None:
        self.log_event(event, status=status, token_usage=token_usage, **kwargs)


logging_service = LoggingService()
