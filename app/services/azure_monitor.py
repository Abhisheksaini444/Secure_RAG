from __future__ import annotations

import importlib
from typing import Any

from app.config import Settings, get_settings
from app.services.azure_secrets import AzureSecretManager, get_secret_manager
from app.services.logging_service import logging_service


def _resolve_connection_string(settings: Settings, secret_manager: AzureSecretManager | None = None) -> str:
    manager = secret_manager or get_secret_manager()
    secret_value = manager.get_secret(
        "AZURE_MONITOR_CONNECTION_STRING",
        env_var="AZURE_MONITOR_CONNECTION_STRING",
        required=False,
    )
    if secret_value:
        return secret_value

    if settings.azure_monitor_connection_string:
        return settings.azure_monitor_connection_string.strip()

    return ""


def configure_azure_monitoring(settings: Settings | None = None, secret_manager: AzureSecretManager | None = None) -> bool:
    """Configure Azure Monitor / Log Analytics export when enabled.

    Logs remain structured JSON locally; in Azure, the OpenTelemetry distro can
    forward them to Application Insights and the connected Log Analytics
    workspace.
    """

    resolved = settings or get_settings()
    connection_string = _resolve_connection_string(resolved, secret_manager)
    if not connection_string:
        logging_service.log_event("azure_monitor_disabled", status="ok", extra={"reason": "missing_connection_string"})
        return False

    try:
        monitor_module = importlib.import_module("azure.monitor.opentelemetry")
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("azure.monitor.opentelemetry is not installed") from exc

    monitor_module.configure_azure_monitor(
        connection_string=connection_string,
        logger_name="secure_rag",
    )
    logging_service.log_event("azure_monitor_enabled", status="ok", extra={"workspace": "log_analytics"})
    return True
