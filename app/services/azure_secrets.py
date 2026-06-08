from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
from app.services.azure_identity import build_azure_credential
from app.services.logging_service import logging_service


class SecretNotFoundError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SecretResolution:
    name: str
    value: str
    source: str


class AzureSecretManager:
    """Resolve secrets from Key Vault first, then local environment in dev.

    Security rules:
    - Secret values are never logged.
    - Key Vault is preferred whenever configured.
    - Managed identity is used before any secret-based Azure credential path.
    - Local development may fall back to environment variables.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        credential: Any | None = None,
        secret_client: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._credential = credential
        self._secret_client = secret_client

    def _allow_env_fallback(self) -> bool:
        return (not self.settings.is_production()) or self.settings.allow_local_key_fallback

    def _resolve_env_value(self, env_var: str | None) -> str:
        if not env_var:
            return ""
        return os.environ.get(env_var, "").strip()

    def _build_secret_client(self):
        if self._secret_client is not None:
            return self._secret_client
        if not self.settings.azure_key_vault_url:
            return None

        try:
            keyvault_module = importlib.import_module("azure.keyvault.secrets")
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("azure.keyvault.secrets is not installed") from exc

        credential = self._credential or build_azure_credential(self.settings)
        self._secret_client = keyvault_module.SecretClient(
            vault_url=self.settings.azure_key_vault_url,
            credential=credential,
        )
        return self._secret_client

    def get_secret(
        self,
        secret_name: str,
        *,
        env_var: str | None = None,
        default: str | None = None,
        required: bool = True,
    ) -> str:
        client = self._build_secret_client()
        if client is not None:
            try:
                value = client.get_secret(secret_name).value
                resolved = (value or "").strip()
                logging_service.log_event("secret_resolved", status="ok", extra={"secret_name": secret_name, "source": "key_vault"})
                if resolved:
                    return resolved
            except Exception as exc:
                # Never log the secret value or the exception chain in detail.
                logging_service.log_rejection("secret_resolution_failed", "Unable to retrieve secret from Key Vault", extra={"secret_name": secret_name, "source": "key_vault", "error": exc.__class__.__name__})
                if not self._allow_env_fallback():
                    if not required:
                        return default or ""
                    raise SecretNotFoundError(f"Secret {secret_name} could not be resolved from Key Vault") from exc

        env_value = self._resolve_env_value(env_var or secret_name)
        if env_value:
            logging_service.log_event("secret_resolved", status="ok", extra={"secret_name": secret_name, "source": "environment"})
            return env_value

        if default is not None:
            return default

        if required:
            raise SecretNotFoundError(f"Secret {secret_name} is not configured")
        return ""


_secret_manager: AzureSecretManager | None = None


def get_secret_manager() -> AzureSecretManager:
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = AzureSecretManager()
    return _secret_manager
