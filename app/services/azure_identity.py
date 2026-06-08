from __future__ import annotations

import importlib
import os
from dataclasses import dataclass

from app.config import Settings, get_settings


@dataclass(frozen=True, slots=True)
class AzureCredentialInfo:
    source: str
    has_managed_identity: bool


def managed_identity_available() -> bool:
    return any(
        os.environ.get(name)
        for name in (
            "IDENTITY_ENDPOINT",
            "MSI_ENDPOINT",
            "AZURE_FEDERATED_TOKEN_FILE",
            "WEBSITE_INSTANCE_ID",
        )
    )


def build_azure_credential(settings: Settings | None = None):
    """Build an Azure credential chain that prefers managed identity.

    Managed identity is attempted before any secret-based credential path.
    Local development can still work without Azure auth because callers should
    fall back to environment values when Key Vault is not configured.
    """

    resolved = settings or get_settings()
    try:
        azure_identity = importlib.import_module("azure.identity")
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("azure.identity is not installed") from exc

    managed_identity_client_id = resolved.azure_managed_identity_client_id or os.environ.get("AZURE_MANAGED_IDENTITY_CLIENT_ID")
    credentials = []

    if managed_identity_available():
        credentials.append(azure_identity.ManagedIdentityCredential(client_id=managed_identity_client_id))

    # DefaultAzureCredential may use environment-based secrets, workload identity,
    # Azure CLI, or other local development auth paths. We keep it behind managed
    # identity so MI is always preferred when available.
    credentials.append(azure_identity.DefaultAzureCredential(exclude_managed_identity_credential=True))

    if len(credentials) == 1:
        return credentials[0]

    return azure_identity.ChainedTokenCredential(*credentials)
