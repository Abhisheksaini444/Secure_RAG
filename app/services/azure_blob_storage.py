from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
from app.services.azure_identity import build_azure_credential
from app.services.logging_service import logging_service


@dataclass(frozen=True, slots=True)
class BlobObjectInfo:
    name: str
    size: int | None = None
    content_type: str | None = None


class AzureBlobStorageService:
    """Private-by-default Azure Blob storage abstraction."""

    def __init__(
        self,
        settings: Settings | None = None,
        credential: Any | None = None,
        blob_service_client: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._credential = credential
        self._blob_service_client = blob_service_client

        if not self.settings.azure_storage_account_url:
            raise ValueError("AZURE_STORAGE_ACCOUNT_URL is required")
        if not self.settings.azure_blob_container:
            raise ValueError("AZURE_BLOB_CONTAINER is required")

    def _build_blob_service_client(self):
        if self._blob_service_client is not None:
            return self._blob_service_client

        try:
            storage_module = importlib.import_module("azure.storage.blob")
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("azure.storage.blob is not installed") from exc

        credential = self._credential or build_azure_credential(self.settings)
        self._blob_service_client = storage_module.BlobServiceClient(
            account_url=self.settings.azure_storage_account_url,
            credential=credential,
        )
        return self._blob_service_client

    def _container_client(self):
        return self._build_blob_service_client().get_container_client(self.settings.azure_blob_container)

    def ensure_private_container(self) -> None:
        container_client = self._container_client()
        try:
            container_client.create_container(public_access=None)
            logging_service.log_event("blob_container_created", status="ok", extra={"container": self.settings.azure_blob_container, "public_access": "private"})
        except Exception as exc:
            # Container may already exist; treat that as safe and continue.
            if exc.__class__.__name__ not in {"ResourceExistsError", "ResourceExistsException"}:
                raise

    def upload_bytes(self, blob_name: str, data: bytes, *, content_type: str | None = None, overwrite: bool = True, metadata: dict[str, str] | None = None) -> None:
        self.ensure_private_container()
        blob_client = self._container_client().get_blob_client(blob_name)
        blob_client.upload_blob(data, overwrite=overwrite, metadata=metadata or {}, content_type=content_type)
        logging_service.log_event("blob_upload", status="ok", extra={"container": self.settings.azure_blob_container, "blob_name": blob_name, "content_type": content_type})

    def download_bytes(self, blob_name: str) -> bytes:
        blob_client = self._container_client().get_blob_client(blob_name)
        downloader = blob_client.download_blob()
        data = downloader.readall()
        logging_service.log_event("blob_download", status="ok", extra={"container": self.settings.azure_blob_container, "blob_name": blob_name})
        return data

    def list_blob_names(self, prefix: str | None = None) -> list[str]:
        container_client = self._container_client()
        return [item.name for item in container_client.list_blobs(name_starts_with=prefix)]
