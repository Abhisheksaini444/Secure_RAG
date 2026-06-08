from __future__ import annotations

from types import SimpleNamespace


def test_blob_storage_is_private_by_default(monkeypatch):
    from app.config import Settings
    from app.services.azure_blob_storage import AzureBlobStorageService

    created = {}

    class FakeBlobClient:
        def __init__(self, name):
            self.name = name
            self.payload = b""

        def upload_blob(self, data, overwrite=True, metadata=None, content_type=None):
            self.payload = data
            created["upload"] = {"overwrite": overwrite, "metadata": metadata, "content_type": content_type, "data": data}

        def download_blob(self):
            return SimpleNamespace(readall=lambda: self.payload)

    class FakeContainerClient:
        def __init__(self):
            self.created_public_access = None
            self._blobs = {}

        def create_container(self, public_access=None):
            self.created_public_access = public_access

        def get_blob_client(self, blob_name):
            client = self._blobs.get(blob_name)
            if client is None:
                client = FakeBlobClient(blob_name)
                self._blobs[blob_name] = client
            return client

        def list_blobs(self, name_starts_with=None):
            return [SimpleNamespace(name=name) for name in self._blobs if name_starts_with is None or name.startswith(name_starts_with)]

    class FakeBlobServiceClient:
        def __init__(self):
            self.container_client = FakeContainerClient()

        def get_container_client(self, container_name):
            self.container_name = container_name
            return self.container_client

    settings = Settings.model_validate(
        {
            "API_KEY": "test-key",
            "AZURE_STORAGE_ACCOUNT_URL": "https://account.blob.core.windows.net/",
            "AZURE_BLOB_CONTAINER": "private-container",
        }
    )
    service = AzureBlobStorageService(settings=settings, credential=object(), blob_service_client=FakeBlobServiceClient())

    service.ensure_private_container()
    service.upload_bytes("artifact.txt", b"hello", content_type="text/plain")
    assert service.download_bytes("artifact.txt") == b"hello"
    assert service.list_blob_names() == ["artifact.txt"]
    assert service._blob_service_client.container_client.created_public_access is None
