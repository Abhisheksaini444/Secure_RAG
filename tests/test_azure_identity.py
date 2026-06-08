from __future__ import annotations

import types


def test_managed_identity_is_preferred_when_available(monkeypatch):
    created = []

    class FakeManagedIdentityCredential:
        def __init__(self, client_id=None):
            created.append(("managed", client_id))

    class FakeDefaultAzureCredential:
        def __init__(self, exclude_managed_identity_credential=False):
            created.append(("default", exclude_managed_identity_credential))

    class FakeChainedTokenCredential:
        def __init__(self, *credentials):
            created.append(("chain", len(credentials)))
            self.credentials = credentials

    fake_module = types.SimpleNamespace(
        ManagedIdentityCredential=FakeManagedIdentityCredential,
        DefaultAzureCredential=FakeDefaultAzureCredential,
        ChainedTokenCredential=FakeChainedTokenCredential,
    )

    monkeypatch.setenv("IDENTITY_ENDPOINT", "http://localhost/msi")
    monkeypatch.setattr("app.services.azure_identity.importlib.import_module", lambda name: fake_module)

    from app.services.azure_identity import build_azure_credential

    credential = build_azure_credential()

    assert created[0][0] == "managed"
    assert created[1][0] == "default"
    assert created[-1][0] == "chain"
    assert credential.credentials[0].__class__.__name__ == "FakeManagedIdentityCredential"
