from __future__ import annotations

import types


def test_secret_manager_prefers_key_vault_over_env(monkeypatch):
    from app.config import Settings
    from app.services.azure_secrets import AzureSecretManager

    monkeypatch.setenv("GEMINI_API_KEY", "env-fallback")

    class FakeSecret:
        def __init__(self, value):
            self.value = value

    class FakeSecretClient:
        def __init__(self, vault_url=None, credential=None):
            self.vault_url = vault_url
            self.credential = credential

        def get_secret(self, name):
            return FakeSecret("vault-secret")

    settings = Settings.model_validate(
        {
            "API_KEY": "local-api-key",
            "GEMINI_API_KEY": "env-fallback",
            "AZURE_KEY_VAULT_URL": "https://vault.vault.azure.net/",
            "ALLOW_LOCAL_KEY_FALLBACK": True,
        }
    )
    manager = AzureSecretManager(settings=settings, secret_client=FakeSecretClient())

    assert manager.get_secret("GEMINI_API_KEY", env_var="GEMINI_API_KEY") == "vault-secret"


def test_secret_manager_falls_back_to_env_in_local_dev(monkeypatch):
    from app.config import Settings
    from app.services.azure_secrets import AzureSecretManager

    monkeypatch.setenv("API_KEY", "local-secret")

    settings = Settings.model_validate(
        {
            "API_KEY": "local-secret",
            "ALLOW_LOCAL_KEY_FALLBACK": False,
        }
    )
    manager = AzureSecretManager(settings=settings, secret_client=None)

    assert manager.get_secret("API_KEY", env_var="API_KEY") == "local-secret"
