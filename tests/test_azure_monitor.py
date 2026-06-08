from __future__ import annotations

import types


def test_azure_monitor_configures_when_connection_string_present(monkeypatch):
    from app.config import Settings
    from app.services.azure_monitor import configure_azure_monitoring

    calls = {}

    fake_module = types.SimpleNamespace(
        configure_azure_monitor=lambda **kwargs: calls.update(kwargs)
    )

    monkeypatch.setattr("app.services.azure_monitor.importlib.import_module", lambda name: fake_module)

    settings = Settings.model_validate(
        {
            "API_KEY": "test-key",
            "AZURE_MONITOR_CONNECTION_STRING": "InstrumentationKey=fake;IngestionEndpoint=https://example/",
        }
    )

    assert configure_azure_monitoring(settings=settings) is True
    assert calls["connection_string"].startswith("InstrumentationKey=")
    assert calls["logger_name"] == "secure_rag"


def test_azure_monitor_noops_without_connection_string(monkeypatch):
    from app.config import Settings
    from app.services.azure_monitor import configure_azure_monitoring

    settings = Settings.model_validate({"API_KEY": "test-key"})
    assert configure_azure_monitoring(settings=settings) is False
