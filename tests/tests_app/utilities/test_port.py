from unittest.mock import MagicMock

import pytest
from lightning.app.utilities import port
from lightning.app.utilities.port import _find_lit_app_port, disable_port, enable_port
from lightning_cloud.openapi import V1NetworkConfig


def test_find_lit_app_port(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(port, "LightningClient", MagicMock(return_value=client))

    assert _find_lit_app_port(5701) == 5701

    resp = MagicMock()
    lit_app = MagicMock()
    lit_app.id = "a"
    lit_app.spec.network_config = [
        V1NetworkConfig(host="a", port=0, enable=True),
        V1NetworkConfig(host="a", port=1, enable=False),
    ]
    resp.lightningapps = [lit_app]
    client.lightningapp_instance_service_list_lightningapp_instances.return_value = resp

    monkeypatch.setenv("LIGHTNING_CLOUD_APP_ID", "a")
    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "a")
    monkeypatch.setenv("ENABLE_MULTIPLE_WORKS_IN_DEFAULT_CONTAINER", "1")

    assert _find_lit_app_port(5701) == 1

    lit_app.spec.network_config = [
        V1NetworkConfig(host="a", port=0, enable=True),
        V1NetworkConfig(host="a", port=1, enable=True),
    ]

    with pytest.raises(RuntimeError, match="No available port was found. Please"):
        _find_lit_app_port(5701)


def test_enable_port(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(port, "LightningClient", MagicMock(return_value=client))

    assert _find_lit_app_port(5701) == 5701

    resp = MagicMock()
    lit_app = MagicMock()
    lit_app.id = "a"
    lit_app.spec.network_config = [
        V1NetworkConfig(host="a", port=0, enable=True),
        V1NetworkConfig(host="a", port=1, enable=False),
    ]
    resp.lightningapps = [lit_app]
    client.lightningapp_instance_service_list_lightningapp_instances.return_value = resp

    monkeypatch.setenv("LIGHTNING_CLOUD_APP_ID", "a")
    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "a")
    monkeypatch.setenv("ENABLE_MULTIPLE_WORKS_IN_DEFAULT_CONTAINER", "1")

    assert enable_port()

    lit_app.spec.network_config = [
        V1NetworkConfig(host="a", port=0, enable=True),
        V1NetworkConfig(host="a", port=1, enable=True),
    ]

    with pytest.raises(RuntimeError, match="No available port was found. Please"):
        assert enable_port()


def test_disable_port(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(port, "LightningClient", MagicMock(return_value=client))

    assert _find_lit_app_port(5701) == 5701

    resp = MagicMock()
    lit_app = MagicMock()
    lit_app.id = "a"
    lit_app.spec.network_config = [
        V1NetworkConfig(host="a", port=0, enable=True),
        V1NetworkConfig(host="a", port=1, enable=False),
    ]
    resp.lightningapps = [lit_app]
    client.lightningapp_instance_service_list_lightningapp_instances.return_value = resp

    monkeypatch.setenv("LIGHTNING_CLOUD_APP_ID", "a")
    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "a")
    monkeypatch.setenv("ENABLE_MULTIPLE_WORKS_IN_DEFAULT_CONTAINER", "1")

    disable_port(0)
    assert not lit_app.spec.network_config[0].enable

    lit_app.spec.network_config = [
        V1NetworkConfig(host="a", port=0, enable=True),
        V1NetworkConfig(host="a", port=1, enable=False),
    ]

    with pytest.raises(RuntimeError, match="The port 1 was already disabled."):
        disable_port(1, ignore_disabled=False)

    lit_app.spec.network_config = [
        V1NetworkConfig(host="a", port=0, enable=True),
        V1NetworkConfig(host="a", port=1, enable=False),
    ]

    with pytest.raises(ValueError, match="[0, 1]"):
        assert disable_port(10)
