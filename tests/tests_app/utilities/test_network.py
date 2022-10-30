from lightning_app.utilities.network import _MethodsRetryWrapperMeta, find_free_network_port, LightningClient


def test_port():
    assert find_free_network_port()


def test_lightning_client_retry_enabled():
    client = LightningClient()  # default: retry=False
    assert not hasattr(client.lightningwork_service_create_lightningwork, "__wrapped__")
    assert not isinstance(client, _MethodsRetryWrapperMeta)

    client = LightningClient(retry=True)
    assert hasattr(client.lightningwork_service_create_lightningwork, "__wrapped__")
    assert isinstance(client, _MethodsRetryWrapperMeta)
