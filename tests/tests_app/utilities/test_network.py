from lightning_app.utilities.network import find_free_network_port, LightningClient


def test_port():
    assert find_free_network_port()


def test_lightning_client_retry_enabled():
    client = LightningClient()  # default: retry=False
    assert client.lightningwork_service_create_lightningwork.__name__ == "lightningwork_service_create_lightningwork"

    client = LightningClient(retry=True)
    assert client.lightningwork_service_create_lightningwork.__name__ == "wrapped"
