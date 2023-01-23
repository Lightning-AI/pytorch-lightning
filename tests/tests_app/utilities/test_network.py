from lightning_app.utilities.network import find_free_network_port, LightningClient


def test_port():
    assert find_free_network_port()


def test_lightning_client_retry_enabled():

    client = LightningClient()  # default: retry=True
    assert hasattr(client.auth_service_get_user_with_http_info, "__wrapped__")

    client = LightningClient(retry=False)
    assert not hasattr(client.auth_service_get_user_with_http_info, "__wrapped__")

    client = LightningClient(retry=True)
    assert hasattr(client.auth_service_get_user_with_http_info, "__wrapped__")
