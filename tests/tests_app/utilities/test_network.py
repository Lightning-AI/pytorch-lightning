from unittest.mock import patch

from lightning_app.utilities.network import find_free_network_port, LightningClient


def test_port():
    assert find_free_network_port()


def test_lightning_client_retry_enabled():
    with patch("lightning_app.utilities.network._retry_wrapper") as wrapper:
        LightningClient()  # default: retry=False
        wrapper.assert_not_called()

    with patch("lightning_app.utilities.network._retry_wrapper") as wrapper:
        LightningClient(retry=True)
        wrapper.assert_called()
