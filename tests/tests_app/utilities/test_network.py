from unittest import mock

import pytest

from lightning.app.core import constants
from lightning.app.utilities.network import find_free_network_port, LightningClient


def test_find_free_network_port():
    """Tests that `find_free_network_port` gives expected outputs and raises if a free port couldn't be found."""
    assert find_free_network_port()

    with mock.patch("lightning.app.utilities.network.socket") as mock_socket:
        mock_socket.socket().getsockname.return_value = [0, 8888]
        assert find_free_network_port() == 8888

        with pytest.raises(RuntimeError, match="Couldn't find a free port."):
            find_free_network_port()

        mock_socket.socket().getsockname.return_value = [0, 9999]
        assert find_free_network_port() == 9999


@mock.patch("lightning.app.utilities.network.socket")
@pytest.mark.parametrize(
    "patch_constants",
    [{"LIGHTNING_CLOUDSPACE_HOST": "any", "LIGHTNING_CLOUDSPACE_EXPOSED_PORT_COUNT": 10}],
    indirect=True,
)
def test_find_free_network_port_cloudspace(_, patch_constants):
    """Tests that `find_free_network_port` gives expected outputs and raises if a free port couldn't be found when
    cloudspace env variables are set."""
    ports = set()
    num_ports = 0

    with pytest.raises(RuntimeError, match="All 10 ports are already in use."):
        for _ in range(11):
            ports.add(find_free_network_port())
            num_ports = num_ports + 1

    # Check that all ports are unique
    assert len(ports) == num_ports

    # Shouldn't use the APP_SERVER_PORT
    assert constants.APP_SERVER_PORT not in ports


def test_lightning_client_retry_enabled():
    client = LightningClient()  # default: retry=True
    assert hasattr(client.auth_service_get_user_with_http_info, "__wrapped__")

    client = LightningClient(retry=False)
    assert not hasattr(client.auth_service_get_user_with_http_info, "__wrapped__")

    client = LightningClient(retry=True)
    assert hasattr(client.auth_service_get_user_with_http_info, "__wrapped__")
