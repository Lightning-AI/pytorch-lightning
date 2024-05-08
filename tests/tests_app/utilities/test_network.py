from http.client import HTTPMessage
from unittest import mock

import pytest
from lightning.app.core import constants
from lightning.app.utilities.network import HTTPClient, find_free_network_port


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


@mock.patch("urllib3.connectionpool.HTTPConnectionPool._get_conn")
def test_http_client_retry_post(getconn_mock):
    getconn_mock.return_value.getresponse.side_effect = [
        mock.Mock(status=500, msg=HTTPMessage()),
        mock.Mock(status=429, msg=HTTPMessage()),
        mock.Mock(status=200, msg=HTTPMessage()),
    ]

    client = HTTPClient(base_url="http://test.url")
    r = client.post("/test")
    r.raise_for_status()

    assert getconn_mock.return_value.request.mock_calls == [
        mock.call("POST", "/test", body=None, headers=mock.ANY),
        mock.call("POST", "/test", body=None, headers=mock.ANY),
        mock.call("POST", "/test", body=None, headers=mock.ANY),
    ]


@mock.patch("urllib3.connectionpool.HTTPConnectionPool._get_conn")
def test_http_client_retry_get(getconn_mock):
    getconn_mock.return_value.getresponse.side_effect = [
        mock.Mock(status=500, msg=HTTPMessage()),
        mock.Mock(status=429, msg=HTTPMessage()),
        mock.Mock(status=200, msg=HTTPMessage()),
    ]

    client = HTTPClient(base_url="http://test.url")
    r = client.get("/test")
    r.raise_for_status()

    assert getconn_mock.return_value.request.mock_calls == [
        mock.call("GET", "/test", body=None, headers=mock.ANY),
        mock.call("GET", "/test", body=None, headers=mock.ANY),
        mock.call("GET", "/test", body=None, headers=mock.ANY),
    ]
