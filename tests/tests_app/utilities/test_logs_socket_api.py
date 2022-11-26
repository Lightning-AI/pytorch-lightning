from unittest import mock

from lightning_app.utilities.logs_socket_api import _ClusterLogsSocketAPI


def test_cluster_logs_socket_url1():
    assert (
        _ClusterLogsSocketAPI._cluster_logs_socket_url(
            "example.org", "my-cluster", 1661100000.123, 1661101000.321, 10, "TOKEN"
        )
        == "wss://example.org/v1/core/clusters/my-cluster/logs?start=1661100000.123&"
        "limit=10&token=TOKEN&end=1661101000.321"
    )


def test_cluster_logs_socket_url2():
    assert (
        _ClusterLogsSocketAPI._cluster_logs_socket_url("example.org", "my-cluster", 1661100000, None, 10, "TOKEN")
        == "wss://example.org/v1/core/clusters/my-cluster/logs?start=1661100000&limit=10&token=TOKEN&follow=true"
    )


def test_cluster_logs_socket_api():
    api_client = mock.Mock()
    api_client.configuration.host = "https://example.com"
    api_client.call_api.return_value.token = "TOKEN"
    cluster_logs_api = _ClusterLogsSocketAPI(api_client)

    def on_message_func():
        return None

    web_socket_app = cluster_logs_api.create_cluster_logs_socket(
        "my-cluster", 1661100000, 1661101000, 10, on_message_func
    )

    assert web_socket_app.on_message == on_message_func
