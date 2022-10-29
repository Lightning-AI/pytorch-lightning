from lightning_app.utilities.network import find_free_network_port


def test_port():
    assert find_free_network_port()
