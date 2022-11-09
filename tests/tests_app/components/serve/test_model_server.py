import base64
import multiprocessing as mp
import os
from unittest.mock import ANY, MagicMock

import pytest

from lightning_app.components.serve import model_server
from lightning_app.utilities.imports import _is_numpy_available, _is_torch_available
from lightning_app.utilities.network import _configure_session, find_free_network_port


class SimpleServer(model_server.ModelServer):
    def __init__(self, port):
        super().__init__(port=port)
        self._model = None

    def setup(self):
        self._model = lambda x: x

    def predict(self, data):
        return self._model(data)


def target_fn(port):
    image_server = SimpleServer(port=port)
    image_server.run()


def test_model_server_component():
    port = find_free_network_port()
    process = mp.Process(target=target_fn, args=(port,))
    process.start()
    session = _configure_session()
    res = session.post(f"http://127.0.0.1:{port}/predict", data='"test"')
    process.terminate()
    assert res.text == '"test"'
