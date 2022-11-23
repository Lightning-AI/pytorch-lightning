import multiprocessing as mp

from lightning_app.components import Image, Number, PythonServer
from lightning_app.utilities.network import _configure_session, find_free_network_port


class SimpleServer(PythonServer):
    def __init__(self, port):
        super().__init__(port=port)
        self._model = None

    def setup(self):
        self._model = lambda x: x

    def predict(self, data):
        return {"prediction": self._model(data.payload)}


def target_fn(port):
    image_server = SimpleServer(port=port)
    image_server.run()


def test_python_server_component():
    port = find_free_network_port()
    process = mp.Process(target=target_fn, args=(port,))
    process.start()
    session = _configure_session()
    res = session.post(f"http://127.0.0.1:{port}/predict", json={"payload": "test"})
    process.terminate()
    assert res.json()["prediction"] == "test"


def test_image_sample_data():
    data = Image()._get_sample_data()
    assert isinstance(data, dict)
    assert "image" in data
    assert len(data["image"]) > 100


def test_number_sample_data():
    data = Number()._get_sample_data()
    assert isinstance(data, dict)
    assert "prediction" in data
    assert data["prediction"] == 463
