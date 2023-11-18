import base64
import multiprocessing as mp
import os
from unittest.mock import ANY, MagicMock

import pytest
from lightning.app.components.serve import serve
from lightning.app.testing.helpers import _RunIf
from lightning.app.utilities.imports import _is_numpy_available, _is_torch_available
from lightning.app.utilities.network import _configure_session, find_free_network_port
from tests_app import _PROJECT_ROOT

if _is_numpy_available():
    import numpy as np

if _is_torch_available():
    import torch


class ImageServer(serve.ModelInferenceAPI):
    def build_model(self):
        return lambda x: x

    def predict(self, image):
        image = self.model(image)
        return torch.from_numpy(np.asarray(image))


def target_fn(port, workers):
    image_server = ImageServer(input="image", output="image", port=port, workers=workers)
    image_server.run()


@pytest.mark.xfail(strict=False, reason="test has been ignored for a while and seems not to be working :(")
@pytest.mark.skipif(not (_is_torch_available() and _is_numpy_available()), reason="Missing torch and numpy")
@pytest.mark.parametrize("workers", [0])
# avoid the error: Failed to establish a new connection: [WinError 10061] No connection could be made because the
# target machine actively refused it
@_RunIf(skip_windows=True)
def test_model_inference_api(workers):
    port = find_free_network_port()
    process = mp.Process(target=target_fn, args=(port, workers))
    process.start()

    image_path = os.path.join(_PROJECT_ROOT, "docs/source-app/_static/images/logo.png")
    with open(image_path, "rb") as f:
        imgstr = base64.b64encode(f.read()).decode("UTF-8")

    session = _configure_session()
    res = session.post(f"http://127.0.0.1:{port}/predict", params={"data": imgstr})
    process.terminate()
    # TODO: Investigate why this doesn't match exactly `imgstr`.
    assert res.json()
    process.kill()


class EmptyServer(serve.ModelInferenceAPI):
    def build_model(self):
        return lambda x: x

    def serialize(self, x):
        return super().serialize(x)

    def deserialize(self, x):
        return super().deserialize(x)

    def predict(self, x):
        return super().predict(x)


def test_model_inference_api_mock(monkeypatch):
    monkeypatch.setattr(serve, "uvicorn", MagicMock())
    comp = EmptyServer()
    comp.run()
    serve.uvicorn.run.assert_called_once_with(app=ANY, host=comp.host, port=comp.port, log_level="error")

    with pytest.raises(Exception, match="Only input in"):
        EmptyServer(input="something")

    with pytest.raises(Exception, match="Only output in"):
        EmptyServer(output="something")
