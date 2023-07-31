from unittest import mock

from lightning.app import CloudCompute
from lightning.app.launcher.lightning_hybrid_backend import CloudHybridBackend


@mock.patch("lightning.app.launcher.lightning_backend.LightningClient")
def test_backend_selection(client_mock):
    cloud_backend = CloudHybridBackend("", queue_id="")
    work = mock.MagicMock()
    work.cloud_compute = CloudCompute()
    assert cloud_backend._get_backend(work) == cloud_backend.backends["multiprocess"]
    work.cloud_compute = CloudCompute("gpu")
    assert cloud_backend._get_backend(work) == cloud_backend.backends["cloud"]
