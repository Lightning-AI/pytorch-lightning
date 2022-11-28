from pathlib import Path

from lightning_app.core.app import LightningApp
from lightning_app.components import TritonServer
from lightning_app.components.serve import triton_server


class TestTritonServer(TritonServer):
    def infer(self, request):
        return {"prediction": 0}


def test_model_repository(monkeypatch, tmpdir):
    p = Path(tmpdir)
    work = TestTritonServer()

    # creating the app object so _LightningappRef will be populated
    _ = LightningApp(work)

    monkeypatch.setattr(triton_server.Path, "cwd", lambda *args, **kwargs: p)
    work._setup_model_repository()
    assert p.joinpath("__model_repository").is_dir()
    modeldir = p / "__model_repository/lightning-triton/1"
    assert modeldir.is_dir()
    assert (modeldir / "__lightningapp_triton_model_file.py").exists()
    assert (modeldir / "__lightningapp_triton_model_file.py").read_text() == triton_server.triton_model_file_template
    assert (modeldir / "__lightning_work.pkl").exists()
    assert (modeldir.parent / "config.pbtxt").exists()


def test_get_config_file():
    work = TestTritonServer()

    # creating the app object so _LightningappRef will be populated
    _ = LightningApp(work)

    # input and output type with one property
    work._input_type = None
    work._output_type = None
    work._get_config_file()

    # input and output type with multiple properties of different types
    work._input_type = None
    work._output_type = None
    work._get_config_file()


def test_attach_triton_proxy_fn():
    pass
