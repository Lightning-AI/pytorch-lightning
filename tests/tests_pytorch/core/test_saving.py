import pytest
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
from tests_pytorch.conftest import mock_cuda_count, mock_mps_count
from tests_pytorch.helpers.runif import RunIf


def create_boring_checkpoint(tmp_path, model, accelerator="cuda"):
    checkpoint_callback = ModelCheckpoint(dirpath=tmp_path, filename="checkpoint")
    trainer = pl.Trainer(
        devices=1,
        accelerator=accelerator,
        max_epochs=1,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)


@pytest.mark.parametrize(
    "accelerator",
    [
        "cpu",
        pytest.param("cuda", marks=RunIf(min_cuda_gpus=1)),
        pytest.param("mps", marks=RunIf(mps=True)),
    ],
)
def test_load_from_checkpoint_map_location_automatic(accelerator, tmp_path, monkeypatch):
    """Test that the default `map_location` provided by Lightning moves parameters to CPU if the accelerator is not
    available at the time of loading."""
    create_boring_checkpoint(tmp_path, BoringModel(), accelerator=accelerator)

    # The checkpoint contains tensors with storage tag on the accelerator
    checkpoint = torch.load(f"{tmp_path}/checkpoint.ckpt")
    assert checkpoint["state_dict"]["layer.weight"].device.type.startswith(accelerator)

    # Pretend that the accelerator is not available
    mock_cuda_count(monkeypatch, 0)
    mock_mps_count(monkeypatch, 0)

    model = BoringModel.load_from_checkpoint(f"{tmp_path}/checkpoint.ckpt")
    _ = BoringDataModule.load_from_checkpoint(f"{tmp_path}/checkpoint.ckpt")
    assert model.device.type == "cpu"
    assert model.layer.weight.device.type == "cpu"


@pytest.mark.parametrize(
    "map_location", [None, "cpu", torch.device("cpu"), lambda storage, loc: storage, {"cpu": "cpu"}]
)
def test_load_from_checkpoint_map_location_cpu(tmp_path, map_location):
    create_boring_checkpoint(tmp_path, BoringModel(), accelerator="cpu")
    model = BoringModel.load_from_checkpoint(f"{tmp_path}/checkpoint.ckpt", map_location=map_location)
    _ = BoringDataModule.load_from_checkpoint(f"{tmp_path}/checkpoint.ckpt", map_location=map_location)
    assert model.device.type == "cpu"


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "map_location", [None, "cuda", torch.device("cuda"), lambda storage, loc: storage.cuda(), {"cpu": "cuda"}]
)
def test_load_from_checkpoint_map_location_gpu(tmp_path, map_location):
    create_boring_checkpoint(tmp_path, BoringModel(), accelerator="cuda")
    model = BoringModel.load_from_checkpoint(f"{tmp_path}/checkpoint.ckpt", map_location=map_location)
    _ = BoringDataModule.load_from_checkpoint(f"{tmp_path}/checkpoint.ckpt", map_location=map_location)
    assert model.device.type == "cuda"


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("map_location", ["cpu", torch.device("cpu"), lambda storage, loc: storage, {"cuda": "cpu"}])
def test_load_from_checkpoint_map_location_gpu_to_cpu(tmp_path, map_location):
    create_boring_checkpoint(tmp_path, BoringModel(), accelerator="cpu")
    model = BoringModel.load_from_checkpoint(f"{tmp_path}/checkpoint.ckpt", map_location=map_location)
    _ = BoringDataModule.load_from_checkpoint(f"{tmp_path}/checkpoint.ckpt", map_location=map_location)
    assert model.device.type == "cpu"


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "map_location", ["cuda", torch.device("cuda"), lambda storage, loc: storage.cuda(), {"cpu": "cuda"}]
)
def test_load_from_checkpoint_map_location_cpu_to_gpu(tmp_path, map_location):
    create_boring_checkpoint(tmp_path, BoringModel(), accelerator="cpu")
    model = BoringModel.load_from_checkpoint(f"{tmp_path}/checkpoint.ckpt", map_location=map_location)
    _ = BoringDataModule.load_from_checkpoint(f"{tmp_path}/checkpoint.ckpt", map_location=map_location)
    assert model.device.type == "cuda"


@RunIf(min_cuda_gpus=1)
def test_load_from_checkpoint_device_placement_with_extra_state(tmp_path):
    """Test that the device gets chosen based on the device of the saved tensors in the checkpoint."""

    class ExtraStateModel(BoringModel):
        def get_extra_state(self):
            return {"extra": "state"}  # state without tensors

        def set_extra_state(self, state):
            pass

    create_boring_checkpoint(tmp_path, ExtraStateModel(), accelerator="cuda")
    model = ExtraStateModel.load_from_checkpoint(f"{tmp_path}/checkpoint.ckpt", map_location=None)
    assert model.device.type == "cuda"
