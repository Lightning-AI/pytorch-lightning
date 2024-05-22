from unittest.mock import ANY, Mock

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel

from tests_pytorch.conftest import mock_cuda_count, mock_mps_count
from tests_pytorch.helpers.runif import RunIf


def create_boring_checkpoint(tmp_path, model, accelerator="cuda"):
    checkpoint_callback = ModelCheckpoint(dirpath=tmp_path, filename="checkpoint")
    trainer = pl.Trainer(
        default_root_dir=tmp_path,
        logger=False,
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


def test_load_from_checkpoint_warn_on_empty_state_dict(tmp_path):
    """Test that checkpoints can be loaded with an empty state dict and that the appropriate warning is raised."""
    create_boring_checkpoint(tmp_path, BoringModel(), accelerator="cpu")
    # Now edit so the state_dict is empty
    checkpoint = torch.load(tmp_path / "checkpoint.ckpt")
    checkpoint["state_dict"] = {}
    torch.save(checkpoint, tmp_path / "checkpoint.ckpt")

    with pytest.warns(UserWarning, match="contains no parameters"):
        model = BoringModel.load_from_checkpoint(tmp_path / "checkpoint.ckpt", strict=False)
    assert model.device.type == "cpu"


@pytest.mark.parametrize(
    ("strict", "strict_loading", "expected"),
    [
        (None, None, True),
        (None, True, True),
        (None, False, False),
        (True, None, True),
        (True, True, True),
        (True, False, "error"),
        (False, None, False),
        (False, True, "error"),
        (False, False, False),
    ],
)
def test_load_from_checkpoint_strict(strict, strict_loading, expected, tmp_path):
    """Test that strict loading works both with the `strict` argument and the model's `strict_loading` attribute."""

    class LoadingModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.strict_loading = strict_loading

    LoadingModel.load_state_dict = Mock()

    create_boring_checkpoint(tmp_path, LoadingModel(), accelerator="cpu")

    if expected == "error":
        with pytest.raises(ValueError, match="in conflict with .*strict_loading"):
            LoadingModel.load_from_checkpoint(tmp_path / "checkpoint.ckpt", strict=strict)
    else:
        model = LoadingModel.load_from_checkpoint(tmp_path / "checkpoint.ckpt", strict=strict)
        model.load_state_dict.assert_called_once_with(ANY, strict=expected)
