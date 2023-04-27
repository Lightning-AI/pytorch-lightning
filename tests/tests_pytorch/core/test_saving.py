import pytest
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.demos.boring_classes import BoringModel
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
    "map_location", (None, "cpu", torch.device("cpu"), lambda storage, loc: storage, {"cpu": "cpu"})
)
def test_load_from_checkpoint_map_location_cpu(tmp_path, map_location):
    create_boring_checkpoint(tmp_path, BoringModel(), accelerator="cpu")
    model = BoringModel.load_from_checkpoint(f"{tmp_path}/checkpoint.ckpt", map_location=map_location)
    assert model.device.type == "cpu"


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "map_location", (None, "cuda", torch.device("cuda"), lambda storage, loc: storage.cuda(), {"cpu": "cuda"})
)
def test_load_from_checkpoint_map_location_gpu(tmp_path, map_location):
    create_boring_checkpoint(tmp_path, BoringModel(), accelerator="cuda")
    model = BoringModel.load_from_checkpoint(f"{tmp_path}/checkpoint.ckpt", map_location=map_location)
    assert model.device.type == "cuda"


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("map_location", ("cpu", torch.device("cpu"), lambda storage, loc: storage, {"cuda": "cpu"}))
def test_load_from_checkpoint_map_location_gpu_to_cpu(tmp_path, map_location):
    create_boring_checkpoint(tmp_path, BoringModel(), accelerator="cpu")
    model = BoringModel.load_from_checkpoint(f"{tmp_path}/checkpoint.ckpt", map_location=map_location)
    assert model.device.type == "cpu"


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "map_location", ("cuda", torch.device("cuda"), lambda storage, loc: storage.cuda(), {"cpu": "cuda"})
)
def test_load_from_checkpoint_map_location_cpu_to_gpu(tmp_path, map_location):
    create_boring_checkpoint(tmp_path, BoringModel(), accelerator="cpu")
    model = BoringModel.load_from_checkpoint(f"{tmp_path}/checkpoint.ckpt", map_location=map_location)
    assert model.device.type == "cuda"
