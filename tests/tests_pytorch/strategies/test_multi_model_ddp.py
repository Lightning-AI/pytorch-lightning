# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from unittest import mock

import pytest
import torch
from torch.nn.parallel.distributed import DistributedDataParallel

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import MultiModelDDPStrategy
from lightning.pytorch.trainer import seed_everything
from tests_pytorch.helpers.datamodules import MNISTDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import GenerationModel

from ..helpers.advanced_models import Generator, Discriminator


@RunIf(min_cuda_gpus=2, standalone=True, sklearn=False)
def test_multi_gpu_with_multi_model_ddp_fit_only(tmp_path):
    dm = MNISTDataModule()
    model = GenerationModel()
    trainer = Trainer(
        default_root_dir=tmp_path, max_epochs=1, accelerator="gpu", devices=2, strategy=MultiModelDDPStrategy()
    )
    trainer.fit(model, datamodule=dm)


@RunIf(min_cuda_gpus=2, standalone=True, sklearn=False)
def test_multi_gpu_with_multi_model_ddp_test_only(tmp_path):
    dm = MNISTDataModule()
    model = GenerationModel()
    trainer = Trainer(
        default_root_dir=tmp_path, max_epochs=1, accelerator="gpu", devices=2, strategy=MultiModelDDPStrategy()
    )
    trainer.test(model, datamodule=dm)


@RunIf(min_cuda_gpus=2, standalone=True, sklearn=False)
def test_multi_gpu_multi_model_ddp_fit_test(tmp_path):
    seed_everything(4321)
    dm = MNISTDataModule()
    model = GenerationModel()
    trainer = Trainer(
        default_root_dir=tmp_path, max_epochs=1, accelerator="gpu", devices=2, strategy=MultiModelDDPStrategy()
    )

    before = trainer.test(model, datamodule=dm)
    trainer.fit(model, datamodule=dm)
    after = trainer.test(model, datamodule=dm)

    before_g, before_d = before[0]["test/g_loss"], before[0]["test/d_loss"]
    after_g, after_d = after[0]["test/g_loss"], after[0]["test/d_loss"]

    # need to find a more appropriate assertion than this; the discriminator converges more easily than the generator.
    assert after_d <= before_d


@RunIf(min_cuda_gpus=2, standalone=True)
@pytest.mark.parametrize("precision", ["16-mixed", "32-true"])
def test_multi_model_ddp_wrapper(tmp_path, precision):
    class WeirdModule(torch.nn.Module):
        def _save_to_state_dict(self, destination, prefix, keep_vars):
            return {"something": "something"}

    class WeirdModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(32, 2)
            self.weird_module = WeirdModule()

            self._ddp_params_and_buffers_to_ignore = ["something"]

        def forward(self, x):
            return self.layer(x)

    class CustomMultiModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.weird_model = WeirdModel()

    class CustomCallback(Callback):
        def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            assert isinstance(trainer.strategy, MultiModelDDPStrategy)

            for name, model in trainer.strategy.model.named_children():
                assert isinstance(model, DistributedDataParallel)

                if name == "weird_model":
                    expected = {"something"}
                    assert model.parameters_to_ignore == expected
                    assert model.module._ddp_params_and_buffers_to_ignore == ["something"]

    model = CustomMultiModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        fast_dev_run=True,
        precision=precision,
        strategy=MultiModelDDPStrategy(),
        accelerator="gpu",
        devices=2,
        callbacks=CustomCallback(),
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)


@RunIf(min_cuda_gpus=2)
def test_multi_model_ddp_all_dataloaders_passed_to_fit(tmp_path):
    """Make sure MultiModelDDPStrategy works with dataloaders passed to fit()"""
    model = GenerationModel()

    trainer = Trainer(
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        max_epochs=1,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        accelerator="gpu",
        devices=[0, 1],
        strategy=MultiModelDDPStrategy(),
    )

    trainer.fit(model, train_dataloaders=model.train_dataloader(), val_dataloaders=model.val_dataloader())


# class GeneratorWithUnused(Generator):
#     def __init__(self, latent_dim, img_shape):
#         super().__init__(latent_dim, img_shape)
#         self.unused = torch.nn.Linear(latent_dim, latent_dim)

#     def forward(self, z):
#         z = self.unused(z)
#         z = z.detach()
#         return super().forward(z)


# class UnusedParametersModel(GenerationModel):
#     def __init__(self):
#         super().__init__()
#         self.generator = GeneratorWithUnused(latent_dim=128, img_shape=(1, 28, 28))

#     def training_step(self, batch, batch_idx):
#         return super().training_step(batch, batch_idx)


# @RunIf(standalone=True)
# def test_find_unused_parameters_multi_model_ddp_raises():
#     trainer = Trainer(accelerator="cpu", devices=1, strategy=MultiModelDDPStrategy(), max_steps=2, logger=False)
#     with pytest.raises(RuntimeError, match="It looks like your LightningModule has parameters that were not used in"):
#         trainer.fit(UnusedParametersModel())


class MultiModelDDPCPU(GenerationModel):
    def on_train_start(self) -> None:
        # make sure that the model is on CPU when training
        assert self.device == torch.device("cpu")


@RunIf(skip_windows=True)
def test_multi_model_ddp_with_cpu():
    """Tests if device is set correctly when training for MultiModelDDPStrategy."""
    trainer = Trainer(devices=2, strategy=MultiModelDDPStrategy(), accelerator="cpu", fast_dev_run=True)
    # assert strategy attributes for device setting
    assert isinstance(trainer.strategy, MultiModelDDPStrategy)
    assert trainer.strategy.root_device == torch.device("cpu")
    model = MultiModelDDPCPU()
    trainer.fit(model)


class CheckOptimizerDeviceModel(GenerationModel):
    def configure_optimizers(self):
        assert all(param.device.type == "cuda" for param in self.parameters())
        return super().configure_optimizers()


@RunIf(min_cuda_gpus=1)
def test_multi_model_ddp_parameters_on_device_for_optimizer():
    """Test that the strategy has moved the parameters to the device by the time the optimizer gets created."""
    model = CheckOptimizerDeviceModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        fast_dev_run=1,
        accelerator="gpu",
        devices=1,
        strategy=MultiModelDDPStrategy(),
    )
    trainer.fit(model)


class MultiModelDDPGPU(GenerationModel):
    def on_train_start(self) -> None:
        # make sure that the model is on GPU when training
        assert self.device == torch.device(f"cuda:{self.trainer.strategy.local_rank}")
        self.start_cuda_memory = torch.cuda.memory_allocated()


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
def test_multi_model_ddp_with_2_gpus():
    """Tests if device is set correctly when training and after teardown for MultiModelDDPStrategy."""
    trainer = Trainer(
        accelerator="gpu",
        devices=2,
        strategy=MultiModelDDPStrategy(),
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    # assert strategy attributes for device setting
    assert isinstance(trainer.strategy, MultiModelDDPStrategy)
    local_rank = trainer.strategy.local_rank
    assert trainer.strategy.root_device == torch.device(f"cuda:{local_rank}")

    model = MultiModelDDPGPU()

    trainer.fit(model)

    # assert after training, model is moved to CPU and memory is deallocated
    assert model.device == torch.device("cpu")
    cuda_memory = torch.cuda.memory_allocated()
    assert cuda_memory < model.start_cuda_memory


@RunIf(min_cuda_gpus=4, standalone=True)
@mock.patch("torch.distributed.barrier")
def test_multi_model_ddp_barrier_non_consecutive_device_ids(barrier_mock, tmp_path):
    """Test correct usage of barriers when device ids do not start at 0 or are not consecutive."""
    model = GenerationModel()
    gpus = [1, 3]
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_steps=1,
        accelerator="gpu",
        devices=gpus,
        strategy=MultiModelDDPStrategy(),
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)
    barrier_mock.assert_any_call(device_ids=[gpus[trainer.local_rank]])
