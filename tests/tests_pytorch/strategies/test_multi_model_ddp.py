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

import pytest
import torch
from torch.multiprocessing import ProcessRaisedException

from lightning.pytorch import Trainer
from lightning.pytorch.strategies import MultiModelDDPStrategy
from lightning.pytorch.trainer import seed_everything
from tests_pytorch.helpers.advanced_models import BasicGAN
from tests_pytorch.helpers.runif import RunIf


@RunIf(min_cuda_gpus=2, standalone=True, sklearn=True)
def test_multi_gpu_with_multi_model_ddp_fit_only(tmp_path):
    dm = BasicGAN.train_dataloader()
    model = BasicGAN()
    trainer = Trainer(
        default_root_dir=tmp_path, max_epochs=1, accelerator="gpu", devices=-1, strategy=MultiModelDDPStrategy()
    )
    trainer.fit(model, datamodule=dm)


@RunIf(min_cuda_gpus=2, standalone=True, sklearn=True)
def test_multi_gpu_with_multi_model_ddp_predict_only(tmp_path):
    dm = BasicGAN.train_dataloader()
    model = BasicGAN()
    trainer = Trainer(
        default_root_dir=tmp_path, max_epochs=1, accelerator="gpu", devices=-1, strategy=MultiModelDDPStrategy()
    )
    trainer.predict(model, datamodule=dm)


@RunIf(min_cuda_gpus=2, standalone=True, sklearn=True)
def test_multi_gpu_multi_model_ddp_fit_predict(tmp_path):
    seed_everything(4321)
    dm = BasicGAN.train_dataloader()
    model = BasicGAN()
    trainer = Trainer(
        default_root_dir=tmp_path, max_epochs=1, accelerator="gpu", devices=-1, strategy=MultiModelDDPStrategy()
    )
    trainer.fit(model, datamodule=dm)
    trainer.predict(model, datamodule=dm)


class UnusedParametersBasicGAN(BasicGAN):
    def __init__(self):
        super().__init__()
        mnist_shape = (1, 28, 28)
        self.intermediate_layer = torch.nn.Linear(mnist_shape[-1], mnist_shape[-1])

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            img = self.intermediate_layer(batch[0])
            batch[0] = img  # modify the batch to use the intermediate layer result
        return super().training_step(batch, batch_idx)


@RunIf(standalone=True)
def test_find_unused_parameters_ddp_spawn_raises():
    """Test that the DDP strategy can change PyTorch's error message so that it's more useful for Lightning users."""
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        strategy=MultiModelDDPStrategy(),
        max_steps=2,
        logger=False,
    )
    with pytest.raises(
        ProcessRaisedException, match="It looks like your LightningModule has parameters that were not used in"
    ):
        trainer.fit(UnusedParametersBasicGAN())


@RunIf(standalone=True)
def test_find_unused_parameters_ddp_exception():
    """Test that the DDP strategy can change PyTorch's error message so that it's more useful for Lightning users."""
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        strategy=MultiModelDDPStrategy(),
        max_steps=2,
        logger=False,
    )
    with pytest.raises(RuntimeError, match="It looks like your LightningModule has parameters that were not used in"):
        trainer.fit(UnusedParametersBasicGAN())


class CheckOptimizerDeviceModel(BasicGAN):
    def configure_optimizers(self):
        assert all(param.device.type == "cuda" for param in self.parameters())
        super().configure_optimizers()


@RunIf(min_cuda_gpus=1)
def test_model_parameters_on_device_for_optimizer():
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


class BasicGANCPU(BasicGAN):
    def on_train_start(self) -> None:
        # make sure that the model is on CPU when training
        assert self.device == torch.device("cpu")


@RunIf(skip_windows=True)
def test_multi_model_ddp_with_cpu():
    """Tests if device is set correctly when training for MultiModelDDPStrategy."""
    trainer = Trainer(
        accelerator="cpu",
        devices=-1,
        strategy=MultiModelDDPStrategy(),
        fast_dev_run=True,
    )
    # assert strategy attributes for device setting
    assert isinstance(trainer.strategy, MultiModelDDPStrategy)
    assert trainer.strategy.root_device == torch.device("cpu")
    model = BasicGANCPU()
    trainer.fit(model)


class BasicGANGPU(BasicGAN):
    def on_train_start(self) -> None:
        # make sure that the model is on GPU when training
        assert self.device == torch.device(f"cuda:{self.trainer.strategy.local_rank}")
        self.start_cuda_memory = torch.cuda.memory_allocated()


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
def test_multi_model_ddp_with_gpus():
    """Tests if device is set correctly when training and after teardown for MultiModelDDPStrategy."""
    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        strategy=MultiModelDDPStrategy(),
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    # assert strategy attributes for device setting
    assert isinstance(trainer.strategy, MultiModelDDPStrategy)
    local_rank = trainer.strategy.local_rank
    assert trainer.strategy.root_device == torch.device(f"cuda:{local_rank}")

    model = BasicGANGPU()

    trainer.fit(model)

    # assert after training, model is moved to CPU and memory is deallocated
    assert model.device == torch.device("cpu")
    cuda_memory = torch.cuda.memory_allocated()
    assert cuda_memory < model.start_cuda_memory
