# Copyright The PyTorch Lightning team.
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
from unittest import mock

import pytest
import torch
from torch.nn.parallel.distributed import DistributedDataParallel

import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.demos.boring_classes import BoringModel
from pytorch_lightning.strategies import DDPStrategy
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel


@RunIf(min_cuda_gpus=2, standalone=True)
def test_multi_gpu_model_ddp_fit_only(tmpdir):
    dm = ClassifDataModule()
    model = ClassificationModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, accelerator="gpu", devices=2, strategy="ddp")
    trainer.fit(model, datamodule=dm)


@RunIf(min_cuda_gpus=2, standalone=True)
def test_multi_gpu_model_ddp_test_only(tmpdir):
    dm = ClassifDataModule()
    model = ClassificationModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, accelerator="gpu", devices=2, strategy="ddp")
    trainer.test(model, datamodule=dm)


@RunIf(min_cuda_gpus=2, standalone=True)
def test_multi_gpu_model_ddp_fit_test(tmpdir):
    seed_everything(4321)
    dm = ClassifDataModule()
    model = ClassificationModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, accelerator="gpu", devices=2, strategy="ddp")
    trainer.fit(model, datamodule=dm)
    result = trainer.test(model, datamodule=dm)

    for out in result:
        assert out["test_acc"] > 0.7


@RunIf(skip_windows=True)
def test_torch_distributed_backend_invalid(cuda_count_2, tmpdir):
    """This test set `undefined` as torch backend and should raise an `Backend.UNDEFINED` ValueError."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        strategy=DDPStrategy(process_group_backend="undefined"),
        accelerator="cuda",
        devices=2,
        logger=False,
    )
    with pytest.raises(ValueError, match="Invalid backend: 'undefined'"):
        trainer.fit(model)


@RunIf(skip_windows=True)
@mock.patch("torch.cuda.set_device")
def test_ddp_torch_dist_is_available_in_setup(mock_set_device, cuda_count_1, tmpdir):
    """Test to ensure torch distributed is available within the setup hook using ddp."""

    class TestModel(BoringModel):
        def setup(self, stage: str) -> None:
            assert torch.distributed.is_initialized()
            raise SystemExit()

    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        strategy=DDPStrategy(process_group_backend="gloo"),
        accelerator="gpu",
        devices=1,
    )
    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_cuda_gpus=2, min_torch="1.8.1", standalone=True)
@pytest.mark.parametrize("precision", (16, 32))
def test_ddp_wrapper(tmpdir, precision):
    """Test parameters to ignore are carried over for DDP."""

    class WeirdModule(torch.nn.Module):
        def _save_to_state_dict(self, destination, prefix, keep_vars):
            return {"something": "something"}

    class CustomModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.weird_module = WeirdModule()

            # should be skip.
            self._ddp_params_and_buffers_to_ignore = ["something"]

    class CustomCallback(Callback):
        def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            assert isinstance(trainer.strategy.model, DistributedDataParallel)
            assert trainer.strategy.model.parameters_to_ignore == ["module.something"]
            assert trainer.strategy.model.module._ddp_params_and_buffers_to_ignore == ["module.something"]

    model = CustomModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        precision=precision,
        strategy="ddp",
        accelerator="gpu",
        devices=2,
        callbacks=CustomCallback(),
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)


@pytest.mark.parametrize(
    ["process_group_backend", "device_str", "expected_process_group_backend"],
    [
        pytest.param("foo", "cpu", "foo"),
        pytest.param("foo", "cuda:0", "foo"),
        pytest.param(None, "cuda:0", "nccl"),
        pytest.param(None, "cpu", "gloo"),
    ],
)
def test_ddp_process_group_backend(process_group_backend, device_str, expected_process_group_backend):
    """Test settings for process group backend."""

    class MockDDPStrategy(DDPStrategy):
        def __init__(self, root_device, process_group_backend):
            self._root_device = root_device
            super().__init__(process_group_backend=process_group_backend)

        @property
        def root_device(self):
            return self._root_device

    strategy = MockDDPStrategy(process_group_backend=process_group_backend, root_device=torch.device(device_str))
    assert strategy._get_process_group_backend() == expected_process_group_backend


@pytest.mark.parametrize(
    "strategy_name,expected_ddp_kwargs",
    [
        ("ddp", {}),
        ("ddp_find_unused_parameters_false", {"find_unused_parameters": False}),
    ],
)
def test_ddp_kwargs_from_registry(strategy_name, expected_ddp_kwargs):
    trainer = Trainer(strategy=strategy_name)
    assert trainer.strategy._ddp_kwargs == expected_ddp_kwargs
