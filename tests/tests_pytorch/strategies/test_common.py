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
from unittest.mock import Mock

import pytest
import torch
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning.pytorch import Trainer
from lightning.pytorch.plugins import DoublePrecision, HalfPrecision, Precision
from lightning.pytorch.strategies import SingleDeviceStrategy

from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel


@pytest.mark.parametrize(
    "trainer_kwargs",
    [
        pytest.param({"accelerator": "gpu", "devices": 1}, marks=RunIf(min_cuda_gpus=1)),
        pytest.param({"strategy": "ddp_spawn", "accelerator": "gpu", "devices": 2}, marks=RunIf(min_cuda_gpus=2)),
        pytest.param({"accelerator": "mps", "devices": 1}, marks=RunIf(mps=True)),
    ],
)
@RunIf(sklearn=True)
def test_evaluate(tmpdir, trainer_kwargs):
    dm = ClassifDataModule()
    model = ClassificationModel()
    trainer = Trainer(
        default_root_dir=tmpdir, max_epochs=2, limit_train_batches=10, limit_val_batches=10, **trainer_kwargs
    )

    trainer.fit(model, datamodule=dm)
    assert "ckpt" in trainer.checkpoint_callback.best_model_path

    old_weights = model.layer_0.weight.clone().detach().cpu()

    trainer.validate(datamodule=dm)
    trainer.test(datamodule=dm)

    # make sure weights didn't change
    new_weights = model.layer_0.weight.clone().detach().cpu()
    torch.testing.assert_close(old_weights, new_weights)


@RunIf(min_torch="1.13")
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda:0", marks=RunIf(min_cuda_gpus=1)),
        pytest.param("mps:0", marks=RunIf(mps=True)),
    ],
)
@pytest.mark.parametrize(
    ("precision", "dtype"),
    [
        (Precision(), torch.float32),
        pytest.param(DoublePrecision(), torch.float64, marks=RunIf(mps=False)),
        (HalfPrecision("16-true"), torch.float16),
        pytest.param(HalfPrecision("bf16-true"), torch.bfloat16, marks=RunIf(bf16_cuda=True)),
    ],
)
@pytest.mark.parametrize("empty_init", [None, True, False])
def test_module_init_context(device, precision, dtype, empty_init, monkeypatch):
    """Test that the module under the init-module-context gets moved to the right device and dtype."""
    init_mock = Mock()
    monkeypatch.setattr(torch.Tensor, "uniform_", init_mock)

    device = torch.device(device)
    strategy = SingleDeviceStrategy(device=device, precision_plugin=precision)  # surrogate class to test base class
    with strategy.tensor_init_context(empty_init=empty_init):
        module = torch.nn.Linear(2, 2)

    expected_device = device if _TORCH_GREATER_EQUAL_2_0 else torch.device("cpu")
    assert module.weight.device == module.bias.device == expected_device
    assert module.weight.dtype == module.bias.dtype == dtype
    if not empty_init:
        init_mock.assert_called()
    else:
        init_mock.assert_not_called()
