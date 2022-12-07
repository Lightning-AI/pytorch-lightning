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
import tempfile
from unittest import mock

import pytest
import torch
from tests_lite.helpers.models import RandomDataset
from tests_lite.helpers.runif import RunIf
from torch.utils.data import DataLoader

from lightning_lite import LightningLite
from lightning_lite.plugins import FSDPPrecision
from lightning_lite.strategies import FSDPStrategy
from lightning_lite.utilities.imports import _TORCH_GREATER_EQUAL_1_12

if _TORCH_GREATER_EQUAL_1_12:
    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import wrap


def _get_model(manual_wrapping=False):
    model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    if not manual_wrapping:
        return model

    for i, layer in enumerate(model):
        if i % 2 == 0:
            model[i] = wrap(layer)
    return model


def _step(lite, model, batch):
    forward_module = model._forward_module
    original_module = model.module
    assert isinstance(forward_module, FullyShardedDataParallel)
    assert isinstance(lite._precision, FSDPPrecision)

    precision = torch.float16 if lite._precision.precision == 16 else torch.bfloat16
    assert forward_module.mixed_precision.param_dtype == precision
    assert forward_module.mixed_precision.reduce_dtype == precision
    assert forward_module.mixed_precision.buffer_dtype == precision

    for layer_num in [0, 2]:
        assert isinstance(original_module[layer_num], FullyShardedDataParallel)
        assert original_module[layer_num].mixed_precision.param_dtype == precision
        assert original_module[layer_num].mixed_precision.reduce_dtype == precision
        assert original_module[layer_num].mixed_precision.buffer_dtype == precision

    output = model(batch)
    loss = torch.nn.functional.mse_loss(output, torch.ones_like(output))
    return loss


def _assert_save_equality(lite, model, ckpt_path):
    current_state_dict = lite._strategy.get_module_state_dict(model)

    checkpoint = lite.load(ckpt_path)
    loaded_model = _get_model()
    loaded_model.load_state_dict(checkpoint)

    # model parameters are identical after loading
    for current_param, loaded_param in zip(current_state_dict.values(), loaded_model.state_dict().values()):
        assert torch.allclose(current_param.float().cpu(), loaded_param.cpu())


def _custom_auto_wrap_policy(module, recurse, unwrapped_params: int, min_num_params: int = int(1e8)) -> bool:
    return unwrapped_params >= 2


@RunIf(min_cuda_gpus=1, skip_windows=True, standalone=True, min_torch="1.12")
@pytest.mark.parametrize("precision", (16, pytest.param("bf16", marks=RunIf(bf16_cuda=True))))
@pytest.mark.parametrize("manual_wrapping", [True, False])
def test_fsdp_train_save_load(manual_wrapping, precision):
    """Test FSDP training, saving and loading with different wrapping and precision settings."""
    strategy = FSDPStrategy(
        auto_wrap_policy=_custom_auto_wrap_policy,
        activation_checkpointing=[torch.nn.Linear],
    )
    lite = LightningLite(accelerator="cuda", strategy=strategy, devices=2, precision=precision)
    lite.launch()

    with lite.sharded_model():
        model = _get_model(manual_wrapping)

    dataloader = DataLoader(RandomDataset(32, 64))

    # model needs to be set up first in FSDP
    model = lite.setup_module(model)

    # get parameters on the wrapped model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # optimizer nees to be set up independently
    optimizer = lite.setup_optimizers(optimizer)

    dataloader = lite.setup_dataloaders(dataloader)
    model.train()

    data_iter = iter(dataloader)
    batch = next(data_iter)
    loss = _step(lite, model, batch)
    lite.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

    with tempfile.TemporaryFile() as ckpt_path:
        ckpt_path = lite.broadcast(str(ckpt_path))
        lite._strategy.save_checkpoint(model.state_dict(), ckpt_path)

    _assert_save_equality(lite, model, ckpt_path)


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True, min_torch="1.12")
@pytest.mark.parametrize("move_to_device", [True, False])
@mock.patch("lightning_lite.wrappers._LiteModule")
def test_setup_module_move_to_device(lite_module_mock, move_to_device):
    """Test that `move_to_device` does nothing, FSDP decides which device parameters get moved to which device
    (sharding)."""
    strategy = FSDPStrategy(auto_wrap_policy=_custom_auto_wrap_policy)
    lite = LightningLite(accelerator="cuda", devices=2, strategy=strategy)
    lite.launch()

    model = torch.nn.Linear(10, 10, bias=False)  # total params: 10 * 10 = 100
    lite_model = lite.setup_module(model, move_to_device=move_to_device)
    lite_module_mock.assert_not_called()

    assert list(param.device for param in model.parameters()) == []
    assert len(list(lite_model.parameters())) == 1

    # the linear layer got sharded and each part is on the expected device
    assert next(lite_model.parameters()).device == torch.device("cuda", lite.local_rank)
    assert next(lite_model.parameters()).numel() == 50

    # The _DeviceDtypeModuleMixin currently can't represent the device in a meaningful way for sharded models
    assert lite_model.device == torch.device("cpu")
    assert lite.device == torch.device("cuda", lite.local_rank)
