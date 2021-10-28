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
from unittest.mock import ANY, Mock

import pytest
import torch

from pytorch_lightning.lite import LightningLite
from pytorch_lightning.lite.wrappers import _LiteDataLoader, _LiteModule, _LiteOptimizer
from tests.helpers.runif import RunIf


class EmptyLite(LightningLite):
    def run(self):
        pass


def test_lite_module_wraps():
    """Test that the wrapped module is accessible via the property."""
    module = Mock()
    assert _LiteModule(module, Mock()).module is module


@RunIf(min_gpus=1)
@pytest.mark.parametrize(
    "precision, input_type, expected_type",
    [
        (32, torch.float16, torch.float32),
        (32, torch.float32, torch.float32),
        (32, torch.float64, torch.float32),
        (16, torch.float32, torch.float16),
        (16, torch.float64, torch.float16),
    ],
)
def test_lite_module_forward_conversion(precision, input_type, expected_type):
    """Test that the LiteModule performs autocasting on the input tensors and during forward()."""
    lite = EmptyLite(precision=precision, accelerator="gpu", devices=1)
    device = torch.device("cuda", 0)

    def check_autocast(forward_input):
        assert precision != 16 or torch.is_autocast_enabled()
        return forward_input

    module = Mock(wraps=torch.nn.Linear(1, 1), side_effect=check_autocast)
    lite_module = _LiteModule(module, lite._accelerator).to(device)
    out = lite_module(torch.rand(1, dtype=input_type, device=device))
    assert module.call_args[0][0].dtype == expected_type
    assert out.dtype == torch.get_default_dtype()


@pytest.mark.parametrize(
    "src_device, dest_device",
    [
        (torch.device("cpu"), torch.device("cpu")),
        pytest.param(torch.device("cpu"), torch.device("cuda", 0), marks=RunIf(min_gpus=1)),
        pytest.param(torch.device("cuda", 0), torch.device("cpu"), marks=RunIf(min_gpus=1)),
    ],
)
def test_lite_dataloader_device_placement(src_device, dest_device):
    """Test that the LiteDataLoader moves data to the device in its iterator."""
    sample0 = torch.tensor(0, device=src_device)
    sample1 = torch.tensor(1, device=src_device)
    sample2 = {"data": torch.tensor(2, device=src_device)}
    sample3 = {"data": torch.tensor(3, device=src_device)}
    data = [sample0, sample1, sample2, sample3]
    lite_dataloader = _LiteDataLoader(device=dest_device, dataset=data, batch_size=2)
    iterator = iter(lite_dataloader)

    batch0 = next(iterator)
    assert torch.equal(batch0, torch.tensor([0, 1], device=dest_device))

    batch1 = next(iterator)
    assert torch.equal(batch1["data"], torch.tensor([2, 3], device=dest_device))


def test_lite_optimizer_wraps():
    """Test that the LiteOptimizer fully wraps the optimizer."""
    optimizer_cls = torch.optim.SGD
    optimizer = Mock(spec=optimizer_cls)
    lite_optimizer = _LiteOptimizer(optimizer, Mock())
    assert lite_optimizer.optimizer is optimizer
    assert isinstance(lite_optimizer, optimizer_cls)


def test_lite_optimizer_steps():
    """Test that the LiteOptimizer forwards the step() and zero_grad() calls to the wrapped optimizer."""
    optimizer = Mock()
    accelerator = Mock()
    lite_optimizer = _LiteOptimizer(optimizer=optimizer, accelerator=accelerator)
    lite_optimizer.step()
    accelerator.optimizer_step.assert_called_once()
    accelerator.optimizer_step.assert_called_with(optimizer, opt_idx=0, lambda_closure=ANY, model=accelerator.model)
    lite_optimizer.zero_grad()
    optimizer.zero_grad.assert_called_once()
