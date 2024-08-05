import dataclasses

import pytest
import torch
from lightning.fabric.utilities.optimizer import _optimizer_to_device
from torch import Tensor

from tests_fabric.helpers.runif import RunIf


@pytest.mark.parametrize(
    "optimizer_class",
    [
        torch.optim.Adam,
        torch.optim.AdamW,
        torch.optim.SGD,
        torch.optim.RMSprop,
        torch.optim.Adagrad,
        torch.optim.Adadelta,
        torch.optim.Adamax,
    ],
)
@pytest.mark.parametrize(
    "src_device",
    [
        torch.device("cpu"),
        pytest.param(torch.device("cuda"), marks=RunIf(min_cuda_gpus=1)),
    ],
)
@pytest.mark.parametrize(
    "dst_device",
    [
        torch.device("cpu"),
        pytest.param(torch.device("cuda"), marks=RunIf(min_cuda_gpus=1)),
    ],
)
def test_optimizer_to_device(optimizer_class, src_device, dst_device):
    # Optimizer with no state initialized
    model = torch.nn.Linear(2, 2, device=src_device)
    optimizer = optimizer_class(model.parameters(), lr=0.1)
    _optimizer_to_device(optimizer, dst_device)
    _assert_opt_parameters_on_device(optimizer, dst_device)

    # Optimizer with state initialized
    model = torch.nn.Linear(2, 2, device=src_device)
    optimizer = optimizer_class(model.parameters(), lr=0.1)
    model(torch.randn(2, 2, device=src_device)).sum().backward()
    optimizer.step()
    _optimizer_to_device(optimizer, dst_device)
    _assert_opt_parameters_on_device(optimizer, dst_device)


def _assert_opt_parameters_on_device(opt, device):
    for _, v in opt.state.items():
        for key, item in v.items():
            if not isinstance(item, Tensor):
                continue
            if key == "step":
                # The "step" tensor needs to remain on CPU
                assert item.device.type == "cpu"
            else:
                assert item.device.type == device.type


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("frozen", [True, False])
def test_optimizer_to_device_with_dataclass_in_state(frozen):
    src_device = torch.device("cpu")
    dst_device = torch.device("cuda")
    model = torch.nn.Linear(32, 2, device=src_device)

    @dataclasses.dataclass(frozen=frozen)
    class FooState:
        integer: int
        tensor: Tensor

    class TestOptimizer(torch.optim.SGD):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.state[model.weight] = {"dummy": torch.tensor(0)}
            self.state[model.bias] = FooState(0, torch.tensor(0))

    optimizer = TestOptimizer(model.parameters(), lr=0.1)
    _optimizer_to_device(optimizer, dst_device)
    assert optimizer.state[model.weight]["dummy"].device.type == dst_device.type
    assert optimizer.state[model.bias].tensor.device.type == ("cpu" if frozen else dst_device.type)
