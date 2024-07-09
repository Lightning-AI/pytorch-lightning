import collections
import copy
import dataclasses
from typing import Tuple

import pytest
import torch
import torch.nn as nn
from lightning.fabric.utilities.optimizer import _optimizer_to_device
from torch import Tensor
from torch.utils.data import DataLoader


def test_optimizer_to_device_match_locations():
    @dataclasses.dataclass(frozen=True)
    class FooState:
        bar: int

    class CachedRandomTensorDataset(torch.utils.data.Dataset):
        """Very low overhead torch dataset for training for a given number of steps."""

        def __init__(self, batch_size: int, num_features: int, num_responses: int, length: int, device: str) -> None:
            self.x = torch.randn((batch_size, num_features), device=torch.device(device))
            self.y = torch.randn((batch_size, num_responses), device=torch.device(device))
            self.length = length

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self.x.clone(), self.y.clone()

        def __len__(self) -> int:
            return self.length

    def simple_training(optimizer, model, dataset, loss_fn):
        for input, target in dataset:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

    if not torch.cuda.is_available():
        return

    gpu_device = "cuda"
    devices = ["cpu", gpu_device]

    num_features = 32
    num_responses = 2
    batch_size = 16

    optimizer_on_device = {}
    for device in devices:
        dataset = CachedRandomTensorDataset(batch_size, num_features, num_responses, batch_size * 16, device)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False)
        model = torch.nn.Linear(num_features, num_responses)
        model = model.to(device=device)
        fused_vals = [False] if device == "cpu" else [False, True]

        for fused in fused_vals:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1, fused=fused)
            simple_training(optimizer, model, dataloader, loss_fn=nn.MSELoss())
            if fused:
                optimizer_on_device[device + "_fused_" + str(fused)] = optimizer
            else:
                optimizer_on_device[device] = optimizer

    # Test _optimizer_to_device function
    # Test cpu-->gpu, fused = False from CPU
    opt_to_gpu = copy.deepcopy(optimizer_on_device["cpu"])
    _optimizer_to_device(opt_to_gpu, gpu_device)
    assert_opt_state_in_expected_location(opt_to_gpu, optimizer_on_device[gpu_device])

    # Test gpu-->cpu, fused = False
    opt_to_cpu = copy.deepcopy(optimizer_on_device[gpu_device])
    _optimizer_to_device(opt_to_cpu, "cpu")
    assert_opt_state_in_expected_location(opt_to_cpu, optimizer_on_device["cpu"])

    # Test gpu-->cpu, fused = True
    opt_to_cpu = copy.deepcopy(optimizer_on_device[gpu_device + "_fused_True"])
    _optimizer_to_device(opt_to_cpu, "cpu")
    assert_opt_state_in_expected_location(opt_to_cpu, optimizer_on_device["cpu"])

    # Try from_dict
    # These all pretend that we have an appropriate prototype, I don't think we can actually do this since
    # all we may have is a CPU pickle

    # GPU prototypes
    # Use from_dict with gpu prototype, fused = False
    opt_cpu_dict = optimizer_on_device["cpu"].state_dict()
    gpu_prototype = copy.deepcopy(optimizer_on_device[gpu_device])
    gpu_prototype.load_state_dict(opt_cpu_dict)
    assert_opt_state_in_expected_location(gpu_prototype, optimizer_on_device[gpu_device])

    # Use from_dict with gpu prototype, fused = True
    opt_cpu_dict = optimizer_on_device["cpu"].state_dict()
    gpu_prototype = copy.deepcopy(optimizer_on_device[gpu_device + "_fused_True"])
    gpu_prototype.load_state_dict(opt_cpu_dict)
    assert_opt_state_in_expected_location(
        gpu_prototype, optimizer_on_device[gpu_device]
    )  # fused=False from CPU, overrides prototype

    # CPU prototypes
    # Use from_dict with cpu prototype, fused = False
    opt_gpu_dict = optimizer_on_device[gpu_device].state_dict()
    cpu_prototype = copy.deepcopy(optimizer_on_device["cpu"])
    cpu_prototype.load_state_dict(opt_gpu_dict)
    assert_opt_state_in_expected_location(cpu_prototype, optimizer_on_device["cpu"])

    # Use from_dict with cpu prototype, fused = True
    opt_gpu_dict = optimizer_on_device[gpu_device + "_fused_True"].state_dict()
    cpu_prototype = copy.deepcopy(optimizer_on_device["cpu"])
    cpu_prototype.load_state_dict(opt_gpu_dict)  # This should give an error / refuse to allow fused = True
    assert_opt_state_in_expected_location(cpu_prototype, optimizer_on_device["cpu"])


def assert_opt_state_in_expected_location(opt, expected_opt):
    opt_dict = opt.state_dict()
    expected_opt_dict = expected_opt.state_dict()
    for key, param in opt_dict["state"].items():
        if isinstance(param, Tensor) and param.data.device.type != expected_opt_dict["state"][key].device.type:
            pytest.fail(f"Optimizer device mismatch for state[{key}]")
        elif isinstance(param, collections.abc.Mapping):
            for subkey, subparam in param.items():
                if (
                    isinstance(subparam, Tensor)
                    and subparam.data.device.type != expected_opt_dict["state"][key][subkey].device.type
                ):
                    pytest.fail(f"Optimizer device mismatch for state[{key}][{subkey}]")
