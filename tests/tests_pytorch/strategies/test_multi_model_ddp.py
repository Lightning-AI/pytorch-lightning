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
from unittest import mock
from unittest.mock import PropertyMock

import torch
from torch import nn

from lightning.pytorch.strategies.ddp import MultiModelDDPStrategy


def test_multi_model_ddp_setup_and_register_hooks():
    class Parent(nn.Module):
        def __init__(self):
            super().__init__()
            self.gen = nn.Linear(1, 1)
            self.dis = nn.Linear(1, 1)

    model = Parent()
    original_children = [model.gen, model.dis]

    strategy = MultiModelDDPStrategy(parallel_devices=[torch.device("cpu")])

    wrapped_modules = []
    wrapped_device_ids = []

    class DummyDDP(nn.Module):
        def __init__(self, module: nn.Module, device_ids=None, **kwargs):
            super().__init__()
            self.module = module
            wrapped_modules.append(module)
            wrapped_device_ids.append(device_ids)

    with mock.patch("lightning.pytorch.strategies.ddp.DistributedDataParallel", DummyDDP):
        returned_model = strategy._setup_model(model)
        assert returned_model is model
        assert isinstance(model.gen, DummyDDP)
        assert isinstance(model.dis, DummyDDP)
        assert wrapped_modules == original_children
        assert wrapped_device_ids == [None, None]

        strategy.model = model
        with (
            mock.patch("lightning.pytorch.strategies.ddp._register_ddp_comm_hook") as register_hook,
            mock.patch.object(MultiModelDDPStrategy, "root_device", new_callable=PropertyMock) as root_device,
        ):
            root_device.return_value = torch.device("cuda", 0)
            strategy._register_ddp_hooks()

        assert register_hook.call_count == 2
        register_hook.assert_any_call(
            model=model.gen,
            ddp_comm_state=strategy._ddp_comm_state,
            ddp_comm_hook=strategy._ddp_comm_hook,
            ddp_comm_wrapper=strategy._ddp_comm_wrapper,
        )
        register_hook.assert_any_call(
            model=model.dis,
            ddp_comm_state=strategy._ddp_comm_state,
            ddp_comm_hook=strategy._ddp_comm_hook,
            ddp_comm_wrapper=strategy._ddp_comm_wrapper,
        )


def test_multi_model_ddp_register_hooks_cpu_noop():
    class Parent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gen = nn.Linear(1, 1)
            self.dis = nn.Linear(1, 1)

    model = Parent()
    strategy = MultiModelDDPStrategy(parallel_devices=[torch.device("cpu")])

    class DummyDDP(nn.Module):
        def __init__(self, module: nn.Module, device_ids=None, **kwargs):
            super().__init__()
            self.module = module

    with mock.patch("lightning.pytorch.strategies.ddp.DistributedDataParallel", DummyDDP):
        strategy.model = strategy._setup_model(model)

    with mock.patch("lightning.pytorch.strategies.ddp._register_ddp_comm_hook") as register_hook:
        strategy._register_ddp_hooks()

    register_hook.assert_not_called()
