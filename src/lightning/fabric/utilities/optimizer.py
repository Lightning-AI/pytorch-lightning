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

import contextlib
from typing import Iterable

from torch.optim import Optimizer

from lightning.fabric.utilities.apply_func import move_data_to_device
from lightning.fabric.utilities.types import _DEVICE


def _optimizers_to_device(optimizers: Iterable[Optimizer], device: _DEVICE) -> None:
    """Moves optimizer states for a sequence of optimizers to the device."""
    for opt in optimizers:
        _optimizer_to_device(opt, device)


def _optimizer_to_device_deprecated(optimizer: Optimizer, device: _DEVICE) -> None:
    """Moves the state of a single optimizer to the device."""
    """Deprecated because it seems this is unnecessary."""
    # Note special logic for 'step' parameter
    # The 'step' parameter needs to remain unmoved (possibly on the CPU) since that is where the optimizer needs it.
    # See https://github.com/pytorch/pytorch/issues/74424 and
    # _process_value_according_to_param_policy in torch/optim/optimizer.py:618
    fused = False
    with contextlib.suppress(Exception):
        fused = optimizer.param_groups[0]["fused"]

    for p, v in optimizer.state.items():
        for key, val in v.items():
            if key != "step" or fused:
                v[key] = move_data_to_device(val, device)


def _optimizer_to_device(optimizer: Optimizer, device: _DEVICE) -> None:
    """Moves the state of a single optimizer to the device.

    In fact, it looks like we dont need this function but can rely on optimizer.load_state_dict to do the right thing
    after given a correct prototype on the target device. For now we do nothing and assume that we don't care about
    transferring the optimizer back to the CPU on teardown. See details below.

    """
    pass

    # To test for correct behaviour here we have created two tests:
    # 1. tests/tests_fabric/utilities/test_optimizer.py to test Optimizer.load_state_dict with a prototype
    # 2. tests/tests_pytorch/checkpointing/test_trainer_move_device.py to test higher level checkpointing on
    #    one device and resuming on a different device

    # Details on how this function is called.
    # 1st call is in Strategy.setup(), to initialize empty optimizer. src/lightning/pytorch/strategies/strategy.py: 158
    # Note: Strategy.setup() first calls Strategy.setup_optimizers which eventually invokes Model.configure_optimizers()
    # based on a model that has been moved to the device. Thus it essentially creates a prototype optimizer on the
    # target device and then, eventually, relies on Optimizer.load_state_dict() to transfer the state.
    # 2nd call when restoring checkpoint, as part of Strategy.load_optimizer_state_dict(). Source strategy.py: 377
    # Final call in Strategy.teardown(), move optimizer back to CPU. Source strategy.py: 525
