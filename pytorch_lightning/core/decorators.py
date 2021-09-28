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
"""Decorator for LightningModule methods."""

from functools import wraps
from typing import Callable, Dict, Optional

from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.utilities import rank_zero_deprecation
from pytorch_lightning.utilities.model_helpers import is_overridden


def auto_weight_tying(model_to_device: Callable) -> Callable:
    """Enables auto parameters tying on TPUs.

    Args:
        model_to_device: ``TrainingTypePlugin.model_to_device`` method

    Note:
        TPU's require weights to be tied/shared after moving the module to the device.
        Failure to do this results in the initialization of new weights which are not tied.
        We apply auto parameters tying after the module has been moved to the device.

    See Also:
        - `XLA Documentation <https://github.com/pytorch/xla/blob/master/TROUBLESHOOTING.md#xla-tensor-quirks>`_
    """

    @wraps(model_to_device)
    def inner_fn(self, *args, **kwargs):
        shared_params = find_shared_parameters(self.model)
        model_to_device(self, *args, **kwargs)
        module = self.model.module if isinstance(self.model, LightningDistributedModule) else self.model
        if is_overridden("on_post_move_to_device", self.lightning_module):
            rank_zero_deprecation(
                "Method `on_post_move_to_device` has been deprecated and will be removed in v1.7.0."
                " We perform auto parameters tying without the need of implementing `on_post_move_to_device`"
            )
            module.on_post_move_to_device()
        else:
            apply_weight_tying(module, shared_params)

    return inner_fn


def find_shared_parameters(module, tied_parameters: Optional[Dict] = None, prefix: str = ""):
    if tied_parameters is None:
        first_call = True
        tied_parameters = {}
    else:
        first_call = False
    for name, param in module._parameters.items():
        param_prefix = prefix + ("." if prefix else "") + name
        if param is None:
            continue
        if param not in tied_parameters:
            tied_parameters[param] = []
        tied_parameters[param].append(param_prefix)
    for name, m in module._modules.items():
        if m is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        find_shared_parameters(m, tied_parameters, submodule_prefix)
    if first_call:
        return [x for x in tied_parameters.values() if len(x) > 1]


def apply_weight_tying(module, shared_params):
    for shared_param in shared_params:
        ref = _get_module_by_path(module, shared_param[0])
        for path in shared_param[1:]:
            _set_module_by_path(module, path, ref)
    return module


def _get_module_by_path(module, path):
    path = path.split(".")
    for name in path:
        module = getattr(module, name)
    return module


def _set_module_by_path(module, path, value):
    path = path.split(".")
    for name in path[:-1]:
        module = getattr(module, name)
    setattr(module, path[-1], value)
