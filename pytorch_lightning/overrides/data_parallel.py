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

import itertools
import numbers
import warnings
from typing import Any

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.warnings import WarningCache


def _find_tensors(obj):  # pragma: no-cover
    r"""
    Recursively find all tensors contained in the specified object.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []


warning_cache = WarningCache()


class LightningDataParallel(DataParallel):

    def __init__(self, module: LightningModule, *args, **kwargs):
        warnings.warn(
            "The usage of `LightningDataParallel` is deprecated since v1.2 and will be removed in v1.4."
            " From now on we recommend to directly subclass `torch.nn.parallel.DataParallel`.",
            DeprecationWarning
        )
        super().__init__(LightningParallelModule(module), *args, **kwargs)


class LightningDistributedDataParallel(DistributedDataParallel):

    def __init__(self, module: LightningModule, *args, **kwargs):
        warnings.warn(
            "The usage of `LightningDistributedDataParallel` is deprecated since v1.2 and will be removed in v1.4."
            " From now on we recommend to directly subclass `torch.nn.parallel.DistributedDataParallel`.",
            DeprecationWarning
        )
        super().__init__(LightningDistributedModule(module), *args, **kwargs)


class _LightningModuleWrapperBase(torch.nn.Module):

    def __init__(self, pl_module: LightningModule):
        """
        Wraps the user's LightningModule and redirects the forward call to the appropriate
        method, either ``training_step``, ``validation_step`` or ``test_step``.
        If the LightningModule is in none of the states `training`, `testing` or `validation`,
        the inputs will be redirected to the
        :meth:`~pytorch_lightning.core.lightning.LightningModule.predict` method.
        Inheriting classes may also modify the inputs or outputs of forward.

        Args:
            pl_module: the model to wrap
        """
        super().__init__()
        self.module = pl_module

    def forward(self, *inputs, **kwargs):
        running_stage = self.module.running_stage

        if running_stage == RunningStage.TRAINING:
            output = self.module.training_step(*inputs, **kwargs)
            warn_if_output_is_none(output, "training_step")
        elif running_stage == RunningStage.TESTING:
            output = self.module.test_step(*inputs, **kwargs)
            warn_if_output_is_none(output, "test_step")
        elif running_stage == RunningStage.EVALUATING:
            output = self.module.validation_step(*inputs, **kwargs)
            warn_if_output_is_none(output, "validation_step")
        else:
            output = self.module.predict(*inputs, **kwargs)

        return output


class LightningParallelModule(_LightningModuleWrapperBase):
    """
    Wraps the user's LightningModule and redirects the forward call to the appropriate
    method, either ``training_step``, ``validation_step``, ``test_step`` or ``predict``.
    This class is used in combination with :class:`~torch.nn.parallel.DataParallel` as
    shown in the example. It also takes care of converting Python scalars to Tensors and
    un-squeezes 0-dimensional Tensors as it is required by :class:`~torch.nn.parallel.DataParallel`.

    Example:

        dp_model = torch.nn.DataParallel(
            module=LightningParallelModule(lightning_module),
            device_ids=[3, 4],
            ...
        )

    Args:
        pl_module: the model to wrap

    """
    def __init__(self, pl_module: LightningModule):
        super().__init__(pl_module)

    def forward(self, *inputs, **kwargs):
        output = super().forward(*inputs, **kwargs)
        output = apply_to_collection(
            output,
            dtype=numbers.Number,
            function=python_scalar_to_tensor,
            device=self.module.device
        )
        output = apply_to_collection(
            output,
            dtype=torch.Tensor,
            function=unsqueeze_scalar_tensor,
        )
        return output


class LightningDistributedModule(_LightningModuleWrapperBase):

    def __init__(self, pl_module: LightningModule):
        """
        Wraps the user's LightningModule and redirects the forward call to the appropriate
        method, either ``training_step``, ``validation_step``, ``test_step`` or ``predict``.
        This class is used in combination with :class:`~torch.nn.parallel.DistributedDataParallel` as
        shown in the example.

        Example:

            ddp_model = torch.nn.parallel.DistributedDataParallel(
                module=LightningDistributedModule(lightning_module),
                device_ids=[local_rank],
                ...
            )

        Args:
            pl_module: the model to wrap

        """
        super().__init__(pl_module)

    def forward(self, *inputs, **kwargs):
        return super().forward(*inputs, **kwargs)


# In manual_optimization, we need to call reducer prepare_for_backward.
# Note: Keep track of Pytorch DDP and update if there is a change
# https://github.com/pytorch/pytorch/blob/v1.7.1/torch/nn/parallel/distributed.py#L626-L638
def prepare_for_backward(model: DistributedDataParallel, output: Any):
    if torch.is_grad_enabled() and model.require_backward_grad_sync:
        model.require_forward_param_sync = True
        # We'll return the output object verbatim since it is a freeform
        # object. We need to find any tensors in this object, though,
        # because we need to figure out which parameters were used during
        # this forward pass, to ensure we short circuit reduction for any
        # unused parameters. Only if `find_unused_parameters` is set.
        if model.find_unused_parameters:
            model.reducer.prepare_for_backward(list(_find_tensors(output)))
        else:
            model.reducer.prepare_for_backward([])
    else:
        model.require_forward_param_sync = False


def warn_if_output_is_none(output: Any, method_name: str) -> None:
    """ Warns user about which method returned None. """
    if output is None:
        warning_cache.warn(f'Your {method_name} returned None. Did you forget to return an output?')


def python_scalar_to_tensor(scalar: numbers.Number, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """ Converts a Python scalar number to a torch tensor and places it on the given device. """
    return torch.tensor([scalar], device=device)


def unsqueeze_scalar_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """ Un-squeezes a 0-dim tensor. """
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    return tensor
