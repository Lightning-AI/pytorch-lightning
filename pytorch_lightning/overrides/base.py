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
from typing import Any, Union

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

import pytorch_lightning as pl
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin


class _LightningPrecisionModuleWrapperBase(DeviceDtypeModuleMixin, torch.nn.Module):

    def __init__(self, pl_module: 'pl.LightningModule') -> None:
        """
        Wraps the user's LightningModule. Requires overriding all ``*_step`` methods and ``forward`` so that it can
        safely be wrapped by a ``_LightningModuleWrapperBase`` and a ``*DataParallel``.

        Args:
            pl_module: the model to wrap
        """
        super().__init__()
        self.module = pl_module

        # set the parameters_to_ignore from LightningModule.
        self._ddp_params_and_buffers_to_ignore = getattr(pl_module, "_ddp_params_and_buffers_to_ignore", [])

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def validation_step(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def on_post_move_to_device(self) -> None:
        pass


class _LightningModuleWrapperBase(DeviceDtypeModuleMixin, torch.nn.Module):

    def __init__(self, pl_module: Union['pl.LightningModule', _LightningPrecisionModuleWrapperBase]):
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

        # set the parameters_to_ignore from LightningModule.
        self._ddp_params_and_buffers_to_ignore = getattr(pl_module, "_ddp_params_and_buffers_to_ignore", [])

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        lightning_module = unwrap_lightning_module(self.module)
        trainer = lightning_module.trainer

        if trainer and trainer.training:
            output = self.module.training_step(*inputs, **kwargs)

            # In manual_optimization, we need to prevent DDP reducer as
            # it is done manually in ``LightningModule.manual_backward``
            # `require_backward_grad_sync` will be reset in the
            # ddp_plugin ``post_training_step`` hook
            if not lightning_module.automatic_optimization:
                trainer.model.require_backward_grad_sync = False
        elif trainer and trainer.testing:
            output = self.module.test_step(*inputs, **kwargs)
        elif trainer and (trainer.sanity_checking or trainer.validating):
            output = self.module.validation_step(*inputs, **kwargs)
        elif trainer and trainer.predicting:
            output = self.module.predict_step(*inputs, **kwargs)
        else:
            output = self.module(*inputs, **kwargs)

        return output

    def on_post_move_to_device(self) -> None:
        pass


def unwrap_lightning_module(wrapped_model) -> 'pl.LightningModule':
    model = wrapped_model
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        model = unwrap_lightning_module(model.module)
    if isinstance(model, (_LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase)):
        model = unwrap_lightning_module(model.module)
    return model
