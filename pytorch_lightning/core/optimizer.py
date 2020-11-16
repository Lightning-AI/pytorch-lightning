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
import collections
import copy
import inspect
import os
import re
import tempfile
import types
from abc import ABC
from argparse import Namespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import ScriptModule, Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.grads import GradInformation
from pytorch_lightning.core.hooks import CheckpointHooks, DataHooks, ModelHooks
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.core.saving import ALLOWED_CONFIG_TYPES, PRIMITIVE_TYPES, ModelIO
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.utilities import AMPType, rank_zero_warn
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import AttributeDict, collect_init_args, get_init_args
from pytorch_lightning.utilities.xla_device_utils import XLADeviceUtils

TPU_AVAILABLE = XLADeviceUtils.tpu_device_exists()

if TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm


def do_nothing_closure():
    return


class LightningOptimizer(Optimizer):

    """
    This class is used to wrap the user optimizers and handle properly
    the backward and optimizer_step logic across accelerators, AMP, accumulated_grad_batches
    """

    def __init__(self,
                 trainer,
                 optimizer: Optimizer = None,
                 optimizer_idx: Optional[int] = None,
                 accumulate_grad_batches: Optional[int] = None):

        if isinstance(accumulate_grad_batches, int) and accumulate_grad_batches >= 1:
            raise MisconfigurationException(f"accumulate_grad_batches parameters "
                                            f"{accumulate_grad_batches} should be >= 1")

        self._trainer = trainer
        self._optimizer = optimizer
        self._optimizer_idx = optimizer_idx
        if accumulate_grad_batches is None:
            self._accumulate_grad_batches = trainer.accumulate_grad_batches
        else:
            self._accumulate_grad_batches = accumulate_grad_batches
        self._expose_optimizer_attr()

    @property
    def accumulate_grad_batches(self):
        return self._accumulate_grad_batches

    @accumulate_grad_batches.setter
    def accumulate_grad_batches(self, accumulate_grad_batches):
        self._accumulate_grad_batches = accumulate_grad_batches

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        return self._optimizer

    def _expose_optimizer_attr(self):
        for attr_name in dir(self._optimizer):
            if ('__' in attr_name) or "step" == attr_name:
                continue
            setattr(self, attr_name, getattr(self._optimizer, attr_name))

    def __getstate__(self):
        return {
            'defaults': self._optimizer.defaults,
            'state': self._optimizer.state,
            'param_groups': self._optimizer.param_groups,
            'optimizer_cls': self._optimizer.__class__,
            'optimizer_idx': self._optimizer_idx,
            "accumulate_grad_batches": self._accumulate_grad_batches,
        }

    def __setstate__(self, state):
        self._optimizer_idx = state["optimizer_idx"]
        self._accumulate_grad_batches = state["accumulate_grad_batches"]
        self._optimizer = state["optimizer_cls"](state['param_groups'], ** state['defaults'])
        self._expose_optimizer_attr()

    def __repr__(self):
        if hasattr(self, "_optimizer"):
            format_string = "Lightning" + self._optimizer.__class__.__name__ + ' ('
            for i, group in enumerate(self.param_groups):
                format_string += '\n'
                format_string += 'Parameter Group {0}\n'.format(i)
                for key in sorted(group.keys()):
                    if key != 'params':
                        format_string += '    {0}: {1}\n'.format(key, group[key])
            format_string += ')'
            return format_string
        else:
            return self.__class__.__name__

    def _accumulated_batches_reached(self):
        return (self._trainer.batch_idx + 1) % self._accumulate_grad_batches == 0

    def _num_training_batches_reached(self):
        return (self._trainer.batch_idx + 1) == self._trainer.num_training_batches

    @property
    def _should_accumulate(self):
        # checks if backward or backward + optimizer step (via closure)
        accumulation_done = self._accumulated_batches_reached()
        is_final_batch = self._num_training_batches_reached()
        return not (accumulation_done or is_final_batch)

    def backward(self, loss: Tensor, *args, **kwargs) -> None:
        """
        Call this directly from your training_step when doing optimizations manually.
        By using this we can ensure that all the proper scaling when using 16-bit etc has been done for you

        .. tip:: In manual mode we still automatically accumulate grad over batches if
           Trainer(accumulate_grad_batches=x) is set.

        Args:
            loss: Optimizer used to perform `.step()` call

        Example::

            def training_step(...):
                (opt_a, opt_b) = self.optimizers()
                loss_a = ...

                # automatically applies scaling, etc...
                opt_a.backward(loss_a)
                opt_a.step()

        Example::

            def training_step(...):
                (opt_a, opt_b) = self.optimizers()
                loss_a = ...

                # automatically applies scaling, etc...
                def closure_a():
                    loss_a = ...
                    opt_a.backward(loss)

                opt_a.step(closure=closure_a)

        """

        model_ref = self._trainer.get_model()

        # toggle params
        model_ref.toggle_optimizer(self, self._optimizer_idx)

        # perform manual_backward
        model_ref.manual_backward(loss, self, *args, **kwargs)

    def step(self, *args, closure: Callable = do_nothing_closure, **kwargs):
        """
        Call this directly from your training_step when doing optimizations manually.
        By using this we can ensure that all the proper scaling when using 16-bit etc has been done for you

        .. tip:: In manual mode we still automatically accumulate grad over batches if
           Trainer(accumulate_grad_batches=x) is set.

        Args:
            closure: Closure should contain forward and backward step
        """

        if not self._should_accumulate:
            if self._trainer.on_tpu:
                xm.optimizer_step(self._optimizer, optimizer_args={'closure': closure, **kwargs})
            elif self._trainer.amp_backend == AMPType.NATIVE:
                # native amp does not yet support closures.
                # TODO: pass the closure to the step ASAP
                closure()
                self._trainer.scaler.step(self._optimizer)
                self._trainer.scaler.update()
            elif self._trainer.amp_backend == AMPType.APEX:
                # apex amp does not yet support closures.
                # TODO: pass the closure to the step ASAP
                closure()
                self._optimizer.step()
            else:
                self._optimizer.step(closure=closure, *args, **kwargs)

            # perform zero grad
            self._optimizer.zero_grad()
        else:
            # make sure to call optimizer_closure when accumulating
            if isinstance(closure, types.FunctionType):
                closure()
