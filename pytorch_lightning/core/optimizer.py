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
import os
import tempfile
import collections
import copy
import inspect
import re
import types
from abc import ABC
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Mapping
from torch.optim.optimizer import Optimizer

import torch
from pytorch_lightning import _logger as log
from pytorch_lightning.core.grads import GradInformation
from pytorch_lightning.core.hooks import CheckpointHooks, DataHooks, ModelHooks
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.core.saving import ALLOWED_CONFIG_TYPES, PRIMITIVE_TYPES, ModelIO
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.utilities import rank_zero_warn, AMPType
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin
from pytorch_lightning.utilities.xla_device_utils import XLADeviceUtils
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.parsing import (
    AttributeDict,
    collect_init_args,
    get_init_args,
)
from pytorch_lightning.callbacks import Callback
from torch import ScriptModule, Tensor
from torch.nn import Module


TPU_AVAILABLE = XLADeviceUtils.tpu_device_exists()

if TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm


def do_nothing_closure():
    return


class LightningOptimizer(Optimizer):

    def __init__(self, trainer, optimizer: Optimizer = None, optimizer_idx: Optional[int] = None):
        self._trainer = trainer
        self._optimizer = optimizer
        self._optimizer_idx = optimizer_idx
        self._expose_optimizer_attr()

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
        }

    def __setstate__(self, state):
        self._optimizer_idx = state["optimizer_idx"]
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

    def step(self, *args, closure=do_nothing_closure, **kwargs):
        if self._trainer.on_tpu:
            xm.optimizer_step(self._optimizer, optimizer_args={'closure': closure, **kwargs})
        elif self._trainer.amp_backend == AMPType.NATIVE:
            # native amp does not yet support closures.
            # TODO: pass the closure to the step ASAP
            closure()
            self._trainer.scaler.step(self._optimizer)
        elif self._trainer.amp_backend == AMPType.APEX:
            # apex amp does not yet support closures.
            # TODO: pass the closure to the step ASAP
            closure()
            self._optimizer.step(*args, **kwargs)
        else:
            self._optimizer.step(closure=closure, *args, **kwargs)
