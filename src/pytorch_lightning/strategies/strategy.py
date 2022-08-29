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
import contextlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Tuple, TypeVar, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import _init_optimizers_and_lr_schedulers, LightningOptimizer
from pytorch_lightning.plugins import TorchCheckpointIO
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.io.wrapper import _WrappingCheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.launchers.base import _Launcher
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.distributed import ReduceOp
from pytorch_lightning.utilities.optimizer import optimizer_to_device, optimizers_to_device
from pytorch_lightning.utilities.types import (
    _PATH,
    LRSchedulerConfig,
    PredictStep,
    STEP_OUTPUT,
    TestStep,
    TrainingStep,
    ValidationStep,
)

TBroadcast = TypeVar("TBroadcast")
TReduce = TypeVar("TReduce")

log = logging.getLogger(__name__)


class Strategy(ABC):
    """Base class for all strategies that change the behaviour of the training, validation and test- loop."""

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ) -> None:
        self._accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = accelerator
        self._checkpoint_io: Optional[CheckpointIO] = checkpoint_io
        self._precision_plugin: Optional[PrecisionPlugin] = precision_plugin
